"""CIF/POSCAR structures to the graph inputs required by DBGFN.

The learned MACE and ALIGNN embeddings are checkpoint-dependent and are
therefore created inside :mod:`model`. This module prepares the two graph
representations consumed by that model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data


MACE_CUTOFF = 6.0
ALIGNN_RADIUS = 8.0
ALIGNN_MAX_NEIGHBORS = 12
ALIGNN_DISTANCE_BINS = 80
ALIGNN_ANGLE_BINS = 40


class RBFExpansion:
    """Gaussian radial-basis expansion used for distances and bond angles."""

    def __init__(self, vmin: float, vmax: float, bins: int):
        self.centers = torch.linspace(vmin, vmax, bins)
        self.width = (vmax - vmin) / bins

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -((values.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2)
            / (2 * self.width**2)
        )


class DualGraphData(Data):
    """A PyG graph containing MACE fields and ALIGNN crystal/line graphs."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "alignn_edge_index":
            return self.num_nodes
        if key == "alignn_line_edge_index":
            return self.alignn_edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("alignn_edge_index", "alignn_line_edge_index"):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def _atom_onehot(atomic_number: int, max_atomic_number: int = 94) -> torch.Tensor:
    feature = torch.zeros(max_atomic_number)
    if 1 <= atomic_number <= max_atomic_number:
        feature[atomic_number - 1] = 1.0
    return feature


def _build_alignn_graph(
    structure: Structure,
    distance_rbf: RBFExpansion,
    angle_rbf: RBFExpansion,
) -> Data:
    """Create the atom graph and line graph used by the ALIGNN branch."""

    n_atoms = len(structure)
    atom_features = torch.stack(
        [_atom_onehot(site.specie.Z) for site in structure]
    )

    sources, destinations, vectors = [], [], []
    for source, neighbors in enumerate(
        structure.get_all_neighbors(ALIGNN_RADIUS, include_index=True)
    ):
        for neighbor in sorted(neighbors, key=lambda item: item[1])[
            :ALIGNN_MAX_NEIGHBORS
        ]:
            destination = neighbor[2]
            sources.append(source)
            destinations.append(destination)
            vectors.append(neighbor[0].coords - structure[source].coords)

    if not sources:
        raise ValueError(
            f"No neighbors found within {ALIGNN_RADIUS} Å for the input structure."
        )

    edge_index = torch.tensor([sources, destinations], dtype=torch.long)
    edge_vectors = torch.tensor(np.asarray(vectors), dtype=torch.float32)
    edge_attributes = distance_rbf(edge_vectors.norm(dim=1))

    edge_sources, edge_destinations = edge_index
    line_sources, line_destinations = [], []
    for center in range(n_atoms):
        incoming = (edge_destinations == center).nonzero(as_tuple=False).flatten()
        outgoing = (edge_sources == center).nonzero(as_tuple=False).flatten()
        if incoming.numel() and outgoing.numel():
            line_sources.append(incoming.repeat_interleave(outgoing.numel()))
            line_destinations.append(outgoing.repeat(incoming.numel()))

    if line_sources:
        line_source = torch.cat(line_sources)
        line_destination = torch.cat(line_destinations)
    else:
        line_source = torch.empty(0, dtype=torch.long)
        line_destination = torch.empty(0, dtype=torch.long)
    line_edge_index = torch.stack([line_source, line_destination])

    if line_source.numel():
        incoming_vectors = edge_vectors[line_source]
        outgoing_vectors = edge_vectors[line_destination]
        cosine = (-incoming_vectors * outgoing_vectors).sum(dim=1) / (
            incoming_vectors.norm(dim=1) * outgoing_vectors.norm(dim=1) + 1e-8
        )
        line_attributes = angle_rbf(cosine.clamp(-1.0, 1.0))
    else:
        line_attributes = torch.zeros(0, ALIGNN_ANGLE_BINS)

    return Data(
        x=atom_features,
        edge_index=edge_index,
        edge_attr=edge_attributes,
        line_edge_index=line_edge_index,
        line_attr=line_attributes,
    )


def _build_mace_graph(atoms, atomic_number_table) -> DualGraphData:
    """Create the MACE graph fields for one periodic structure."""

    from mace import data as mace_data

    key_specification = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    configuration = mace_data.config_from_atoms(
        atoms, key_specification=key_specification
    )
    atomic_data = mace_data.AtomicData.from_config(
        configuration,
        z_table=atomic_number_table,
        cutoff=MACE_CUTOFF,
        heads=["Default"],
    )
    fields = atomic_data.to_dict()
    if "pbc" not in fields:
        fields["pbc"] = torch.ones(3, dtype=torch.bool)

    graph = DualGraphData()
    for key in (
        "positions",
        "edge_index",
        "shifts",
        "unit_shifts",
        "cell",
        "pbc",
        "weight",
        "energy_weight",
        "forces_weight",
    ):
        if key in fields:
            graph[key] = fields[key]
    if "head" in fields:
        # ``prepare_graph`` indexes heads with the per-atom batch vector, so
        # a single-structure graph requires a one-element head tensor rather
        # than the scalar emitted by ``AtomicData``.
        head = fields["head"]
        graph.head = head.unsqueeze(0) if head.dim() == 0 else head
    graph.mace_node_attrs = fields["node_attrs"]
    graph.num_nodes = fields["positions"].shape[0]
    return graph


def _attach_alignn_graph(
    graph: DualGraphData,
    structure: Structure,
    distance_rbf: RBFExpansion,
    angle_rbf: RBFExpansion,
) -> DualGraphData:
    alignn_graph = _build_alignn_graph(structure, distance_rbf, angle_rbf)
    graph.alignn_x = alignn_graph.x
    graph.alignn_edge_index = alignn_graph.edge_index
    graph.alignn_edge_attr = alignn_graph.edge_attr
    graph.alignn_line_edge_index = alignn_graph.line_edge_index
    graph.alignn_line_attr = alignn_graph.line_attr
    return graph


def _single_graph_batch(graph: DualGraphData) -> DualGraphData:
    """Add the batch fields expected by PyG and MACE for one structure."""

    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph.ptr = torch.tensor([0, graph.num_nodes], dtype=torch.long)
    n_alignn_nodes = graph.alignn_x.shape[0]
    graph.alignn_batch = torch.zeros(n_alignn_nodes, dtype=torch.long)
    graph.alignn_ptr = torch.tensor([0, n_alignn_nodes], dtype=torch.long)
    return graph


def structure_to_model_input(
    structure: Structure,
    atomic_number_table,
) -> DualGraphData:
    """Convert one pymatgen structure to a batched DBGFN model input."""

    atoms = AseAtomsAdaptor.get_atoms(structure)
    distance_rbf = RBFExpansion(0.0, ALIGNN_RADIUS, ALIGNN_DISTANCE_BINS)
    angle_rbf = RBFExpansion(-1.0, 1.0, ALIGNN_ANGLE_BINS)
    graph = _build_mace_graph(atoms, atomic_number_table)
    graph = _attach_alignn_graph(graph, structure, distance_rbf, angle_rbf)
    return _single_graph_batch(graph)


def file_to_model_input(path: str | Path, atomic_number_table):
    """Read one structure file and return its identity, formula, and DBGFN input."""

    source = Path(path)
    structure = Structure.from_file(source)
    return (
        source.stem,
        structure.composition.reduced_formula,
        structure_to_model_input(structure, atomic_number_table),
    )


def find_structure_files(directory: str | Path) -> list[Path]:
    """Return supported structure files in one directory, without recursion."""

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {root}")
    files: set[Path] = set()
    for pattern in ("*.cif", "*.vasp", "POSCAR*", "*.poscar"):
        files.update(root.glob(pattern))
    return sorted(files)
