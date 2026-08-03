"""DBGFN checkpoint model and command-line predictor.

Run ``python 03-code/model.py INPUT_DIRECTORY OUTPUT.csv`` to load the five
released checkpoints and make ensemble predictions for CIF, VASP, or POSCAR
structures. Graph preparation lives in :mod:`data`.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch_geometric.utils import scatter

from data import file_to_model_input, find_structure_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_FOLDS = 5
METAL_BAND_GAP_THRESHOLD = 0.05
SCREEN_BAND_GAP_MIN = 0.5
SCREEN_BAND_GAP_MAX = 1.1


class MACEFeatureExtractor(nn.Module):
    """Return the concatenated node representations from MACE interactions."""

    def __init__(self, mace_model):
        super().__init__()
        self.node_embedding = mace_model.node_embedding
        self.radial_embedding = mace_model.radial_embedding
        self.spherical_harmonics = mace_model.spherical_harmonics
        self.interactions = mace_model.interactions
        self.products = mace_model.products
        self.atomic_numbers = mace_model.atomic_numbers
        self.n_interactions = len(self.interactions)
        self.per_layer_dim = 128  # MACE-MP-0 small
        self.out_dim = self.per_layer_dim * self.n_interactions

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        from mace.modules.utils import prepare_graph

        graph = prepare_graph(data)
        edge_attributes = self.spherical_harmonics(graph.vectors)
        edge_features = self.radial_embedding(
            graph.lengths,
            data["node_attrs"],
            data["edge_index"],
            self.atomic_numbers,
        )
        node_features = self.node_embedding(data["node_attrs"])
        layer_features = []
        for index, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            interaction_output = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_features,
                edge_attrs=edge_attributes,
                edge_feats=edge_features,
                edge_index=data["edge_index"],
                first_layer=index == 0,
            )
            node_features, skip_connection = interaction_output[:2]
            node_features = product(
                node_feats=node_features,
                sc=skip_connection,
                node_attrs=data["node_attrs"],
            )
            layer_features.append(node_features)
        return torch.cat(layer_features, dim=-1)


class AttentionPooling(nn.Module):
    """Multi-head attention pooling from atom features to crystal features."""

    def __init__(self, input_dim: int, n_heads: int = 8):
        super().__init__()
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, input_dim // 4),
                    nn.Softplus(),
                    nn.Linear(input_dim // 4, 1),
                )
                for _ in range(n_heads)
            ]
        )
        self.combine = nn.Linear(input_dim * n_heads, input_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        heads = []
        for gate in self.gates:
            scores = gate(atom_features)
            pooled = []
            for graph_index in range(num_graphs):
                mask = batch == graph_index
                features = atom_features[mask]
                weights = functional.softmax(scores[mask], dim=0)
                pooled.append((weights * features).sum(dim=0, keepdim=True))
            heads.append(torch.cat(pooled, dim=0))
        return self.combine(torch.cat(heads, dim=-1))


class BandGapHead(nn.Module):
    """Residual regression head used by the released DBGFN checkpoints."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        middle = hidden_dim // 2
        quarter = hidden_dim // 4
        self.fc1 = nn.Linear(hidden_dim, middle)
        self.bn1 = nn.BatchNorm1d(middle)
        self.fc2 = nn.Linear(middle, middle)
        self.bn2 = nn.BatchNorm1d(middle)
        self.fc3 = nn.Linear(middle, quarter)
        self.bn3 = nn.BatchNorm1d(quarter)
        self.fc_out = nn.Linear(quarter, 1)
        self.skip1 = nn.Linear(hidden_dim, middle)
        self.skip2 = nn.Linear(middle, quarter)
        self.drop = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.drop(functional.silu(self.bn1(self.fc1(features))))
        hidden = self.drop(functional.silu(self.bn2(self.fc2(hidden))))
        hidden = hidden + self.skip1(features)
        skip = self.skip2(hidden)
        hidden = self.drop(functional.silu(self.bn3(self.fc3(hidden))))
        return self.fc_out(hidden + skip)


class CosineClassifier(nn.Module):
    """Cosine-similarity classification head used for gap-type prediction."""

    def __init__(self, input_dim: int, n_classes: int, initial_temperature: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, input_dim))
        self.log_temp = nn.Parameter(torch.tensor(math.log(initial_temperature)))
        nn.init.xavier_normal_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        temperature = self.log_temp.exp().clamp(min=0.01, max=1.0)
        normalized_features = functional.normalize(features, dim=1)
        normalized_weights = functional.normalize(self.weight, dim=1)
        return functional.log_softmax(
            normalized_features @ normalized_weights.T / temperature,
            dim=1,
        )


class EdgeGatedConv(nn.Module):
    """ALIGNN edge-gated convolution that updates node and edge features."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        output_node_dim: int,
        output_edge_dim: int,
    ):
        super().__init__()
        input_dim = 2 * node_dim + edge_dim
        self.gate_fc = nn.Linear(input_dim, output_edge_dim)
        self.msg_fc = nn.Linear(input_dim, output_edge_dim)
        self.node_fc = nn.Linear(node_dim + output_edge_dim, output_node_dim)
        self.node_norm = nn.LayerNorm(output_node_dim)
        self.node_res = (
            nn.Linear(node_dim, output_node_dim, bias=False)
            if node_dim != output_node_dim
            else nn.Identity()
        )
        self.edge_res = (
            nn.Linear(edge_dim, output_edge_dim, bias=False)
            if edge_dim != output_edge_dim
            else nn.Identity()
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, destination = edge_index
        concatenated = torch.cat(
            [node_features[source], node_features[destination], edge_features],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate_fc(concatenated))
        message = functional.softplus(self.msg_fc(concatenated))
        updated_edges = self.edge_res(edge_features) + gate * message
        aggregate = scatter(
            updated_edges,
            destination,
            dim=0,
            dim_size=node_features.size(0),
            reduce="sum",
        )
        updated_nodes = self.node_norm(
            self.node_res(node_features)
            + functional.silu(
                self.node_fc(torch.cat([node_features, aggregate], dim=-1))
            )
        )
        return updated_nodes, updated_edges


class ALIGNNLayer(nn.Module):
    """Line-graph update followed by a crystal-graph update."""

    def __init__(self, node_dim: int, edge_dim: int, angle_dim: int):
        super().__init__()
        self.line_conv = EdgeGatedConv(edge_dim, angle_dim, edge_dim, angle_dim)
        self.crystal_conv = EdgeGatedConv(node_dim, edge_dim, node_dim, edge_dim)

    def forward(
        self,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        angle_features: torch.Tensor,
        crystal_edge_index: torch.Tensor,
        line_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bond_features, angle_features = self.line_conv(
            bond_features, line_edge_index, angle_features
        )
        atom_features, _ = self.crystal_conv(
            atom_features, crystal_edge_index, bond_features
        )
        return atom_features, bond_features, angle_features


class ALIGNNFeatureExtractor(nn.Module):
    """The ALIGNN encoder portion stored inside the fused DBGFN model."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_alignn_layers: int = 4,
        n_gcn_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.atom_proj = nn.Sequential(
            nn.Linear(94, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.bond_proj = nn.Sequential(
            nn.Linear(80, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.angle_proj = nn.Sequential(
            nn.Linear(40, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        self.alignn_layers = nn.ModuleList(
            [ALIGNNLayer(hidden_dim, hidden_dim, hidden_dim // 2)
             for _ in range(n_alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
             for _ in range(n_gcn_layers)]
        )
        self.drop = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attributes: torch.Tensor,
        line_edge_index: torch.Tensor,
        line_attributes: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        atom_features = self.atom_proj(atom_features)
        bond_features = self.bond_proj(edge_attributes)
        angle_features = self.angle_proj(line_attributes)
        for layer in self.alignn_layers:
            atom_features, bond_features, angle_features = layer(
                atom_features,
                bond_features,
                angle_features,
                edge_index,
                line_edge_index,
            )
        for layer in self.gcn_layers:
            atom_features, bond_features = layer(
                atom_features, edge_index, bond_features
            )
        return self.drop(
            scatter(
                atom_features,
                batch,
                dim=0,
                dim_size=num_graphs,
                reduce="mean",
            )
        )


class GatedFusion(nn.Module):
    """Fuse MACE and ALIGNN crystal representations with a learned gate."""

    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.LayerNorm(dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Kept for exact checkpoint compatibility; the released forward pass
        # uses the gated and projected branches above.
        self.residual = nn.Linear(dimension * 2, dimension)

    def forward(
        self, mace_features: torch.Tensor, alignn_features: torch.Tensor
    ) -> torch.Tensor:
        concatenated = torch.cat([mace_features, alignn_features], dim=1)
        gate = self.gate(concatenated)
        gated = gate * mace_features + (1 - gate) * alignn_features
        return gated + self.proj(concatenated)


class DBGFNModel(nn.Module):
    """Released dual-backbone graph-fusion architecture."""

    def __init__(
        self,
        mace_model,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        n_attention_heads: int = 8,
    ):
        super().__init__()
        self.register_buffer("cond_weight", torch.tensor(0.0))
        self.mace_backbone = MACEFeatureExtractor(mace_model)
        self.mace_pool = AttentionPooling(
            self.mace_backbone.out_dim, n_attention_heads
        )
        self.alignn_backbone = ALIGNNFeatureExtractor(dropout=dropout)
        if self.mace_backbone.out_dim != self.alignn_backbone.out_dim:
            raise ValueError(
                "Released checkpoints require MACE-MP-0 small "
                f"(MACE={self.mace_backbone.out_dim}, "
                f"ALIGNN={self.alignn_backbone.out_dim})."
            )
        self.fusion = GatedFusion(self.mace_backbone.out_dim, dropout * 0.5)
        self.proj = nn.Linear(self.mace_backbone.out_dim, hidden_dim)
        self.trunk_bn = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head_bg = BandGapHead(hidden_dim, dropout)
        self.head_eh = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 2),
            nn.LogSoftmax(dim=1),
        )
        self.gt_proj = nn.Linear(hidden_dim + 3, hidden_dim)
        self.gt_layers = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.gt_classifier = CosineClassifier(hidden_dim // 2, 2, 0.1)

    def set_cond_weight(self, value: float) -> None:
        self.cond_weight.fill_(min(max(value, 0.0), 1.0))

    def forward(self, data):
        batch = data.batch
        num_graphs = int(batch.max().item()) + 1
        mace_input = {
            "positions": data.positions,
            "node_attrs": data.mace_node_attrs,
            "edge_index": data.edge_index,
            "shifts": data.shifts,
            "unit_shifts": data.unit_shifts,
            "cell": data.cell,
            "batch": batch,
            "ptr": data.ptr,
        }
        if hasattr(data, "pbc"):
            mace_input["pbc"] = data.pbc
        if hasattr(data, "head"):
            mace_input["head"] = data.head
        mace_features = self.mace_pool(
            self.mace_backbone(mace_input), batch, num_graphs
        )
        alignn_features = self.alignn_backbone(
            data.alignn_x,
            data.alignn_edge_index,
            data.alignn_edge_attr,
            data.alignn_line_edge_index,
            data.alignn_line_attr,
            batch,
            num_graphs,
        )
        features = self.drop(
            functional.silu(
                self.trunk_bn(self.proj(self.fusion(mace_features, alignn_features)))
            )
        )
        band_gap = self.head_bg(features)
        stability = self.head_eh(features)
        conditioning_weight = self.cond_weight.item()
        gap_type_input = self.gt_proj(
            torch.cat(
                [
                    features,
                    band_gap.detach() * conditioning_weight,
                    stability.exp().detach() * conditioning_weight,
                ],
                dim=1,
            )
        )
        gap_type = self.gt_classifier(self.gt_layers(gap_type_input))
        return band_gap, gap_type, stability


def _gap_type_name(prediction_index: int, band_gap: float) -> str:
    if prediction_index == 1 or band_gap < METAL_BAND_GAP_THRESHOLD:
        return "Indirect"
    return "Direct"


class DBGFNPredictor:
    """Five-checkpoint DBGFN ensemble for direct prediction of new structures."""

    def __init__(
        self,
        checkpoint_dir: str | Path = PROJECT_ROOT / "01-checkpoints",
        device: str | torch.device | None = None,
        mace_model_name: str = "small",
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        from mace.calculators import mace_mp
        from mace.tools import AtomicNumberTable

        calculator = mace_mp(
            model=mace_model_name,
            default_dtype="float32",
            device=str(self.device),
        )
        self.mace_base = calculator.models[0]
        self.atomic_number_table = AtomicNumberTable(
            [int(number) for number in self.mace_base.atomic_numbers]
        )
        self.models, self.normalizers = self._load_ensemble()

    def _load_ensemble(self):
        models, normalizers = [], []
        for fold in range(N_FOLDS):
            path = self.checkpoint_dir / f"fold{fold}_best.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint: {path}. Run 01-checkpoints/download.py first."
                )
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            model = DBGFNModel(copy.deepcopy(self.mace_base))
            model.load_state_dict(checkpoint["model_state_dict"])
            model.set_cond_weight(1.0)
            models.append(model.to(self.device).eval())
            normalizers.append(
                (
                    float(checkpoint["normalizer_mean"]),
                    float(checkpoint["normalizer_std"]),
                )
            )
        return models, normalizers

    @torch.no_grad()
    def predict_graph(self, graph) -> dict[str, float | str]:
        graph = graph.to(self.device)
        band_gaps, gap_type_probabilities, stability_probabilities = [], [], []
        for model, (mean, std) in zip(self.models, self.normalizers):
            band_gap, gap_type, stability = model(graph)
            band_gaps.append(float(np.expm1(band_gap.item() * std + mean)))
            gap_type_probabilities.append(
                gap_type.exp().squeeze(0).cpu().numpy()
            )
            stability_probabilities.append(
                stability.exp().squeeze(0).cpu().numpy()
            )
        mean_band_gap = float(np.mean(band_gaps))
        gap_type_index = int(np.mean(gap_type_probabilities, axis=0).argmax())
        stability_index = int(np.mean(stability_probabilities, axis=0).argmax())
        gap_type = _gap_type_name(gap_type_index, mean_band_gap)
        stability = "Stable" if stability_index == 0 else "Unstable"
        screening_pass = (
            gap_type == "Direct"
            and SCREEN_BAND_GAP_MIN <= mean_band_gap <= SCREEN_BAND_GAP_MAX
            and stability == "Stable"
        )
        return {
            "bg_type": gap_type,
            "bg_eV": round(mean_band_gap, 4),
            "bg_std_eV": round(float(np.std(band_gaps)), 4),
            "ehull": stability,
            "screen_pass": "Yes" if screening_pass else "No",
        }

    def predict_file(self, path: str | Path) -> dict[str, str | float]:
        structure_id, formula, graph = file_to_model_input(
            path, self.atomic_number_table
        )
        return {
            "structure_id": structure_id,
            "source_file": Path(path).name,
            "source_path": str(Path(path).resolve()),
            "formula": formula,
            **self.predict_graph(graph),
        }


def predict_directory(
    input_directory: str | Path,
    output_csv: str | Path,
    checkpoint_dir: str | Path = PROJECT_ROOT / "01-checkpoints",
    device: str | torch.device | None = None,
    mace_model_name: str = "small",
) -> tuple[list[dict[str, str | float]], list[tuple[Path, Exception]]]:
    """Run the full five-model ensemble and save predictions to CSV."""

    files = find_structure_files(input_directory)
    if not files:
        raise FileNotFoundError(f"No CIF, VASP, or POSCAR files found in {input_directory}")
    predictor = DBGFNPredictor(checkpoint_dir, device, mace_model_name)
    rows, failures = [], []
    for path in files:
        try:
            rows.append(predictor.predict_file(path))
        except Exception as error:  # Continue to report all malformed structures.
            failures.append((path, error))
    if not rows:
        raise RuntimeError("No structure could be converted and predicted.")

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Predict new crystal structures with the released DBGFN ensemble."
    )
    parser.add_argument("input_directory", help="Directory containing CIF, VASP, or POSCAR files")
    parser.add_argument("output_csv", help="Destination CSV for ensemble predictions")
    parser.add_argument(
        "--checkpoint-dir",
        default=PROJECT_ROOT / "01-checkpoints",
        type=Path,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--mace-model", default="small")
    args = parser.parse_args()
    try:
        rows, failures = predict_directory(
            args.input_directory,
            args.output_csv,
            args.checkpoint_dir,
            args.device,
            args.mace_model,
        )
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Saved {len(rows)} predictions to {args.output_csv}")
    if failures:
        for path, error in failures:
            print(f"SKIPPED {path.name}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
