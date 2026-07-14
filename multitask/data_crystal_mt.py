"""
Shared crystal graph dataset for ALIGNN and M3GNet (PyG-based, no DGL needed).

Builds:
  1. Crystal graph: atoms as nodes, bonds as directed edges
  2. Line graph: bonds as nodes, i→j→k triplets as directed edges (for 3-body interactions)

Data format (id_prop.csv):
  material_id, bandgap(eV), gap_type(0=direct/1=indirect/2=metal), E_hull(eV/atom)
"""

import os
import csv
import warnings
from collections import Counter

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")


# ─── RBF expansion ────────────────────────────────────────────────────────────

class RBFExpansion:
    """Gaussian radial basis function expansion."""
    def __init__(self, vmin: float, vmax: float, bins: int):
        self.centers = torch.linspace(vmin, vmax, bins)
        self.width = (vmax - vmin) / bins

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -((values.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2)
            / (2 * self.width ** 2)
        )  # [N, bins]


# ─── Atom features ────────────────────────────────────────────────────────────

def atom_onehot(atomic_num: int, max_z: int = 94) -> torch.Tensor:
    feat = torch.zeros(max_z)
    z = int(atomic_num)
    if 1 <= z <= max_z:
        feat[z - 1] = 1.0
    return feat


# ─── Crystal graph + line graph ────────────────────────────────────────────────

class CrystalData(Data):
    """PyG Data subclass with crystal graph + line graph."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "line_edge_index":
            # Line graph nodes = crystal graph edges → offset by E
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("line_edge_index",):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def build_crystal_data(
    struct,
    rbf_dist: RBFExpansion,
    rbf_angle: RBFExpansion,
    radius: float = 8.0,
    max_num_nbr: int = 12,
    max_z: int = 94,
) -> CrystalData:
    """Convert a pymatgen Structure → CrystalData with crystal + line graphs."""
    n_atoms = len(struct)

    # ── Atom features ──────────────────────────────────────────────
    x = torch.stack([atom_onehot(s.specie.Z, max_z) for s in struct])  # [N, max_z]

    # ── Bonds (directed, with periodic images) ─────────────────────
    all_nbrs = struct.get_all_neighbors(radius, include_index=True)

    src_list, dst_list, vec_list = [], [], []
    for i, nbrs in enumerate(all_nbrs):
        nbrs_sorted = sorted(nbrs, key=lambda n: n[1])[:max_num_nbr]
        for nbr in nbrs_sorted:
            j = nbr[2]           # neighbor site index
            nbr_cart = nbr[0].coords  # Cartesian coords of neighbor image
            site_cart = struct[i].coords
            vec = nbr_cart - site_cart   # i→j vector (Cartesian, Å)
            src_list.append(i)
            dst_list.append(j)
            vec_list.append(vec)

    if len(src_list) == 0:
        raise ValueError(f"No neighbors found within radius {radius} Å")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, E]
    edge_vec = torch.tensor(np.array(vec_list), dtype=torch.float32)    # [E, 3]
    edge_dist = edge_vec.norm(dim=1)                                     # [E]
    edge_attr = rbf_dist(edge_dist)                                      # [E, dist_bins]

    # ── Line graph (directed triplets i→j→k) ────────────────────────
    src_e = edge_index[0]  # source atom of each edge [E]
    dst_e = edge_index[1]  # dest   atom of each edge [E]
    E = edge_index.size(1)

    line_src_parts, line_dst_parts = [], []
    for j in range(n_atoms):
        in_e = (dst_e == j).nonzero(as_tuple=False).view(-1)   # edges i→j
        out_e = (src_e == j).nonzero(as_tuple=False).view(-1)  # edges j→k
        if len(in_e) == 0 or len(out_e) == 0:
            continue
        # Cartesian product: all (e1, e2) pairs
        rep_in  = in_e.repeat_interleave(len(out_e))   # [n_in * n_out]
        rep_out = out_e.repeat(len(in_e))               # [n_in * n_out]
        line_src_parts.append(rep_in)
        line_dst_parts.append(rep_out)

    if line_src_parts:
        line_src = torch.cat(line_src_parts)
        line_dst = torch.cat(line_dst_parts)
    else:
        line_src = torch.zeros(0, dtype=torch.long)
        line_dst = torch.zeros(0, dtype=torch.long)

    line_edge_index = torch.stack([line_src, line_dst], dim=0)  # [2, T]

    # Angle features: cos of angle at atom j between bonds i→j and j→k
    if line_edge_index.size(1) > 0:
        v_in  = edge_vec[line_src]   # [T, 3]  (i→j vectors)
        v_out = edge_vec[line_dst]   # [T, 3]  (j→k vectors)
        # angle at j: between -v_ij and v_jk
        cos_angle = (-v_in * v_out).sum(dim=1) / (
            v_in.norm(dim=1) * v_out.norm(dim=1) + 1e-8
        )
        cos_angle = cos_angle.clamp(-1.0, 1.0)
        line_attr = rbf_angle(cos_angle)  # [T, angle_bins]
    else:
        line_attr = torch.zeros(0, rbf_angle.centers.size(0))

    return CrystalData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_vec=edge_vec,
        line_edge_index=line_edge_index,
        line_attr=line_attr,
        n_atoms=torch.tensor(n_atoms),
    )


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CrystalMultiTaskDataset(torch.utils.data.Dataset):
    """
    Crystal multi-task dataset.

    id_prop.csv columns: material_id, bandgap(eV), gap_type(0/1/2), E_hull(eV/atom)
    - BG: continuous regression target (log-transformed if log_bg=True)
    - GT: binary classification — 0=Direct, 1=Indirect+Metal (merged)
    - EH: binary classification — 0=Stable (EH<eh_threshold), 1=Unstable
    """

    def __init__(
        self,
        root_dir: str,
        cache_dir: str = None,
        radius: float = 8.0,
        max_num_nbr: int = 12,
        dist_bins: int = 80,
        angle_bins: int = 40,
        max_z: int = 94,
        log_bg: bool = True,
        merge_metal_indirect: bool = True,
        eh_threshold: float = 0.1,
    ):
        self.root_dir = root_dir
        self.cache_dir = cache_dir or os.path.join(root_dir, "crystal_cached_graphs")
        self.log_bg = log_bg
        self.merge_metal_indirect = merge_metal_indirect
        self.eh_threshold = eh_threshold

        self.rbf_dist  = RBFExpansion(0.0, radius, dist_bins)
        self.rbf_angle = RBFExpansion(-1.0, 1.0, angle_bins)
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.max_z = max_z

        # Load targets
        self.samples = []
        id_prop_path = os.path.join(root_dir, "id_prop.csv")
        with open(id_prop_path) as f:
            for row in csv.reader(f):
                mid = row[0].strip()
                bg_raw = float(row[1])
                gt_raw = int(row[2])
                eh_raw = float(row[3])

                # BG target
                if log_bg:
                    bg = float(np.log1p(max(bg_raw, 0.0)))
                else:
                    bg = bg_raw

                # GT: -1 = unknown (JARVIS), merge metal (2) → indirect (1) if requested
                if gt_raw == -1:
                    gt = -1  # unknown gap type, will be masked in loss
                elif merge_metal_indirect and gt_raw == 2:
                    gt = 1
                else:
                    gt = gt_raw if gt_raw < 2 else 2  # keep 3-class otherwise

                # EH: binary stable/unstable
                eh = 0 if eh_raw < eh_threshold else 1

                cif_path = os.path.join(root_dir, f"{mid}.cif")
                if os.path.exists(cif_path):
                    self.samples.append({
                        "id": mid, "cif": cif_path,
                        "bg": bg, "gt": gt, "eh": eh,
                    })

        os.makedirs(self.cache_dir, exist_ok=True)
        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cache_path = os.path.join(self.cache_dir, f"{s['id']}.pt")

        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, weights_only=False)
            except Exception:
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                data = self._build_data(s)
                self._atomic_save(data, cache_path)
        else:
            data = self._build_data(s)
            self._atomic_save(data, cache_path)

        data.bg = torch.tensor(s["bg"], dtype=torch.float32)
        data.gt = torch.tensor(s["gt"], dtype=torch.long)
        data.eh = torch.tensor(s["eh"], dtype=torch.long)
        return data


    def _build_data(self, s):
        from pymatgen.core import Structure
        struct = Structure.from_file(s["cif"])
        return build_crystal_data(
            struct, self.rbf_dist, self.rbf_angle,
            self.radius, self.max_num_nbr, self.max_z,
        )

    def _atomic_save(self, data, cache_path):
        tmp = cache_path + f".tmp{os.getpid()}"
        torch.save(data, tmp)
        os.replace(tmp, cache_path)

    def get_gt_dist(self):
        return Counter(s["gt"] for s in self.samples)

    def get_eh_dist(self):
        return Counter(s["eh"] for s in self.samples)


# ─── Stratified k-fold split ──────────────────────────────────────────────────

def _load_aligned_split(dataset, fold):
    split_dir = os.path.join(dataset.root_dir, "splits_mace_alignn")
    paths = {name: os.path.join(split_dir, f"fold{fold}_{name}.txt")
             for name in ("train", "val", "test")}
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    id_to_idx = {sample["id"]: i for i, sample in enumerate(dataset.samples)}
    result = []
    for name in ("train", "val", "test"):
        idxs = []
        missing = []
        with open(paths[name]) as f:
            for line in f:
                mid = line.strip()
                if not mid:
                    continue
                if mid in id_to_idx:
                    idxs.append(id_to_idx[mid])
                else:
                    missing.append(mid)
        if missing:
            raise RuntimeError(f"Aligned split {paths[name]} has {len(missing)} ids missing from dataset; first={missing[:3]}")
        result.append(idxs)
    print(f"Using aligned MACE+ALIGNN split files: {split_dir} fold {fold}")
    return tuple(result)

def stratified_kfold_split(dataset, k_folds=5, fold=0, val_ratio=0.1, seed=42):
    aligned = _load_aligned_split(dataset, fold)
    if aligned is not None:
        return aligned
    """
    Returns (train_idx, val_idx, test_idx) using stratified k-fold on GT labels.
    Samples with gt=-1 (unknown gap type, e.g. JARVIS) are always added to train.
    Test set = current fold of known-GT samples; remaining split into train/val.
    """
    known_idx = [i for i, s in enumerate(dataset.samples) if s["gt"] >= 0]
    unknown_idx = [i for i, s in enumerate(dataset.samples) if s["gt"] < 0]

    known_labels = [dataset.samples[i]["gt"] for i in known_idx]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(known_idx, known_labels))
    rel_train_val, rel_test = splits[fold]

    test_idx = [known_idx[i] for i in rel_test]
    train_val_known = [known_idx[i] for i in rel_train_val]

    # Further split train_val into train/val
    n_val = max(1, int(len(train_val_known) * val_ratio))
    rng = np.random.RandomState(seed + fold)
    rng.shuffle(train_val_known)
    val_idx = train_val_known[:n_val]
    train_idx = train_val_known[n_val:] + unknown_idx  # unknown GT always in train

    return train_idx, val_idx, test_idx
