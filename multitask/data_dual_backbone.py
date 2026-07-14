"""
Dual-backbone dataset: loads both MACE and ALIGNN (crystal) cached graphs.

Each sample produces a single PyG Data object containing both graph formats:
  - MACE fields: positions, node_attrs, edge_index, shifts, unit_shifts, cell, pbc
  - ALIGNN fields: alignn_x, alignn_edge_index, alignn_edge_attr,
                    alignn_line_edge_index, alignn_line_attr

Custom __inc__ handles batching of both sets of edge indices correctly.
"""
from __future__ import print_function, division

import os
import csv
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Optional


# Keys to transfer from cached MACE graph
# NOTE: "node_attrs" is renamed to "mace_node_attrs" to avoid collision
# with PyG Data.node_attrs() method which breaks batching
_MACE_KEYS = [
    "positions", "edge_index", "shifts", "unit_shifts",
    "cell", "pbc", "head", "weight", "energy_weight", "forces_weight",
]


class DualGraphData(Data):
    """PyG Data with both MACE and ALIGNN graph fields.

    Handles proper batching by overriding __inc__ and __cat_dim__ for
    ALIGNN-specific index tensors.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'alignn_edge_index':
            # ALIGNN crystal graph edges: increment by num atoms
            return self.num_nodes
        if key == 'alignn_line_edge_index':
            # Line graph nodes = crystal graph edges
            return self.alignn_edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('alignn_edge_index', 'alignn_line_edge_index'):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class DualBackboneDataset(Dataset):
    """Dataset that loads both MACE and ALIGNN cached graphs per crystal.

    Both cache directories must contain pre-computed .pt files with matching
    material IDs (e.g., mp-1006611.pt).

    Args:
        root_dir: directory with id_prop.csv
        mace_cache_dir: directory with MACE cached .pt files
        crystal_cache_dir: directory with crystal (ALIGNN/M3GNet) cached .pt files
        merge_metal_indirect: merge Metal (class 2) into Indirect (class 1)
        log_bg: use log(1+bg) for bandgap target
        eh_threshold: threshold for E_hull binary classification
    """

    def __init__(self, root_dir, mace_cache_dir, crystal_cache_dir,
                 merge_metal_indirect=True, log_bg=True, eh_threshold=0.1):
        self.root_dir = root_dir
        self.mace_cache_dir = mace_cache_dir
        self.crystal_cache_dir = crystal_cache_dir
        self.merge_metal_indirect = merge_metal_indirect
        self.log_bg = log_bg
        self.eh_threshold = eh_threshold

        # Read labels
        id_prop_file = os.path.join(root_dir, "id_prop.csv")
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Build sample list (only include samples with both cached graphs)
        self.samples = []
        skipped = 0
        for row in self.id_prop_data:
            mp_id = row[0]
            mace_path = os.path.join(mace_cache_dir, f"{mp_id}.pt")
            crystal_path = os.path.join(crystal_cache_dir, f"{mp_id}.pt")

            if not os.path.exists(mace_path) or not os.path.exists(crystal_path):
                skipped += 1
                continue

            bg = float(row[1])
            gt = int(row[2])
            eh = float(row[3])

            if gt == -1:
                pass  # unknown gap type (JARVIS), keep as -1
            elif merge_metal_indirect and gt == 2:
                gt = 1
            eh_class = 0 if eh < eh_threshold else 1
            bg_target = np.log1p(bg) if log_bg else bg

            self.samples.append({
                "mp_id": mp_id,
                "mace_path": mace_path,
                "crystal_path": crystal_path,
                "bg": bg_target,
                "bg_raw": bg,
                "gt": gt,
                "eh": eh_class,
                "eh_raw": eh,
            })

        if skipped > 0:
            print(f"Warning: skipped {skipped} samples (missing cached graphs)")
        print(f"DualBackbone dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load both cached graphs
        mace_data = torch.load(sample["mace_path"], weights_only=False)
        crystal_data = torch.load(sample["crystal_path"], weights_only=False)

        # Create combined DualGraphData
        dual = DualGraphData()

        # MACE fields — use dict-style access mace_data[key] instead of
        # getattr(mace_data, key) to avoid PyG method/attribute name conflicts
        # (e.g., Data.node_attrs() method shadows the stored tensor)
        for key in _MACE_KEYS:
            if key in mace_data:
                dual[key] = mace_data[key]
        # node_attrs renamed to mace_node_attrs to avoid collision with
        # PyG Data.node_attrs() method which breaks batching
        if 'node_attrs' in mace_data:
            dual['mace_node_attrs'] = mace_data['node_attrs']
        dual.num_nodes = mace_data.num_nodes

        # ALIGNN fields (prefixed to avoid collision)
        dual.alignn_x = crystal_data.x
        dual.alignn_edge_index = crystal_data.edge_index
        dual.alignn_edge_attr = crystal_data.edge_attr
        dual.alignn_line_edge_index = crystal_data.line_edge_index
        dual.alignn_line_attr = crystal_data.line_attr

        # Labels
        dual.bg = torch.tensor([sample["bg"]], dtype=torch.float)
        dual.gt = torch.tensor([sample["gt"]], dtype=torch.long)
        dual.eh = torch.tensor([sample["eh"]], dtype=torch.long)
        dual.mp_id = sample["mp_id"]

        return dual

    def get_gt_class_dist(self):
        from collections import Counter
        return Counter(s["gt"] for s in self.samples)

    def get_eh_class_dist(self):
        from collections import Counter
        return Counter(s["eh"] for s in self.samples)

    def get_stratify_labels(self):
        # gt=-1 (unknown) mapped to label 4 to keep them in their own stratification group
        return [(s["gt"] * 2 + s["eh"]) if s["gt"] >= 0 else 4 for s in self.samples]


def _load_aligned_split(dataset, fold):
    split_dir = os.path.join(dataset.root_dir, "splits_mace_alignn")
    paths = {name: os.path.join(split_dir, f"fold{fold}_{name}.txt")
             for name in ("train", "val", "test")}
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    id_to_idx = {sample["mp_id"]: i for i, sample in enumerate(dataset.samples)}
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
    """Stratified k-fold split (same as data_mace_mt.py)."""
    from sklearn.model_selection import StratifiedKFold

    # Separate unknown-GT (JARVIS) samples — always go to train
    unknown_idx = [i for i, s in enumerate(dataset.samples) if s["gt"] < 0]
    known_idx   = [i for i, s in enumerate(dataset.samples) if s["gt"] >= 0]

    known_labels = [dataset.samples[i]["gt"] * 2 + dataset.samples[i]["eh"]
                    for i in known_idx]
    known_arr = np.array(known_idx)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(known_arr, known_labels))
    rel_train_val, rel_test = splits[fold]

    test_idx      = known_arr[rel_test].tolist()
    train_val_arr = known_arr[rel_train_val]

    rng = np.random.RandomState(seed + fold)
    n_val = max(1, int(len(train_val_arr) * val_ratio))

    tv_labels = [known_labels[i] for i in rel_train_val]
    perm = rng.permutation(len(train_val_arr))
    sorted_idx = sorted(range(len(perm)), key=lambda i: tv_labels[perm[i]])
    step = max(1, len(sorted_idx) // n_val)
    val_mask = set(sorted_idx[::step][:n_val])
    val_local   = [perm[i] for i in val_mask]
    train_local = [perm[i] for i in range(len(perm)) if i not in val_mask]

    train_idx = train_val_arr[train_local].tolist() + unknown_idx
    val_idx   = train_val_arr[val_local].tolist()

    return train_idx, val_idx, test_idx
