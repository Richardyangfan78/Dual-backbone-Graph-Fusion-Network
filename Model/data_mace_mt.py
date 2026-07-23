"""
MACE-compatible dataset for multi-task chalcohalide prediction.

Reads CIF files, converts to MACE-format AtomicData objects,
and attaches multi-task labels (bandgap, gap_type, e_hull).
Supports caching of converted graphs and stratified k-fold splitting.
"""
from __future__ import print_function, division

import os
import csv
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional

import ase.io
from mace import data as mace_data


# Keys to copy from MACE AtomicData to PyG Data
_MACE_KEYS = [
    "positions", "node_attrs", "edge_index", "shifts", "unit_shifts",
    "cell", "pbc", "head", "weight", "energy_weight", "forces_weight",
]


def _mace_to_pyg(atomic_data) -> Data:
    """Convert MACE AtomicData to standard PyG Data object."""
    d = atomic_data.to_dict()
    pyg_data = Data()
    for key in _MACE_KEYS:
        if key in d:
            setattr(pyg_data, key, d[key])
    # num_nodes is needed for proper batching
    pyg_data.num_nodes = d["positions"].shape[0]
    return pyg_data


class MACECrystalDataset(Dataset):
    """Dataset that converts CIF files to MACE-format graphs with MT labels.

    Each sample is a PyG Data object with MACE fields + target labels.

    Args:
        root_dir: directory containing CIF files and id_prop.csv
        z_table: MACE atomic number table (from calculator)
        r_max: cutoff radius (6.0 for MACE-MP-0)
        cache_dir: directory to cache converted .pt files
        merge_metal_indirect: if True, merge Metal (class 2) into Indirect (class 1)
        log_bg: if True, use log(1 + bg) for bandgap target
        eh_threshold: threshold for E_hull binary classification
    """
    def __init__(self, root_dir, z_table, r_max=6.0, cache_dir=None,
                 merge_metal_indirect=True, log_bg=True, eh_threshold=0.1):
        self.root_dir = root_dir
        self.z_table = z_table
        self.r_max = r_max
        self.cache_dir = cache_dir
        self.merge_metal_indirect = merge_metal_indirect
        self.log_bg = log_bg
        self.eh_threshold = eh_threshold

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

        # Read labels
        id_prop_file = os.path.join(root_dir, "id_prop.csv")
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]

        # Build sample list
        self.samples = []
        skipped = 0
        for row in self.id_prop_data:
            mp_id = row[0]
            cif_path = os.path.join(root_dir, "cifs", mp_id + ".cif")
            if not os.path.exists(cif_path):
                # Also accept external datasets that keep CIFs beside id_prop.csv.
                cif_path = os.path.join(root_dir, mp_id + ".cif")
            cache_path = (os.path.join(cache_dir, f"{mp_id}.pt")
                          if cache_dir is not None else None)
            if not os.path.exists(cif_path) and not (cache_path and os.path.exists(cache_path)):
                skipped += 1
                continue

            bg = float(row[1])
            gt = int(row[2])
            eh = float(row[3])

            # Process gap type
            if merge_metal_indirect and gt == 2:
                gt = 1  # Metal → Indirect

            # Process E_hull: binary (stable=0, unstable=1)
            eh_class = 0 if eh < eh_threshold else 1

            # Process bandgap
            bg_target = np.log1p(bg) if log_bg else bg

            self.samples.append({
                "mp_id": mp_id,
                "cif_path": cif_path,
                "bg": bg_target,
                "bg_raw": bg,
                "gt": gt,
                "eh": eh_class,
                "eh_raw": eh,
            })

        if skipped > 0:
            print(f"Warning: skipped {skipped} samples (CIF not found)")
        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mp_id = sample["mp_id"]

        # Check cache
        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, f"{mp_id}.pt")
            if os.path.exists(cache_path):
                data = torch.load(cache_path, weights_only=False)
                # Attach labels (in case they changed)
                data.bg = torch.tensor([sample["bg"]], dtype=torch.float)
                data.gt = torch.tensor([sample["gt"]], dtype=torch.long)
                data.eh = torch.tensor([sample["eh"]], dtype=torch.long)
                data.mp_id = mp_id
                return data

        # Load CIF and convert
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = ase.io.read(sample["cif_path"])

        # Convert to MACE format (MACE AtomicData → PyG Data)
        keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
        config = mace_data.config_from_atoms(atoms, key_specification=keyspec)
        atomic_data = mace_data.AtomicData.from_config(
            config, z_table=self.z_table, cutoff=self.r_max, heads=["Default"]
        )

        # Convert MACE AtomicData to standard PyG Data
        data = _mace_to_pyg(atomic_data)

        # Attach labels
        data.bg = torch.tensor([sample["bg"]], dtype=torch.float)
        data.gt = torch.tensor([sample["gt"]], dtype=torch.long)
        data.eh = torch.tensor([sample["eh"]], dtype=torch.long)
        data.mp_id = mp_id

        # Cache
        if self.cache_dir is not None:
            torch.save(data, cache_path)

        return data

    def get_gt_class_dist(self):
        """Returns gap type class distribution."""
        from collections import Counter
        return Counter(s["gt"] for s in self.samples)

    def get_eh_class_dist(self):
        """Returns E_hull class distribution."""
        from collections import Counter
        return Counter(s["eh"] for s in self.samples)

    def get_stratify_labels(self):
        """Returns combined labels for stratified splitting (GT × EH)."""
        return [s["gt"] * 2 + s["eh"] for s in self.samples]


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
    """Stratified k-fold split preserving GT×EH distribution.

    Returns:
        train_indices, val_indices, test_indices
    """
    from sklearn.model_selection import StratifiedKFold

    labels = dataset.get_stratify_labels()
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(indices, labels))
    train_val_idx, test_idx = splits[fold]

    # Split train_val into train and val
    rng = np.random.RandomState(seed + fold)
    n_val = max(1, int(len(train_val_idx) * val_ratio))

    # Stratified val split
    train_val_labels = [labels[i] for i in train_val_idx]
    perm = rng.permutation(len(train_val_idx))
    # Simple stratified: sort by label, take every k-th
    sorted_idx = sorted(range(len(perm)), key=lambda i: train_val_labels[perm[i]])
    step = max(1, len(sorted_idx) // n_val)
    val_mask = set(sorted_idx[::step][:n_val])
    val_local = [perm[i] for i in val_mask]
    train_local = [perm[i] for i in range(len(perm)) if i not in val_mask]

    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()
