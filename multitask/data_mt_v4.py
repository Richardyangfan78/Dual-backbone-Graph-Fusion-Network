"""
Multi-task dataset v4: Metal→Indirect merge + structure augmentation.

- merge_metal_indirect: map gap_type 2 (Metal) -> 1 (Indirect), binary classification
- Structure augmentation: coordinate perturbation + lattice scaling (train only)
- Cached structure loading for fast augmentation
"""
from __future__ import print_function, division

import csv
import functools
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer

from data_mt import collate_pool_mt


def _map_gaptype(gaptype, merge_metal_indirect):
    """Map gap type: 2 (Metal) -> 1 (Indirect) when merge_metal_indirect."""
    if merge_metal_indirect and int(gaptype) == 2:
        return 1
    return int(gaptype)


def stratified_split_v4(id_prop_data, train_ratio=0.8, val_ratio=0.1,
                        test_ratio=0.1, random_seed=123,
                        merge_metal_indirect=False):
    """Stratified split by gap type (using mapped labels when merge_metal_indirect)."""
    rng = random.Random(random_seed)
    groups = {}
    for i, row in enumerate(id_prop_data):
        label = _map_gaptype(row[2], merge_metal_indirect)
        groups.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(groups.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def stratified_kfold_v4(id_prop_data, n_folds=5, random_seed=123,
                        merge_metal_indirect=False):
    """Stratified K-fold by gap type (using mapped labels when merge_metal_indirect)."""
    rng = random.Random(random_seed)
    groups = {}
    for i, row in enumerate(id_prop_data):
        label = _map_gaptype(row[2], merge_metal_indirect)
        groups.setdefault(label, []).append(i)

    folds = [[] for _ in range(n_folds)]
    for label in sorted(groups.keys()):
        indices = groups[label][:]
        rng.shuffle(indices)
        for j, idx in enumerate(indices):
            folds[j % n_folds].append(idx)
    for fold in folds:
        rng.shuffle(fold)
    return folds


def stratified_split_from_indices(id_prop_data, indices, train_ratio=0.85,
                                  random_seed=123, merge_metal_indirect=False):
    """Stratified split of indices into train/val by gap type.
    Used for v9 protocol: split train_val_idx into train and val."""
    rng = random.Random(random_seed)
    groups = {}
    for i in indices:
        row = id_prop_data[i]
        label = _map_gaptype(row[2], merge_metal_indirect)
        groups.setdefault(label, []).append(i)

    train_idx, val_idx = [], []
    for label, idx_list in sorted(groups.items()):
        rng.shuffle(idx_list)
        n = len(idx_list)
        n_train = max(1, int(train_ratio * n))
        train_idx.extend(idx_list[:n_train])
        val_idx.extend(idx_list[n_train:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


class CIFDataMultiTaskV4(Dataset):
    """
    Multi-task CIF dataset v4.

    - merge_metal_indirect: map Metal (2) -> Indirect (1), n_classes=2
    - augment: enable structure augmentation (set by trainer for train loader)
    - augment_perturb: max coordinate perturbation in Angstrom
    - augment_scale: lattice scale range e.g. 0.01 for ±1%
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, merge_metal_indirect=False,
                 augment_perturb=0.02, augment_scale=0.01):
        self.root_dir = root_dir
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.merge_metal_indirect = merge_metal_indirect
        self.augment_perturb = augment_perturb
        self.augment_scale = augment_scale
        self.augment = False  # set by trainer

        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        atom_init_file = os.path.join(root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.lru_cache(maxsize=None)
    def _load_structure(self, idx):
        """Load and cache structure (no augmentation)."""
        cif_id = self.id_prop_data[idx][0]
        path = os.path.join(self.root_dir, cif_id + '.cif')
        return Structure.from_file(path)

    def _apply_augmentation(self, structure):
        """Apply random structure perturbation and lattice scaling."""
        s = structure.copy()
        if self.augment_perturb > 0:
            s.perturb(distance=self.augment_perturb)
        if self.augment_scale > 0:
            factor = (1.0 + random.uniform(-self.augment_scale, self.augment_scale)) ** 3
            s.scale_lattice(s.volume * factor)
        return s

    def _build_graph(self, crystal, cif_id='?'):
        """Build graph features from structure."""
        atom_fea = np.vstack([
            self.ari.get_atom_fea(crystal[i].specie.number)
            for i in range(len(crystal))
        ])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn(
                    '{} not find enough neighbors. Consider increase radius.'.format(cif_id),
                    stacklevel=0)
                nbr_fea_idx.append(
                    list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(
                    list(map(lambda x: x[1], nbr)) +
                    [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))

        nbr_fea_idx = np.array(nbr_fea_idx)
        nbr_fea = np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        return atom_fea, nbr_fea, nbr_fea_idx

    def __getitem__(self, idx):
        cif_id, bandgap, gaptype, ehull = self.id_prop_data[idx]
        crystal = self._load_structure(idx)
        if self.augment:
            crystal = self._apply_augmentation(crystal)

        atom_fea, nbr_fea, nbr_fea_idx = self._build_graph(crystal, cif_id)
        gaptype = _map_gaptype(gaptype, self.merge_metal_indirect)

        targets = {
            'bandgap': torch.Tensor([float(bandgap)]),
            'gaptype': torch.LongTensor([gaptype]),
            'ehull': torch.Tensor([float(ehull)]),
        }
        return (atom_fea, nbr_fea, nbr_fea_idx), targets, cif_id

    def get_class_weights(self):
        """Inverse-frequency class weights (using mapped labels)."""
        counts = {}
        for row in self.id_prop_data:
            label = _map_gaptype(row[2], self.merge_metal_indirect)
            counts[label] = counts.get(label, 0) + 1
        n_classes = max(counts.keys()) + 1
        total = sum(counts.values())
        weights = [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)]
        return torch.FloatTensor(weights)

    @property
    def n_gap_classes(self):
        return 2 if self.merge_metal_indirect else 3
