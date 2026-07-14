"""Multi-task dataset for CGCNN v2: stratified splitting by gap type."""
from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


# Reuse GaussianDistance and AtomInitializer from original CGCNN
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer


def stratified_split(id_prop_data, train_ratio=0.8, val_ratio=0.1,
                     test_ratio=0.1, random_seed=123):
    """Split data with stratification on gap type (column index 2).

    Returns train_indices, val_indices, test_indices.
    """
    rng = random.Random(random_seed)

    # Group indices by gap type
    groups = {}
    for i, row in enumerate(id_prop_data):
        label = int(row[2])
        groups.setdefault(label, []).append(i)

    train_idx, val_idx, test_idx = [], [], []
    for label, indices in sorted(groups.items()):
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        # rest goes to test
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    # Shuffle within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=0.8,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False,
                              stratified=True, **kwargs):
    """Create data loaders with optional stratified splitting."""
    if stratified:
        train_idx, val_idx, test_idx = stratified_split(
            dataset.id_prop_data, train_ratio, val_ratio, test_ratio,
            random_seed=123)
    else:
        total_size = len(dataset)
        indices = list(range(total_size))
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    loader_kw = dict(batch_size=batch_size, collate_fn=collate_fn,
                     num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(dataset, sampler=train_sampler, **loader_kw)
    val_loader = DataLoader(dataset, sampler=val_sampler, **loader_kw)

    if return_test:
        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = DataLoader(dataset, sampler=test_sampler, **loader_kw)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader


def collate_pool_mt(dataset_list):
    """Collate for multi-task: target is a dict of tensors."""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    batch_bandgap, batch_gaptype, batch_ehull = [], [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), targets, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_bandgap.append(targets['bandgap'])
        batch_gaptype.append(targets['gaptype'])
        batch_ehull.append(targets['ehull'])
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    target_dict = {
        'bandgap': torch.stack(batch_bandgap, dim=0),
        'gaptype': torch.stack(batch_gaptype, dim=0),
        'ehull': torch.stack(batch_ehull, dim=0),
    }
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), target_dict, batch_cif_ids


class CIFDataMultiTask(Dataset):
    """
    Multi-task CIF dataset.

    id_prop.csv format: id, band_gap, gap_type(0/1/2), energy_above_hull
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'
        id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader]
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    def get_class_weights(self):
        """Compute inverse-frequency class weights for gap type."""
        counts = {}
        for row in self.id_prop_data:
            label = int(row[2])
            counts[label] = counts.get(label, 0) + 1
        n_classes = max(counts.keys()) + 1
        total = sum(counts.values())
        weights = []
        for c in range(n_classes):
            w = total / (n_classes * counts.get(c, 1))
            weights.append(w)
        return torch.FloatTensor(weights)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id, bandgap, gaptype, ehull = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir,
                                                   cif_id + '.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf.expand(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        targets = {
            'bandgap': torch.Tensor([float(bandgap)]),
            'gaptype': torch.LongTensor([int(gaptype)]),
            'ehull': torch.Tensor([float(ehull)]),
        }
        return (atom_fea, nbr_fea, nbr_fea_idx), targets, cif_id
