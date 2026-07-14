"""
Pre-compute graph features from CIF files and save as .pt tensors.
This eliminates the CIF parsing bottleneck during training.

Usage:
    python preprocess_graphs.py Data/multitask Data/multitask/cached_graphs
"""
import argparse
import csv
import os
import sys
import time
import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer


def build_graph(crystal, ari, gdf, max_num_nbr=12, radius=8, cif_id='?'):
    atom_fea = np.vstack([
        ari.get_atom_fea(crystal[i].specie.number)
        for i in range(len(crystal))
    ])
    atom_fea = torch.Tensor(atom_fea)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            warnings.warn(f'{cif_id} not find enough neighbors.')
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) +
                [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(
                list(map(lambda x: x[1], nbr)) +
                [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(
                list(map(lambda x: x[1], nbr[:max_num_nbr])))

    nbr_fea = torch.Tensor(gdf.expand(np.array(nbr_fea)))
    nbr_fea_idx = torch.LongTensor(np.array(nbr_fea_idx))
    return atom_fea, nbr_fea, nbr_fea_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help='Data dir with id_prop.csv and CIFs')
    parser.add_argument('cache_dir', help='Output dir for cached .pt files')
    parser.add_argument('--radius', default=8, type=float)
    parser.add_argument('--max-num-nbr', default=12, type=int)
    parser.add_argument('--dmin', default=0, type=float)
    parser.add_argument('--step', default=0.2, type=float)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    id_prop_file = os.path.join(args.data_dir, 'id_prop.csv')
    with open(id_prop_file) as f:
        id_prop_data = list(csv.reader(f))

    atom_init_file = os.path.join(args.data_dir, 'atom_init.json')
    ari = AtomCustomJSONInitializer(atom_init_file)
    gdf = GaussianDistance(dmin=args.dmin, dmax=args.radius, step=args.step)

    print(f'Processing {len(id_prop_data)} structures...')
    t0 = time.time()
    success, failed = 0, 0

    for i, row in enumerate(id_prop_data):
        cif_id = row[0]
        out_path = os.path.join(args.cache_dir, f'{cif_id}.pt')
        if os.path.isfile(out_path):
            success += 1
            continue

        try:
            cif_path = os.path.join(args.data_dir, f'{cif_id}.cif')
            crystal = Structure.from_file(cif_path)
            atom_fea, nbr_fea, nbr_fea_idx = build_graph(
                crystal, ari, gdf, args.max_num_nbr, args.radius, cif_id)

            torch.save({
                'atom_fea': atom_fea,
                'nbr_fea': nbr_fea,
                'nbr_fea_idx': nbr_fea_idx,
                'bandgap': torch.Tensor([float(row[1])]),
                'gaptype': torch.LongTensor([int(row[2])]),
                'ehull': torch.Tensor([float(row[3])]),
            }, out_path)
            success += 1
        except Exception as e:
            print(f'  FAILED {cif_id}: {e}')
            failed += 1

        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(id_prop_data)} done '
                  f'({time.time()-t0:.0f}s)')

    elapsed = time.time() - t0
    print(f'\nDone: {success} cached, {failed} failed in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
