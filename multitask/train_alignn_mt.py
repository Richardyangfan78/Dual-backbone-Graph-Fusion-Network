"""
Multi-task ALIGNN training script.

Uses ALIGNN backbone with 3 heads (BG, GT, EH).
Custom data pipeline that builds DGL graphs + line graphs with multi-task targets.
"""
import argparse
import csv
import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from collections import Counter
from tqdm import tqdm

import dgl

# ALIGNN / jarvis imports
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.graphs import compute_bond_cosines

from alignn_mt_model import ALIGNNMultiTask
from alignn.models.alignn import ALIGNNConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Loss modules (same as v6b)
# ---------------------------------------------------------------------------
class DynamicMultiTaskLoss(nn.Module):
    def __init__(self, log_var_min=-2.0, log_var_max=4.0):
        super().__init__()
        self.log_var_bg = nn.Parameter(torch.tensor(0.0))
        self.log_var_gt = nn.Parameter(torch.tensor(0.0))
        self.log_var_eh = nn.Parameter(torch.tensor(0.0))
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

    def _clamp(self, lv):
        return lv.clamp(self.log_var_min, self.log_var_max)

    def forward(self, loss_bg, loss_gt, loss_eh):
        lv_bg = self._clamp(self.log_var_bg)
        lv_gt = self._clamp(self.log_var_gt)
        lv_eh = self._clamp(self.log_var_eh)
        w_bg = torch.exp(-lv_bg)
        w_gt = torch.exp(-lv_gt)
        w_eh = torch.exp(-lv_eh)
        total = (w_bg * loss_bg + 0.5 * lv_bg +
                 w_gt * loss_gt + 0.5 * lv_gt +
                 w_eh * loss_eh + 0.5 * lv_eh)
        return total, (w_bg.item(), w_gt.item(), w_eh.item())

    def get_weights(self):
        lv_bg = self._clamp(self.log_var_bg)
        lv_gt = self._clamp(self.log_var_gt)
        lv_eh = self._clamp(self.log_var_eh)
        return (torch.exp(-lv_bg).item(),
                torch.exp(-lv_gt).item(),
                torch.exp(-lv_eh).item())


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, log_probs, targets):
        n_classes = log_probs.size(1)
        if self.smoothing > 0:
            one_hot = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1.0)
            smooth = (one_hot * (1 - self.smoothing) +
                      (1 - one_hot) * self.smoothing / (n_classes - 1))
            loss = -(smooth * log_probs).sum(dim=1)
            if self.alpha is not None:
                loss = loss * self.alpha[targets]
            return loss.mean()
        log_pt = log_probs[torch.arange(len(targets)), targets].clamp(min=-100)
        pt = torch.exp(log_pt)
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


class MixedRegressionLoss(nn.Module):
    def __init__(self, mse_weight=0.5, reweight_scale=0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.reweight_scale = reweight_scale

    def forward(self, pred, target, raw_bg=None):
        diff = pred - target
        loss = self.mse_weight * diff.pow(2) + (1 - self.mse_weight) * diff.abs()
        if raw_bg is not None and self.reweight_scale > 0:
            weights = 1.0 + self.reweight_scale * (raw_bg.abs() / 5.0).clamp(0, 1)
            loss = loss * weights.view_as(loss)
        return loss.mean()


class Normalizer:
    def __init__(self, tensor=None, log_transform=False, robust=True):
        self.log_transform = log_transform
        self.robust = robust
        if tensor is not None:
            if log_transform:
                tensor = torch.log1p(tensor.clamp(min=0))
            if robust:
                self.center = torch.median(tensor).item()
                q25 = torch.quantile(tensor, 0.25).item()
                q75 = torch.quantile(tensor, 0.75).item()
                self.scale = max(q75 - q25, 1e-8)
            else:
                self.center = torch.mean(tensor).item()
                self.scale = max(torch.std(tensor).item(), 1e-8)
        else:
            self.center, self.scale = 0.0, 1.0

    def norm(self, t):
        if self.log_transform:
            t = torch.log1p(t.clamp(min=0))
        return (t - self.center) / self.scale

    def denorm(self, t):
        t = t * self.scale + self.center
        if self.log_transform:
            t = torch.expm1(t)
        return t

    def state_dict(self):
        return {'center': self.center, 'scale': self.scale,
                'log_transform': self.log_transform, 'robust': self.robust}

    def load_state_dict(self, d):
        self.center = d['center']
        self.scale = d['scale']
        self.log_transform = d.get('log_transform', False)
        self.robust = d.get('robust', True)


def _map_gaptype(gt, merge=True):
    if merge and int(gt) == 2:
        return 1
    return int(gt)


# ---------------------------------------------------------------------------
# Custom multi-task dataset using ALIGNN graph construction
# ---------------------------------------------------------------------------
class ALIGNNMultiTaskDataset:
    """Pre-build DGL graphs + line graphs for multi-task ALIGNN."""

    def __init__(self, data_dir, id_prop_data, merge_metal_indirect=True,
                 cutoff=8.0, max_neighbors=12, atom_features='cgcnn',
                 use_canonize=False):
        self.id_prop_data = id_prop_data
        self.merge_metal_indirect = merge_metal_indirect

        print(f'Building {len(id_prop_data)} DGL graphs...')
        self.graphs = []
        self.line_graphs = []
        self.bg_targets = []
        self.gt_targets = []
        self.eh_targets = []

        t0 = time.time()
        for i, row in enumerate(tqdm(id_prop_data)):
            cif_id, bg, gt, eh = row[0], float(row[1]), int(row[2]), float(row[3])

            cif_path = os.path.join(data_dir, cif_id + '.cif')
            atoms = Atoms.from_cif(cif_path, use_cif2cell=False)

            g, lg = Graph.atom_dgl_multigraph(
                atoms,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                atom_features=atom_features,
                compute_line_graph=True,
                use_canonize=use_canonize,
            )
            lg.apply_edges(compute_bond_cosines)

            self.graphs.append(g)
            self.line_graphs.append(lg)
            self.bg_targets.append(bg)
            self.gt_targets.append(_map_gaptype(gt, merge_metal_indirect))
            self.eh_targets.append(eh)

            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(id_prop_data)} ({time.time()-t0:.0f}s)')

        print(f'Graphs built in {time.time()-t0:.1f}s')

        # Standardize atom features
        all_fea = torch.cat([g.ndata['atom_features'] for g in self.graphs])
        self.fea_mean = all_fea.mean(0)
        self.fea_std = all_fea.std(0)
        self.fea_std[self.fea_std == 0] = 1.0
        for g in self.graphs:
            g.ndata['atom_features'] = (
                (g.ndata['atom_features'] - self.fea_mean) / self.fea_std)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (self.graphs[idx],
                self.line_graphs[idx],
                self.bg_targets[idx],
                self.gt_targets[idx],
                self.eh_targets[idx])

    def get_class_weights(self):
        counts = {}
        for gt in self.gt_targets:
            counts[gt] = counts.get(gt, 0) + 1
        n_classes = max(counts.keys()) + 1
        total = sum(counts.values())
        weights = [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)]
        return torch.FloatTensor(weights)

    @property
    def n_gap_classes(self):
        return 2 if self.merge_metal_indirect else 3


def collate_mt(samples):
    """Collate multi-task samples into batched graphs."""
    graphs, line_graphs, bgs, gts, ehs = zip(*samples)
    bg = dgl.batch(graphs)
    blg = dgl.batch(line_graphs)
    return (bg, blg,
            torch.tensor(bgs, dtype=torch.float32),
            torch.tensor(gts, dtype=torch.long),
            torch.tensor(ehs, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Oversample minority class (GT) for training
# ---------------------------------------------------------------------------
class WeightedSubsetSampler:
    """Samples from train_idx with weights inverse to GT class frequency."""
    def __init__(self, train_idx, gt_targets):
        self.train_idx = train_idx
        cnt = Counter(gt_targets[i] for i in train_idx)
        weights = [1.0 / cnt[gt_targets[i]] for i in train_idx]
        self.weights = torch.tensor(weights, dtype=torch.double)
        self._inner = WeightedRandomSampler(self.weights, len(train_idx))

    def __iter__(self):
        idx_t = torch.tensor(self.train_idx, dtype=torch.long)
        for j in self._inner:
            yield idx_t[j].item()

    def __len__(self):
        return len(self.train_idx)


# ---------------------------------------------------------------------------
# Stratified K-fold
# ---------------------------------------------------------------------------
def stratified_kfold(id_prop_data, n_folds=3, seed=123, merge=True):
    rng = random.Random(seed)
    groups = {}
    for i, row in enumerate(id_prop_data):
        label = _map_gaptype(row[2], merge)
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


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def run_epoch(loader, model, criterion_bg, criterion_cls, criterion_eh,
              norm_bg, norm_eh, args, optimizer=None, mt_loss=None,
              is_train=True, epoch=0):
    model.train() if is_train else model.eval()

    total_loss, n_batch = 0.0, 0
    all_bg_p, all_bg_t, all_gt_logp, all_gt_t, all_eh_p, all_eh_t = \
        [], [], [], [], [], []

    for batch in loader:
        bg, blg, bg_true, gt_true, eh_true = batch
        bg = bg.to(device)
        blg = blg.to(device)
        bg_true = bg_true.to(device)
        gt_true = gt_true.to(device)
        eh_true = eh_true.to(device)

        t_bg = norm_bg.norm(bg_true)
        t_eh = norm_eh.norm(eh_true)

        # ALIGNN forward expects (g, lg, lattice) tuple
        # We pass dummy lattice (not used by our model)
        dummy_lat = torch.zeros(1)
        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            p_bg, p_gt, p_eh = model((bg, blg, dummy_lat))

            loss_bg = criterion_bg(p_bg, t_bg, bg_true)
            loss_gt = criterion_cls(p_gt, gt_true)
            loss_eh = criterion_eh(p_eh, t_eh)

            w_bg, w_gt, w_eh = args.task_weights
            static_loss = w_bg * loss_bg + w_gt * loss_gt + w_eh * loss_eh

            if (mt_loss is not None and is_train
                    and epoch > args.dw_warmup):
                dyn_loss, _ = mt_loss(loss_bg, loss_gt, loss_eh)
                alpha = min((epoch - args.dw_warmup) / 20.0, 1.0)
                loss = (1 - alpha) * static_loss + alpha * dyn_loss
            else:
                loss = static_loss

        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                params = list(model.parameters())
                if mt_loss is not None:
                    params += list(mt_loss.parameters())
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batch += 1
        all_bg_p.append(norm_bg.denorm(p_bg.detach().cpu()))
        all_bg_t.append(bg_true.cpu())
        all_gt_logp.append(p_gt.detach().cpu())
        all_gt_t.append(gt_true.cpu())
        all_eh_p.append(norm_eh.denorm(p_eh.detach().cpu()))
        all_eh_t.append(eh_true.cpu())

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_logp = torch.cat(all_gt_logp)
    gt_t = torch.cat(all_gt_t).numpy()
    eh_p = torch.cat(all_eh_p).numpy().flatten()
    eh_t = torch.cat(all_eh_t).numpy().flatten()

    gt_pred = gt_logp.argmax(dim=1).numpy()
    gt_pred[bg_p < 0.05] = 1  # rule: low BG → Indirect+Metal

    return total_loss / max(n_batch, 1), {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_pred)),
        'gt_f1': float(f1_score(gt_t, gt_pred, average='binary',
                                zero_division=0)),
        'eh_mae': float(np.mean(np.abs(eh_p - eh_t))),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Multi-task ALIGNN')
    p.add_argument('data_dir', help='Data dir with id_prop.csv + CIF files')
    p.add_argument('--output-dir', default='checkpoints/alignn_mt')
    p.add_argument('--epochs', default=300, type=int)
    p.add_argument('--batch-size', default=64, type=int)
    p.add_argument('--lr', default=0.001, type=float)
    p.add_argument('--weight-decay', default=1e-5, type=float)
    p.add_argument('--dropout', default=0.3, type=float)
    p.add_argument('--grad-clip', default=3.0, type=float)
    p.add_argument('--patience', default=80, type=int)
    p.add_argument('--task-weights', nargs=3, type=float,
                   default=[3.0, 4.0, 1.5])
    p.add_argument('--dw-warmup', default=30, type=int)
    p.add_argument('--alignn-layers', default=4, type=int)
    p.add_argument('--gcn-layers', default=4, type=int)
    p.add_argument('--hidden-features', default=256, type=int)
    p.add_argument('--embedding-features', default=64, type=int)
    p.add_argument('--cutoff', default=8.0, type=float)
    p.add_argument('--max-neighbors', default=12, type=int)
    p.add_argument('--k-folds', default=3, type=int)
    p.add_argument('--n-seeds', default=2, type=int)
    p.add_argument('--print-freq', default=10, type=int)
    p.add_argument('--num-workers', default=0, type=int)
    p.add_argument('--warmup-epochs', default=10, type=int)
    # Improvement options (see docs/improvement_strategies.md)
    p.add_argument('--oversample-gt', action='store_true',
                   help='Use weighted sampler to oversample minority gap type')
    p.add_argument('--composite-gt-acc', default=0.0, type=float,
                   help='Add this * gt_acc to composite (e.g. 0.5 to balance F1 and Acc)')
    p.add_argument('--focal-gamma', default=2.0, type=float,
                   help='Focal loss gamma (higher = more focus on hard examples)')
    p.add_argument('--minority-weight', default=1.5, type=float,
                   help='Extra multiplier for minority class in classification')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    print(f'Device: {device}')

    # Load id_prop data
    id_prop_file = os.path.join(args.data_dir, 'id_prop.csv')
    with open(id_prop_file) as f:
        id_prop_data = list(csv.reader(f))
    random.seed(123)
    random.shuffle(id_prop_data)
    print(f'Loaded {len(id_prop_data)} samples')

    counts = {}
    for row in id_prop_data:
        label = _map_gaptype(row[2], True)
        counts[label] = counts.get(label, 0) + 1
    print(f'Class distribution: {dict(sorted(counts.items()))}')

    # Build ALL graphs once (shared across folds)
    dataset = ALIGNNMultiTaskDataset(
        args.data_dir, id_prop_data,
        merge_metal_indirect=True,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        atom_features='cgcnn',
        use_canonize=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    folds = stratified_kfold(id_prop_data, args.k_folds, seed=123, merge=True)
    all_results = {'composite': [], 'bg_mae': [], 'gt_f1': []}
    ensemble_results = []

    for fi in range(args.k_folds):
        test_idx = folds[fi]
        val_idx = folds[(fi + 1) % args.k_folds]
        train_idx = []
        for j in range(args.k_folds):
            if j not in (fi, (fi + 1) % args.k_folds):
                train_idx.extend(folds[j])

        print(f'\n{"="*60}')
        print(f'Fold {fi+1}/{args.k_folds} | Train {len(train_idx)} '
              f'Val {len(val_idx)} Test {len(test_idx)}')
        print(f'{"="*60}')

        kw = dict(batch_size=args.batch_size, collate_fn=collate_mt,
                  num_workers=args.num_workers, pin_memory=True)
        if getattr(args, 'oversample_gt', False):
            train_sampler = WeightedSubsetSampler(train_idx, dataset.gt_targets)
            train_loader = DataLoader(dataset, sampler=train_sampler, **kw)
        else:
            train_loader = DataLoader(dataset,
                                      sampler=SubsetRandomSampler(train_idx), **kw)
        val_loader = DataLoader(dataset,
                                sampler=SubsetRandomSampler(val_idx), **kw)
        test_loader = DataLoader(dataset,
                                 sampler=SubsetRandomSampler(test_idx), **kw)

        fold_models, fold_norms = [], []

        for s in range(args.n_seeds):
            seed = 42 + s * 137
            print(f'\n--- Seed {seed} ---')
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Normalizers
            tr_bg = torch.tensor([dataset.bg_targets[i] for i in train_idx])
            tr_eh = torch.tensor([dataset.eh_targets[i] for i in train_idx])
            norm_bg = Normalizer(tr_bg, log_transform=False, robust=True)
            norm_eh = Normalizer(tr_eh, log_transform=True, robust=True)

            # Model
            config = ALIGNNConfig(
                name="alignn",
                alignn_layers=args.alignn_layers,
                gcn_layers=args.gcn_layers,
                atom_input_features=92,
                edge_input_features=80,
                triplet_input_features=40,
                embedding_features=args.embedding_features,
                hidden_features=args.hidden_features,
                output_features=1,
            )
            model = ALIGNNMultiTask(
                config, n_gap_classes=2,
                dropout=args.dropout).to(device)

            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            print(f'  Model params: {n_params:,}')

            # Losses
            criterion_bg = MixedRegressionLoss(mse_weight=0.5,
                                               reweight_scale=0.5)
            cw = dataset.get_class_weights().to(device)
            cw[0] = cw[0] * args.minority_weight
            print(f'  Class weights: {cw.tolist()}')
            criterion_cls = FocalLossWithSmoothing(
                alpha=cw, gamma=args.focal_gamma, smoothing=0.05)
            criterion_eh = MixedRegressionLoss(mse_weight=0.5,
                                               reweight_scale=0.0)

            mt_loss = DynamicMultiTaskLoss().to(device)
            opt_params = (list(model.parameters()) +
                          list(mt_loss.parameters()))
            optimizer = optim.AdamW(opt_params, lr=args.lr,
                                    weight_decay=args.weight_decay)

            # OneCycleLR
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr,
                steps_per_epoch=len(train_loader),
                epochs=args.epochs)

            fold_dir = os.path.join(args.output_dir,
                                    f'fold_{fi}_seed_{seed}')
            os.makedirs(fold_dir, exist_ok=True)

            best_composite = -float('inf')
            best_bg_mae = float('inf')
            best_gt_f1 = -float('inf')
            best_ep = 0
            no_improve = 0

            for epoch in range(1, args.epochs + 1):
                t_loss, t_m = run_epoch(
                    train_loader, model, criterion_bg, criterion_cls,
                    criterion_eh, norm_bg, norm_eh, args,
                    optimizer=optimizer, mt_loss=mt_loss,
                    is_train=True, epoch=epoch)
                v_loss, v_m = run_epoch(
                    val_loader, model, criterion_bg, criterion_cls,
                    criterion_eh, norm_bg, norm_eh, args,
                    is_train=False, epoch=epoch)

                composite = (-2.0 * v_m['bg_mae'] + 2.0 * v_m['gt_f1']
                             - v_m['eh_mae']
                             + args.composite_gt_acc * v_m['gt_acc'])

                ckpt = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'norm_bg': norm_bg.state_dict(),
                    'norm_eh': norm_eh.state_dict(),
                }

                if composite > best_composite:
                    best_composite = composite
                    best_ep = epoch
                    no_improve = 0
                    torch.save(ckpt, os.path.join(
                        fold_dir, 'best_composite.pth.tar'))
                else:
                    no_improve += 1

                if v_m['bg_mae'] < best_bg_mae:
                    best_bg_mae = v_m['bg_mae']
                    torch.save(ckpt, os.path.join(
                        fold_dir, 'best_bg_mae.pth.tar'))

                if v_m['gt_f1'] > best_gt_f1:
                    best_gt_f1 = v_m['gt_f1']
                    torch.save(ckpt, os.path.join(
                        fold_dir, 'best_gt_f1.pth.tar'))

                if epoch % args.print_freq == 0 or epoch == 1:
                    dw = mt_loss.get_weights()
                    lr_now = optimizer.param_groups[0]['lr']
                    print(f'  Ep {epoch:03d} lr={lr_now:.5f} '
                          f'dw=[{dw[0]:.2f},{dw[1]:.2f},{dw[2]:.2f}] | '
                          f'TrL {t_loss:.3f} VL {v_loss:.3f} | '
                          f'BG {v_m["bg_mae"]:.4f} GT {v_m["gt_acc"]:.3f}/'
                          f'{v_m["gt_f1"]:.3f} '
                          f'EH {v_m["eh_mae"]:.4f} | comp={composite:.3f}')

                if args.patience > 0 and no_improve >= args.patience:
                    print(f'  Early stop at epoch {epoch} '
                          f'(best comp ep {best_ep})')
                    break

            # Test
            for tag in ['composite', 'bg_mae', 'gt_f1']:
                ckpt_path = os.path.join(fold_dir, f'best_{tag}.pth.tar')
                if os.path.isfile(ckpt_path):
                    ckpt = torch.load(ckpt_path, weights_only=False)
                    model.load_state_dict(ckpt['state_dict'])
                    norm_bg.load_state_dict(ckpt['norm_bg'])
                    norm_eh.load_state_dict(ckpt['norm_eh'])
                    _, test_m = run_epoch(
                        test_loader, model, criterion_bg, criterion_cls,
                        criterion_eh, norm_bg, norm_eh, args,
                        is_train=False)
                    all_results[tag].append(test_m)
                    if tag == 'composite':
                        print(f'  Test ({tag}): BG {test_m["bg_mae"]:.4f} '
                              f'GT {test_m["gt_acc"]:.4f}/'
                              f'{test_m["gt_f1"]:.4f} '
                              f'EH {test_m["eh_mae"]:.4f}')
                    elif tag == 'gt_f1':
                        print(f'  Test (best_gt_f1): BG {test_m["bg_mae"]:.4f} '
                              f'GT {test_m["gt_acc"]:.4f}/{test_m["gt_f1"]:.4f} '
                              f'EH {test_m["eh_mae"]:.4f}')

    # Summary
    print(f'\n{"="*60}')
    print(f'{args.k_folds}-Fold CV Summary (ALIGNN-MT)')
    print(f'{"="*60}')
    for tag in ['composite', 'bg_mae', 'gt_f1']:
        if all_results[tag]:
            print(f'\n  Checkpoint: best_{tag}')
            for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']:
                vals = [m[key] for m in all_results[tag]]
                unit = (' eV' if key == 'bg_mae'
                        else (' eV/atom' if key == 'eh_mae' else ''))
                print(f'    {key:8s}: {np.mean(vals):.4f} '
                      f'+/- {np.std(vals):.4f}{unit}')

    ref = {k: np.mean([m[k] for m in all_results['composite']])
           for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']}
    print(f'\n{"="*60}')
    print('Target check (best_composite):')
    for desc, met, val in [
        ('BG MAE < 0.5', ref['bg_mae'] < 0.5, f'{ref["bg_mae"]:.4f}'),
        ('GT Acc > 0.8', ref['gt_acc'] > 0.8, f'{ref["gt_acc"]:.4f}'),
        ('GT F1  > 0.85', ref['gt_f1'] > 0.85, f'{ref["gt_f1"]:.4f}'),
        ('EH MAE < 0.1', ref['eh_mae'] < 0.1, f'{ref["eh_mae"]:.4f}'),
    ]:
        status = 'PASS' if met else 'FAIL'
        print(f'  [{status}] {desc}: {val}')
    print(f'{"="*60}')
    print(f'\nTotal time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
