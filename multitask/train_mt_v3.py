"""
Multi-task CGCNN v3 training with comprehensive optimizations.

Improvements over v2:
  - Right-sized model (V1-level backbone + attention pooling)
  - Hierarchical prediction (bandgap → gap type conditioning)
  - Focal Loss for classification (handles class imbalance)
  - Huber Loss for regression (robust to outliers)
  - Fixed task weights (no unstable uncertainty weighting)
  - Multi-metric model selection (composite + best-GT-F1 checkpoints)
  - Physical consistency regularization (bandgap ↔ gap type)
  - Feature-space augmentation (Gaussian noise on features)
  - Stratified K-fold cross-validation
"""
import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))

from data_mt import CIFDataMultiTask, collate_pool_mt, stratified_split
from model_mt_v3 import CrystalGraphConvNetMTV3


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017) for class-imbalanced classification.
    Expects log-probabilities (after LogSoftmax) as input."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, log_probs, targets):
        log_pt = log_probs[torch.arange(len(targets)), targets].clamp(min=-100)
        pt = torch.exp(log_pt)
        focal_weight = (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = torch.mean(tensor).item()
            self.std = max(torch.std(tensor).item(), 1e-8)
        else:
            self.mean, self.std = 0.0, 1.0

    def norm(self, t):
        return (t - self.mean) / self.std

    def denorm(self, t):
        return t * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, d):
        self.mean, self.std = d['mean'], d['std']


# ---------------------------------------------------------------------------
# Physical consistency loss
# ---------------------------------------------------------------------------
def physical_consistency_loss(bg_denorm, gt_log_probs, metal_class=2,
                              threshold=0.1):
    """Penalize predictions where bandgap and gap-type are physically
    inconsistent (e.g. bg≈0 but not predicting Metal)."""
    probs = torch.exp(gt_log_probs)
    metal_prob = probs[:, metal_class]
    bg = bg_denorm.squeeze(-1)
    # Smooth indicator: ≈1 when bg < threshold (should be metal)
    should_metal = torch.sigmoid(-10.0 * (bg - threshold))
    loss = should_metal * (1 - metal_prob) + (1 - should_metal) * metal_prob
    return loss.mean()


# ---------------------------------------------------------------------------
# Stratified K-fold
# ---------------------------------------------------------------------------
def stratified_kfold(id_prop_data, n_folds=5, random_seed=123):
    rng = random.Random(random_seed)
    groups = {}
    for i, row in enumerate(id_prop_data):
        label = int(row[2])
        groups.setdefault(label, []).append(i)
    folds = [[] for _ in range(n_folds)]
    for label in sorted(groups.keys()):
        indices = groups[label][:]
        rng.shuffle(indices)
        for i, idx in enumerate(indices):
            folds[i % n_folds].append(idx)
    for fold in folds:
        rng.shuffle(fold)
    return folds


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Multi-task CGCNN v3')
    p.add_argument('data_dir', help='Path to data directory')
    p.add_argument('--epochs', default=300, type=int)
    p.add_argument('-b', '--batch-size', default=128, type=int)
    p.add_argument('--lr', default=0.003, type=float)
    p.add_argument('--weight-decay', default=1e-4, type=float)
    p.add_argument('--optim', default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    p.add_argument('--atom-fea-len', default=64, type=int)
    p.add_argument('--h-fea-len', default=128, type=int)
    p.add_argument('--n-conv', default=4, type=int)
    p.add_argument('--n-h', default=1, type=int)
    p.add_argument('--dropout', default=0.15, type=float)
    p.add_argument('-j', '--workers', default=0, type=int)
    p.add_argument('--print-freq', default=20, type=int)
    p.add_argument('--disable-cuda', action='store_true')
    p.add_argument('--patience', default=50, type=int)
    p.add_argument('--grad-clip', default=5.0, type=float)
    # Task weights
    p.add_argument('--task-weights', nargs=3, type=float,
                   default=[1.0, 5.0, 1.0],
                   help='Fixed weights for [bandgap, gaptype, ehull]')
    # Focal loss
    p.add_argument('--focal-gamma', default=2.0, type=float)
    # Physical consistency
    p.add_argument('--consistency-weight', default=0.1, type=float)
    p.add_argument('--metal-class', default=2, type=int)
    # Augmentation
    p.add_argument('--augment-noise', default=0.05, type=float,
                   help='Gaussian noise std for feature augmentation (0=off)')
    # K-fold CV
    p.add_argument('--k-folds', default=5, type=int,
                   help='Number of CV folds (1 = single 80/10/10 split)')
    p.add_argument('--train-ratio', default=0.8, type=float)
    p.add_argument('--val-ratio', default=0.1, type=float)
    p.add_argument('--test-ratio', default=0.1, type=float)
    p.add_argument('--checkpoint-dir', default='checkpoints/multitask_v3')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------
def run_epoch(loader, model, criterion_reg, criterion_cls,
              normalizer_bg, normalizer_eh, args, device,
              optimizer=None, is_train=True):
    model.train() if is_train else model.eval()
    tot_loss, n = 0.0, 0
    all_bg_p, all_bg_t = [], []
    all_gt_p, all_gt_t = [], []
    all_eh_p, all_eh_t = [], []
    w_bg, w_gt, w_eh = args.task_weights

    for inputs, targets, _ in loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        # Feature-space augmentation (training only)
        if is_train and args.augment_noise > 0:
            atom_fea = atom_fea + torch.randn_like(atom_fea) * args.augment_noise
            nbr_fea = nbr_fea + torch.randn_like(nbr_fea) * args.augment_noise

        t_bg = normalizer_bg.norm(targets['bandgap']).to(device)
        t_gt = targets['gaptype'].squeeze(-1).to(device)
        t_eh = normalizer_eh.norm(targets['ehull']).to(device)

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            p_bg, p_gt, p_eh = model(atom_fea, nbr_fea, nbr_fea_idx,
                                     crystal_atom_idx)
            loss_bg = criterion_reg(p_bg, t_bg)
            loss_gt = criterion_cls(p_gt, t_gt)
            loss_eh = criterion_reg(p_eh, t_eh)
            loss = w_bg * loss_bg + w_gt * loss_gt + w_eh * loss_eh

            if args.consistency_weight > 0:
                bg_denorm = normalizer_bg.denorm(p_bg.detach())
                cons = physical_consistency_loss(
                    bg_denorm, p_gt, metal_class=args.metal_class)
                loss = loss + args.consistency_weight * cons

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        tot_loss += loss.item()
        n += 1
        all_bg_p.append(normalizer_bg.denorm(p_bg.detach().cpu()))
        all_bg_t.append(targets['bandgap'])
        all_gt_p.append(p_gt.detach().cpu().argmax(dim=1))
        all_gt_t.append(targets['gaptype'].squeeze(-1))
        all_eh_p.append(normalizer_eh.denorm(p_eh.detach().cpu()))
        all_eh_t.append(targets['ehull'])

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_p = torch.cat(all_gt_p).numpy()
    gt_t = torch.cat(all_gt_t).numpy()
    eh_p = torch.cat(all_eh_p).numpy().flatten()
    eh_t = torch.cat(all_eh_t).numpy().flatten()

    return tot_loss / max(n, 1), {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_p)),
        'gt_f1':  float(f1_score(gt_t, gt_p, average='macro', zero_division=0)),
        'eh_mae': float(np.mean(np.abs(eh_p - eh_t))),
    }


# ---------------------------------------------------------------------------
# Train one fold
# ---------------------------------------------------------------------------
def train_fold(dataset, train_idx, val_idx, test_idx, fold_idx, args, device):
    # Data loaders
    kw = dict(batch_size=args.batch_size, collate_fn=collate_pool_mt,
              num_workers=args.workers, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), **kw)
    val_loader   = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **kw)
    test_loader  = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    # Normalizers from training split only
    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in train_idx])
    tr_eh = torch.tensor([float(dataset.id_prop_data[i][3]) for i in train_idx])
    n_bg = Normalizer(tr_bg)
    n_eh = Normalizer(tr_eh)
    print(f'  Normalizers - BG: mean={n_bg.mean:.4f} std={n_bg.std:.4f} | '
          f'EH: mean={n_eh.mean:.4f} std={n_eh.std:.4f}')

    # Model
    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV3(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=3, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model params: {n_params:,}')

    # Losses
    criterion_reg = nn.SmoothL1Loss()
    class_weights = dataset.get_class_weights().to(device)
    criterion_cls = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
    print(f'  Class weights: {class_weights.tolist()}')
    print(f'  Task weights:  BG={args.task_weights[0]} '
          f'GT={args.task_weights[1]} EH={args.task_weights[2]}')

    # Optimizer & scheduler
    if args.optim == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr * 0.01)

    fold_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    best_composite, best_gt_f1 = -float('inf'), -float('inf')
    best_ep_comp, best_ep_f1 = 0, 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_m = run_epoch(train_loader, model, criterion_reg, criterion_cls,
                                n_bg, n_eh, args, device, optimizer, is_train=True)
        v_loss, v_m = run_epoch(val_loader, model, criterion_reg, criterion_cls,
                                n_bg, n_eh, args, device, is_train=False)
        scheduler.step()

        composite = -v_m['bg_mae'] + 2.0 * v_m['gt_f1'] - v_m['eh_mae']

        improved = ''
        ckpt = {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer_bg': n_bg.state_dict(),
                'normalizer_eh': n_eh.state_dict(), 'args': vars(args)}

        if composite > best_composite:
            best_composite = composite
            best_ep_comp = epoch
            no_improve = 0
            torch.save(ckpt, os.path.join(fold_dir, 'best_composite.pth.tar'))
            improved += ' *comp'
        else:
            no_improve += 1

        if v_m['gt_f1'] > best_gt_f1:
            best_gt_f1 = v_m['gt_f1']
            best_ep_f1 = epoch
            torch.save(ckpt, os.path.join(fold_dir, 'best_gt_f1.pth.tar'))
            improved += ' *f1'

        if epoch % args.print_freq == 0 or epoch == 1 or improved:
            print(f'  Ep {epoch:03d}/{args.epochs} | '
                  f'TrL {t_loss:.3f} VL {v_loss:.3f} | '
                  f'BG {v_m["bg_mae"]:.4f} | '
                  f'GT {v_m["gt_acc"]:.3f}/{v_m["gt_f1"]:.3f} | '
                  f'EH {v_m["eh_mae"]:.4f}{improved}')

        if args.patience > 0 and no_improve >= args.patience:
            print(f'  Early stop at epoch {epoch} '
                  f'(no composite improvement for {args.patience} epochs)')
            break

    # Test with both best checkpoints
    results = {}
    for tag, ep in [('composite', best_ep_comp), ('gt_f1', best_ep_f1)]:
        ckpt_path = os.path.join(fold_dir, f'best_{tag}.pth.tar')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            n_bg.load_state_dict(ckpt['normalizer_bg'])
            n_eh.load_state_dict(ckpt['normalizer_eh'])
        _, test_m = run_epoch(test_loader, model, criterion_reg, criterion_cls,
                              n_bg, n_eh, args, device, is_train=False)
        results[tag] = test_m
        print(f'  Test [{tag}, ep {ep}] | '
              f'BG {test_m["bg_mae"]:.4f} eV | '
              f'GT Acc {test_m["gt_acc"]:.4f} F1 {test_m["gt_f1"]:.4f} | '
              f'EH {test_m["eh_mae"]:.4f} eV/atom')
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')

    dataset = CIFDataMultiTask(args.data_dir)
    print(f'Dataset: {len(dataset)} samples')

    # Class distribution & verification
    class_info = {}
    for row in dataset.id_prop_data:
        c = int(row[2])
        bg = float(row[1])
        class_info.setdefault(c, []).append(bg)
    for c in sorted(class_info):
        bgs = class_info[c]
        print(f'  Class {c}: n={len(bgs):4d}  '
              f'mean_bg={np.mean(bgs):.3f}  median_bg={np.median(bgs):.3f}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()

    if args.k_folds > 1:
        # ----- K-fold cross-validation -----
        folds = stratified_kfold(dataset.id_prop_data, args.k_folds)
        all_results = {'composite': [], 'gt_f1': []}

        for fi in range(args.k_folds):
            test_idx = folds[fi]
            val_idx = folds[(fi + 1) % args.k_folds]
            train_idx = []
            for j in range(args.k_folds):
                if j != fi and j != (fi + 1) % args.k_folds:
                    train_idx.extend(folds[j])

            print(f'\n{"="*60}')
            print(f'Fold {fi+1}/{args.k_folds}  |  '
                  f'Train {len(train_idx)} | Val {len(val_idx)} | '
                  f'Test {len(test_idx)}')
            print(f'{"="*60}')
            results = train_fold(dataset, train_idx, val_idx, test_idx,
                                 fi, args, device)
            for tag in all_results:
                all_results[tag].append(results[tag])

        # Summary
        print(f'\n{"="*60}')
        print(f'{args.k_folds}-Fold CV Summary')
        print(f'{"="*60}')
        for tag in ['composite', 'gt_f1']:
            metrics_list = all_results[tag]
            print(f'\n  Checkpoint: best_{tag}')
            for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']:
                vals = [m[key] for m in metrics_list]
                unit = ' eV' if key == 'bg_mae' else (
                    ' eV/atom' if key == 'eh_mae' else '')
                print(f'    {key:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{unit}')

    else:
        # ----- Single split -----
        train_idx, val_idx, test_idx = stratified_split(
            dataset.id_prop_data, args.train_ratio, args.val_ratio,
            args.test_ratio)
        print(f'\nSingle split: Train {len(train_idx)} | '
              f'Val {len(val_idx)} | Test {len(test_idx)}')
        train_fold(dataset, train_idx, val_idx, test_idx, 0, args, device)

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
