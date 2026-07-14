"""
Multi-task CGCNN v4: Metal→Indirect merge + all improvements.

Improvements over v3:
  - Merge Metal into Indirect (binary: Direct vs Indirect+Metal)
  - Structure augmentation (coordinate perturb + lattice scale)
  - Multi-seed ensemble training & evaluation
  - Label smoothing for classification
  - LR warmup + cosine decay
  - Physical consistency: bg≈0 -> predict Indirect+Metal (class 1)
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))

from data_mt import collate_pool_mt
from data_mt_v4 import (
    CIFDataMultiTaskV4,
    stratified_split_v4,
    stratified_kfold_v4,
    _map_gaptype,
)
from model_mt_v3 import CrystalGraphConvNetMTV3


# ---------------------------------------------------------------------------
# Focal Loss with Label Smoothing
# ---------------------------------------------------------------------------
class FocalLossWithSmoothing(nn.Module):
    """Focal Loss with optional label smoothing. Expects log-probs."""

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
            # Smooth labels: (1-smoothing) for true, smoothing/(K-1) for others
            one_hot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
            smooth = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_classes - 1)
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
# Physical consistency: bg≈0 -> predict class 1 (Indirect+Metal)
# ---------------------------------------------------------------------------
def physical_consistency_loss(bg_denorm, gt_log_probs, indirect_class=1, threshold=0.1):
    """When bandgap≈0 (metal), should predict Indirect+Metal (class 1)."""
    probs = torch.exp(gt_log_probs)
    indirect_prob = probs[:, indirect_class]
    bg = bg_denorm.squeeze(-1)
    should_indirect = torch.sigmoid(-10.0 * (bg - threshold))
    loss = should_indirect * (1 - indirect_prob) + (1 - should_indirect) * indirect_prob
    return loss.mean()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Multi-task CGCNN v4')
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
    p.add_argument('--task-weights', nargs=3, type=float, default=[1.0, 5.0, 1.0])
    p.add_argument('--focal-gamma', default=2.0, type=float)
    p.add_argument('--label-smoothing', default=0.1, type=float)
    p.add_argument('--consistency-weight', default=0.1, type=float)
    p.add_argument('--augment-noise', default=0.05, type=float)
    p.add_argument('--augment-structure', action='store_true',
                    help='Enable structure augmentation (perturb + lattice scale)')
    p.add_argument('--augment-perturb', default=0.02, type=float)
    p.add_argument('--augment-scale', default=0.01, type=float)
    p.add_argument('--warmup-epochs', default=5, type=int)
    p.add_argument('--merge-metal-indirect', action='store_true', default=True,
                    help='Merge Metal into Indirect (binary classification)')
    p.add_argument('--k-folds', default=5, type=int)
    p.add_argument('--n-seeds', default=3, type=int,
                    help='Number of seeds for ensemble (1=no ensemble)')
    p.add_argument('--train-ratio', default=0.8, type=float)
    p.add_argument('--val-ratio', default=0.1, type=float)
    p.add_argument('--test-ratio', default=0.1, type=float)
    p.add_argument('--checkpoint-dir', default='checkpoints/multitask_v4')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------
def run_epoch(loader, model, criterion_reg, criterion_cls,
              normalizer_bg, normalizer_eh, args, device, dataset,
              optimizer=None, is_train=True):
    model.train() if is_train else model.eval()
    if args.augment_structure and hasattr(dataset, 'augment'):
        dataset.augment = is_train

    tot_loss, n = 0.0, 0
    all_bg_p, all_bg_t = [], []
    all_gt_p, all_gt_t = [], []
    all_eh_p, all_eh_t = [], []
    w_bg, w_gt, w_eh = args.task_weights
    indirect_class = 1 if args.merge_metal_indirect else 2

    for inputs, targets, _ in loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        if is_train and args.augment_noise > 0:
            atom_fea = atom_fea + torch.randn_like(atom_fea) * args.augment_noise
            nbr_fea = nbr_fea + torch.randn_like(nbr_fea) * args.augment_noise

        t_bg = normalizer_bg.norm(targets['bandgap']).to(device)
        t_gt = targets['gaptype'].squeeze(-1).to(device)
        t_eh = normalizer_eh.norm(targets['ehull']).to(device)

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            p_bg, p_gt, p_eh = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            loss_bg = criterion_reg(p_bg, t_bg)
            loss_gt = criterion_cls(p_gt, t_gt)
            loss_eh = criterion_reg(p_eh, t_eh)
            loss = w_bg * loss_bg + w_gt * loss_gt + w_eh * loss_eh

            if args.consistency_weight > 0:
                bg_denorm = normalizer_bg.denorm(p_bg.detach())
                cons = physical_consistency_loss(
                    bg_denorm, p_gt, indirect_class=indirect_class)
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

    avg_type = 'binary' if args.merge_metal_indirect else 'macro'
    return tot_loss / max(n, 1), {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_p)),
        'gt_f1':  float(f1_score(gt_t, gt_p, average=avg_type, zero_division=0)),
        'eh_mae': float(np.mean(np.abs(eh_p - eh_t))),
    }


# ---------------------------------------------------------------------------
# Train one fold, one seed
# ---------------------------------------------------------------------------
def train_fold(dataset, train_idx, val_idx, test_idx, fold_idx, seed,
               args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    kw = dict(batch_size=args.batch_size, collate_fn=collate_pool_mt,
              num_workers=args.workers, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), **kw)
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **kw)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in train_idx])
    tr_eh = torch.tensor([float(dataset.id_prop_data[i][3]) for i in train_idx])
    n_bg = Normalizer(tr_bg)
    n_eh = Normalizer(tr_eh)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV3(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=dataset.n_gap_classes, dropout=args.dropout,
    ).to(device)

    criterion_reg = nn.SmoothL1Loss()
    class_weights = dataset.get_class_weights().to(device)
    criterion_cls = FocalLossWithSmoothing(
        alpha=class_weights, gamma=args.focal_gamma,
        smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                               eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, [warmup, cosine], [args.warmup_epochs])

    fold_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}_seed_{seed}')
    os.makedirs(fold_dir, exist_ok=True)

    best_composite, best_gt_f1 = -float('inf'), -float('inf')
    best_ep_comp, best_ep_f1 = 0, 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_m = run_epoch(train_loader, model, criterion_reg, criterion_cls,
                                n_bg, n_eh, args, device, dataset,
                                optimizer=optimizer, is_train=True)
        v_loss, v_m = run_epoch(val_loader, model, criterion_reg, criterion_cls,
                                n_bg, n_eh, args, device, dataset, is_train=False)
        scheduler.step()

        composite = -v_m['bg_mae'] + 2.0 * v_m['gt_f1'] - v_m['eh_mae']

        ckpt = {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer_bg': n_bg.state_dict(),
                'normalizer_eh': n_eh.state_dict(), 'args': vars(args)}

        if composite > best_composite:
            best_composite = composite
            best_ep_comp = epoch
            no_improve = 0
            torch.save(ckpt, os.path.join(fold_dir, 'best_composite.pth.tar'))
        else:
            no_improve += 1

        if v_m['gt_f1'] > best_gt_f1:
            best_gt_f1 = v_m['gt_f1']
            best_ep_f1 = epoch
            torch.save(ckpt, os.path.join(fold_dir, 'best_gt_f1.pth.tar'))

        if epoch % args.print_freq == 0 or epoch == 1:
            print(f'  Ep {epoch:03d} | TrL {t_loss:.3f} VL {v_loss:.3f} | '
                  f'BG {v_m["bg_mae"]:.4f} GT {v_m["gt_acc"]:.3f}/{v_m["gt_f1"]:.3f} '
                  f'EH {v_m["eh_mae"]:.4f}')

        if args.patience > 0 and no_improve >= args.patience:
            print(f'  Early stop at epoch {epoch}')
            break

    # Test
    results = {}
    for tag, ep in [('composite', best_ep_comp), ('gt_f1', best_ep_f1)]:
        ckpt_path = os.path.join(fold_dir, f'best_{tag}.pth.tar')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            n_bg.load_state_dict(ckpt['normalizer_bg'])
            n_eh.load_state_dict(ckpt['normalizer_eh'])
        _, test_m = run_epoch(test_loader, model, criterion_reg, criterion_cls,
                              n_bg, n_eh, args, device, dataset, is_train=False)
        results[tag] = test_m
    return results, fold_dir, model, n_bg, n_eh, criterion_reg, criterion_cls


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------
def eval_ensemble(models, norm_bg_list, norm_eh_list, test_loader, args, device):
    """Average regression, vote for classification."""
    all_bg_p, all_gt_p, all_eh_p = [], [], []
    all_bg_t, all_gt_t, all_eh_t = [], [], []

    for inputs, targets, _ in test_loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        bg_preds, gt_preds, eh_preds = [], [], []
        for m, nb, ne in zip(models, norm_bg_list, norm_eh_list):
            m.eval()
            with torch.no_grad():
                p_bg, p_gt, p_eh = m(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            bg_preds.append(nb.denorm(p_bg.cpu()))
            gt_preds.append(p_gt.argmax(dim=1).cpu())
            eh_preds.append(ne.denorm(p_eh.cpu()))

        all_bg_p.append(torch.stack(bg_preds).mean(dim=0))
        all_gt_p.append(torch.stack(gt_preds).mode(dim=0)[0])  # vote
        all_eh_p.append(torch.stack(eh_preds).mean(dim=0))
        all_bg_t.append(targets['bandgap'])
        all_gt_t.append(targets['gaptype'].squeeze(-1))
        all_eh_t.append(targets['ehull'])

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_p = torch.cat(all_gt_p).numpy()
    gt_t = torch.cat(all_gt_t).numpy()
    eh_p = torch.cat(all_eh_p).numpy().flatten()
    eh_t = torch.cat(all_eh_t).numpy().flatten()

    avg_type = 'binary' if args.merge_metal_indirect else 'macro'
    return {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_p)),
        'gt_f1':  float(f1_score(gt_t, gt_p, average=avg_type, zero_division=0)),
        'eh_mae': float(np.mean(np.abs(eh_p - eh_t))),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')

    dataset = CIFDataMultiTaskV4(
        args.data_dir,
        merge_metal_indirect=args.merge_metal_indirect,
        augment_perturb=args.augment_perturb if args.augment_structure else 0,
        augment_scale=args.augment_scale if args.augment_structure else 0,
    )
    print(f'Dataset: {len(dataset)} samples')
    print(f'Merge Metal→Indirect: {args.merge_metal_indirect} '
          f'(n_classes={dataset.n_gap_classes})')
    print(f'Structure augmentation: {args.augment_structure}')

    counts = {}
    for row in dataset.id_prop_data:
        l = _map_gaptype(row[2], args.merge_metal_indirect)
        counts[l] = counts.get(l, 0) + 1
    print(f'Class distribution: {dict(sorted(counts.items()))}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()

    if args.k_folds > 1:
        folds = stratified_kfold_v4(
            dataset.id_prop_data, args.k_folds,
            merge_metal_indirect=args.merge_metal_indirect)
        all_results = {'composite': [], 'gt_f1': []}
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

            fold_models, fold_norms = [], []
            test_loader = DataLoader(
                dataset, sampler=SubsetRandomSampler(test_idx),
                batch_size=args.batch_size, collate_fn=collate_pool_mt,
                num_workers=args.workers)

            for s in range(args.n_seeds):
                seed = 42 + s
                print(f'\n--- Seed {seed} ---')
                results, fold_dir, model, n_bg, n_eh, _, _ = train_fold(
                    dataset, train_idx, val_idx, test_idx, fi, seed, args, device)
                for tag in all_results:
                    all_results[tag].append(results[tag])
                print(f'  Test composite: BG {results["composite"]["bg_mae"]:.4f} '
                      f'GT Acc {results["composite"]["gt_acc"]:.4f} '
                      f'F1 {results["composite"]["gt_f1"]:.4f}')
                print(f'  Test gt_f1:    BG {results["gt_f1"]["bg_mae"]:.4f} '
                      f'GT Acc {results["gt_f1"]["gt_acc"]:.4f} '
                      f'F1 {results["gt_f1"]["gt_f1"]:.4f}')

                if args.n_seeds > 1:
                    ckpt = torch.load(
                        os.path.join(fold_dir, 'best_gt_f1.pth.tar'),
                        weights_only=False)
                    model.load_state_dict(ckpt['state_dict'])
                    n_bg.load_state_dict(ckpt['normalizer_bg'])
                    n_eh.load_state_dict(ckpt['normalizer_eh'])
                    fold_models.append(model)
                    fold_norms.append((n_bg, n_eh))

            if args.n_seeds > 1 and fold_models:
                ens = eval_ensemble(
                    fold_models, [x[0] for x in fold_norms], [x[1] for x in fold_norms],
                    test_loader, args, device)
                ensemble_results.append(ens)
                print(f'\n  Ensemble Test: BG {ens["bg_mae"]:.4f} '
                      f'GT Acc {ens["gt_acc"]:.4f} F1 {ens["gt_f1"]:.4f} '
                      f'EH {ens["eh_mae"]:.4f} eV/atom')

        print(f'\n{"="*60}')
        print(f'{args.k_folds}-Fold CV Summary')
        print(f'{"="*60}')
        for tag in ['composite', 'gt_f1']:
            metrics_list = all_results[tag]
            print(f'\n  Checkpoint: best_{tag} (per-seed mean)')
            for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']:
                vals = [m[key] for m in metrics_list]
                unit = ' eV' if key == 'bg_mae' else (' eV/atom' if key == 'eh_mae' else '')
                print(f'    {key:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{unit}')
        if ensemble_results:
            print(f'\n  Ensemble (all seeds):')
            for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']:
                vals = [m[key] for m in ensemble_results]
                unit = ' eV' if key == 'bg_mae' else (' eV/atom' if key == 'eh_mae' else '')
                print(f'    {key:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{unit}')
    else:
        train_idx, val_idx, test_idx = stratified_split_v4(
            dataset.id_prop_data, args.train_ratio, args.val_ratio, args.test_ratio,
            merge_metal_indirect=args.merge_metal_indirect)
        train_fold(dataset, train_idx, val_idx, test_idx, 0, 42, args, device)

    print(f'\nTotal time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
