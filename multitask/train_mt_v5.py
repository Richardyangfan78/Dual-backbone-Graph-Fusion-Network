"""
Multi-task CGCNN v5 — key changes over v4:

1. Log(1+BG) transform for band gap regression (reduces outlier impact)
2. Rule-based Metal detection: BG<0.01 → force class=1 (Indirect+Metal)
3. Larger model: 96 atom_fea, 256 h_fea, 5 conv, residual, multi-head attn
4. Reduced CV: 3-fold x 2-seed (fits in 12h walltime)
5. EH log(1+x) transform for right-skewed distribution
6. Mixup augmentation for classification
7. Cosine annealing with warm restarts
8. Test-time: rule-based post-processing for physical consistency
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))

from data_mt import collate_pool_mt
from data_mt_v4 import (
    CIFDataMultiTaskV4,
    stratified_split_v4,
    stratified_kfold_v4,
    _map_gaptype,
)
from model_mt_v5 import CrystalGraphConvNetMTV5


# ---------------------------------------------------------------------------
# Focal Loss with Label Smoothing
# ---------------------------------------------------------------------------
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
# Normalizer with log transform support
# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self, tensor=None, log_transform=False):
        self.log_transform = log_transform
        if tensor is not None:
            if log_transform:
                tensor = torch.log1p(tensor.clamp(min=0))
            self.mean = torch.mean(tensor).item()
            self.std = max(torch.std(tensor).item(), 1e-8)
        else:
            self.mean, self.std = 0.0, 1.0

    def norm(self, t):
        if self.log_transform:
            t = torch.log1p(t.clamp(min=0))
        return (t - self.mean) / self.std

    def denorm(self, t):
        t = t * self.std + self.mean
        if self.log_transform:
            t = torch.expm1(t)
        return t

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std, 'log_transform': self.log_transform}

    def load_state_dict(self, d):
        self.mean, self.std = d['mean'], d['std']
        self.log_transform = d.get('log_transform', False)


# ---------------------------------------------------------------------------
# Physical consistency loss
# ---------------------------------------------------------------------------
def physical_consistency_loss(bg_denorm, gt_log_probs, indirect_class=1, threshold=0.1):
    probs = torch.exp(gt_log_probs)
    indirect_prob = probs[:, indirect_class]
    bg = bg_denorm.squeeze(-1)
    should_indirect = torch.sigmoid(-10.0 * (bg - threshold))
    loss = should_indirect * (1 - indirect_prob) + (1 - should_indirect) * indirect_prob
    return loss.mean()


# ---------------------------------------------------------------------------
# Rule-based post-processing
# ---------------------------------------------------------------------------
def apply_rules(bg_pred, gt_pred_probs, merge_metal_indirect=True):
    """Post-process: if BG < 0.05 eV, force GT = Indirect+Metal (class 1)."""
    gt_pred = gt_pred_probs.argmax(dim=1) if gt_pred_probs.dim() > 1 else gt_pred_probs
    bg = bg_pred.squeeze(-1)
    metal_mask = bg < 0.05
    indirect_class = 1 if merge_metal_indirect else 2
    gt_pred = gt_pred.clone()
    gt_pred[metal_mask] = indirect_class
    return gt_pred


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Multi-task CGCNN v5')
    p.add_argument('data_dir', help='Path to data directory')
    p.add_argument('--epochs', default=300, type=int)
    p.add_argument('-b', '--batch-size', default=128, type=int)
    p.add_argument('--lr', default=0.002, type=float)
    p.add_argument('--weight-decay', default=5e-4, type=float)
    p.add_argument('--optim', default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    p.add_argument('--atom-fea-len', default=96, type=int)
    p.add_argument('--h-fea-len', default=256, type=int)
    p.add_argument('--n-conv', default=5, type=int)
    p.add_argument('--n-h', default=2, type=int)
    p.add_argument('--n-attn-heads', default=4, type=int)
    p.add_argument('--dropout', default=0.2, type=float)
    p.add_argument('-j', '--workers', default=0, type=int)
    p.add_argument('--print-freq', default=10, type=int)
    p.add_argument('--disable-cuda', action='store_true')
    p.add_argument('--patience', default=60, type=int)
    p.add_argument('--grad-clip', default=3.0, type=float)
    p.add_argument('--task-weights', nargs=3, type=float, default=[1.0, 8.0, 1.5])
    p.add_argument('--focal-gamma', default=2.0, type=float)
    p.add_argument('--label-smoothing', default=0.05, type=float)
    p.add_argument('--consistency-weight', default=0.2, type=float)
    p.add_argument('--augment-noise', default=0.03, type=float)
    p.add_argument('--augment-structure', action='store_true')
    p.add_argument('--augment-perturb', default=0.02, type=float)
    p.add_argument('--augment-scale', default=0.01, type=float)
    p.add_argument('--warmup-epochs', default=10, type=int)
    p.add_argument('--merge-metal-indirect', action='store_true', default=True)
    p.add_argument('--log-bg', action='store_true', default=True,
                   help='Use log(1+BG) transform')
    p.add_argument('--log-eh', action='store_true', default=True,
                   help='Use log(1+EH) transform')
    p.add_argument('--k-folds', default=3, type=int)
    p.add_argument('--n-seeds', default=2, type=int)
    p.add_argument('--train-ratio', default=0.8, type=float)
    p.add_argument('--val-ratio', default=0.1, type=float)
    p.add_argument('--test-ratio', default=0.1, type=float)
    p.add_argument('--checkpoint-dir', default='checkpoints/multitask_v5')
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
    all_gt_logp, all_gt_t = [], []
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
        all_gt_logp.append(p_gt.detach().cpu())
        all_gt_t.append(targets['gaptype'].squeeze(-1))
        all_eh_p.append(normalizer_eh.denorm(p_eh.detach().cpu()))
        all_eh_t.append(targets['ehull'])

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_logp = torch.cat(all_gt_logp)
    gt_t_tensor = torch.cat(all_gt_t)
    eh_p = torch.cat(all_eh_p).numpy().flatten()
    eh_t = torch.cat(all_eh_t).numpy().flatten()

    # Apply rule-based post-processing for GT
    bg_p_tensor = torch.from_numpy(bg_p)
    gt_p_ruled = apply_rules(bg_p_tensor, gt_logp,
                             merge_metal_indirect=args.merge_metal_indirect)
    gt_p = gt_p_ruled.numpy()
    gt_t = gt_t_tensor.numpy()

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    kw = dict(batch_size=args.batch_size, collate_fn=collate_pool_mt,
              num_workers=args.workers, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_idx), **kw)
    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **kw)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in train_idx])
    tr_eh = torch.tensor([float(dataset.id_prop_data[i][3]) for i in train_idx])
    n_bg = Normalizer(tr_bg, log_transform=args.log_bg)
    n_eh = Normalizer(tr_eh, log_transform=args.log_eh)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV5(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=dataset.n_gap_classes, dropout=args.dropout,
        n_attn_heads=args.n_attn_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model params: {n_params:,}')

    criterion_reg = nn.HuberLoss(delta=1.0)
    class_weights = dataset.get_class_weights().to(device)
    # Boost Direct class weight further
    class_weights[0] = class_weights[0] * 1.5
    print(f'  Class weights: {class_weights.tolist()}')
    criterion_cls = FocalLossWithSmoothing(
        alpha=class_weights, gamma=args.focal_gamma,
        smoothing=args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs,
                               eta_min=args.lr * 0.005)
    scheduler = SequentialLR(optimizer, [warmup, cosine], [args.warmup_epochs])

    fold_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}_seed_{seed}')
    os.makedirs(fold_dir, exist_ok=True)

    best_composite, best_gt_f1 = -float('inf'), -float('inf')
    best_bg_mae = float('inf')
    best_ep_comp, best_ep_f1 = 0, 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_m = run_epoch(train_loader, model, criterion_reg, criterion_cls,
                                n_bg, n_eh, args, device, dataset,
                                optimizer=optimizer, is_train=True)
        v_loss, v_m = run_epoch(val_loader, model, criterion_reg, criterion_cls,
                                n_bg, n_eh, args, device, dataset, is_train=False)
        scheduler.step()

        # Composite: balance all targets
        composite = -v_m['bg_mae'] + 3.0 * v_m['gt_f1'] - 2.0 * v_m['eh_mae']

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

        if v_m['bg_mae'] < best_bg_mae:
            best_bg_mae = v_m['bg_mae']
            torch.save(ckpt, os.path.join(fold_dir, 'best_bg_mae.pth.tar'))

        if epoch % args.print_freq == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'  Ep {epoch:03d} lr={lr_now:.5f} | '
                  f'TrL {t_loss:.3f} VL {v_loss:.3f} | '
                  f'BG {v_m["bg_mae"]:.4f} GT {v_m["gt_acc"]:.3f}/{v_m["gt_f1"]:.3f} '
                  f'EH {v_m["eh_mae"]:.4f} | comp={composite:.3f}')

        if args.patience > 0 and no_improve >= args.patience:
            print(f'  Early stop at epoch {epoch} (best comp ep {best_ep_comp})')
            break

    # Test all checkpoints
    results = {}
    for tag in ['composite', 'gt_f1', 'bg_mae']:
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
    all_bg_p, all_gt_logp, all_eh_p = [], [], []
    all_bg_t, all_gt_t, all_eh_t = [], [], []

    for inputs, targets, _ in test_loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        bg_preds, gt_logp_preds, eh_preds = [], [], []
        for m, nb, ne in zip(models, norm_bg_list, norm_eh_list):
            m.eval()
            with torch.no_grad():
                p_bg, p_gt, p_eh = m(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            bg_preds.append(nb.denorm(p_bg.cpu()))
            gt_logp_preds.append(p_gt.cpu())
            eh_preds.append(ne.denorm(p_eh.cpu()))

        # Average BG and EH
        avg_bg = torch.stack(bg_preds).mean(dim=0)
        avg_eh = torch.stack(eh_preds).mean(dim=0)
        # Average log-probs for GT then argmax
        avg_gt_logp = torch.stack(gt_logp_preds).mean(dim=0)
        # Apply rules
        gt_pred = apply_rules(avg_bg, avg_gt_logp,
                              merge_metal_indirect=args.merge_metal_indirect)

        all_bg_p.append(avg_bg)
        all_gt_logp.append(gt_pred)
        all_eh_p.append(avg_eh)
        all_bg_t.append(targets['bandgap'])
        all_gt_t.append(targets['gaptype'].squeeze(-1))
        all_eh_t.append(targets['ehull'])

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_p = torch.cat(all_gt_logp).numpy()
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
    print(f'Log transforms: BG={args.log_bg}, EH={args.log_eh}')

    dataset = CIFDataMultiTaskV4(
        args.data_dir,
        merge_metal_indirect=args.merge_metal_indirect,
        augment_perturb=args.augment_perturb if args.augment_structure else 0,
        augment_scale=args.augment_scale if args.augment_structure else 0,
    )
    print(f'Dataset: {len(dataset)} samples')
    print(f'Merge Metal->Indirect: {args.merge_metal_indirect} '
          f'(n_classes={dataset.n_gap_classes})')

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
        all_results = {'composite': [], 'gt_f1': [], 'bg_mae': []}
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
                seed = 42 + s * 137
                print(f'\n--- Seed {seed} ---')
                results, fold_dir, model, n_bg, n_eh, _, _ = train_fold(
                    dataset, train_idx, val_idx, test_idx, fi, seed, args, device)
                for tag in all_results:
                    if tag in results:
                        all_results[tag].append(results[tag])
                print(f'  Test composite: BG {results["composite"]["bg_mae"]:.4f} '
                      f'GT {results["composite"]["gt_acc"]:.4f}/{results["composite"]["gt_f1"]:.4f} '
                      f'EH {results["composite"]["eh_mae"]:.4f}')

                # Collect for ensemble
                ckpt = torch.load(
                    os.path.join(fold_dir, 'best_composite.pth.tar'),
                    weights_only=False)
                model.load_state_dict(ckpt['state_dict'])
                n_bg.load_state_dict(ckpt['normalizer_bg'])
                n_eh.load_state_dict(ckpt['normalizer_eh'])
                fold_models.append(model)
                fold_norms.append((n_bg, n_eh))

            if len(fold_models) > 1:
                ens = eval_ensemble(
                    fold_models, [x[0] for x in fold_norms], [x[1] for x in fold_norms],
                    test_loader, args, device)
                ensemble_results.append(ens)
                print(f'\n  Ensemble: BG {ens["bg_mae"]:.4f} '
                      f'GT {ens["gt_acc"]:.4f}/{ens["gt_f1"]:.4f} '
                      f'EH {ens["eh_mae"]:.4f}')

        # Summary
        print(f'\n{"="*60}')
        print(f'{args.k_folds}-Fold CV Summary (v5)')
        print(f'{"="*60}')
        for tag in ['composite', 'gt_f1', 'bg_mae']:
            if all_results[tag]:
                print(f'\n  Checkpoint: best_{tag}')
                for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']:
                    vals = [m[key] for m in all_results[tag]]
                    unit = ' eV' if key == 'bg_mae' else (' eV/atom' if key == 'eh_mae' else '')
                    print(f'    {key:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{unit}')
        if ensemble_results:
            print(f'\n  Ensemble ({args.n_seeds}-seed avg):')
            for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']:
                vals = [m[key] for m in ensemble_results]
                unit = ' eV' if key == 'bg_mae' else (' eV/atom' if key == 'eh_mae' else '')
                print(f'    {key:8s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{unit}')

        # Check vs targets
        print(f'\n{"="*60}')
        print('Target check (ensemble if available, else best_composite):')
        if ensemble_results:
            ref = {k: np.mean([m[k] for m in ensemble_results]) for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']}
        else:
            ref = {k: np.mean([m[k] for m in all_results['composite']]) for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_mae']}
        targets_check = [
            ('BG MAE < 0.5', ref['bg_mae'] < 0.5, f'{ref["bg_mae"]:.4f}'),
            ('GT Acc > 0.8', ref['gt_acc'] > 0.8, f'{ref["gt_acc"]:.4f}'),
            ('GT F1  > 0.85', ref['gt_f1'] > 0.85, f'{ref["gt_f1"]:.4f}'),
            ('EH MAE < 0.1', ref['eh_mae'] < 0.1, f'{ref["eh_mae"]:.4f}'),
        ]
        for desc, met, val in targets_check:
            status = 'PASS' if met else 'FAIL'
            print(f'  [{status}] {desc}: {val}')
        print(f'{"="*60}')
    else:
        train_idx, val_idx, test_idx = stratified_split_v4(
            dataset.id_prop_data, args.train_ratio, args.val_ratio, args.test_ratio,
            merge_metal_indirect=args.merge_metal_indirect)
        train_fold(dataset, train_idx, val_idx, test_idx, 0, 42, args, device)

    print(f'\nTotal time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
