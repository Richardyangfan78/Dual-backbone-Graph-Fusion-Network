"""
Multi-task CGCNN v7 — key changes over v6:

1. Wider/deeper model: 128 atom_fea, 256 h_fea, 6 conv, 8-head attention
2. EH task: binary classification (stable: EH<0.01 eV/atom vs unstable)
3. Stronger GT focal loss: gamma=3.0
4. CosineAnnealingWarmRestarts (T0=100, T_mult=2) for better exploration
5. lr=0.001, dropout=0.3, patience=100, epochs=600, n_seeds=3
6. Updated composite: -2*bg_mae + 3*gt_f1 + 1*eh_acc
7. Target: BG MAE<0.5, GT Acc>0.8, EH Acc>0.8 (stable/unstable classification)
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))

from data_mt_v4 import (
    CIFDataMultiTaskV4,
    stratified_kfold_v4,
    _map_gaptype,
)
from model_mt_v7 import CrystalGraphConvNetMTV7

# EH stability threshold: EH < STABLE_THRESH → class 0 (stable)
STABLE_THRESH = 0.01  # eV/atom


def collate_pool_mt_v7(dataset_list):
    """Collate for v7: targets include bandgap, gaptype, eh_label, ehull_raw."""
    import numpy as np
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    batch_bandgap, batch_gaptype, batch_eh_label, batch_ehull_raw = [], [], [], []
    batch_cif_ids = []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), targets, cif_id in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_bandgap.append(targets["bandgap"])
        batch_gaptype.append(targets["gaptype"])
        batch_eh_label.append(targets["eh_label"])
        batch_ehull_raw.append(targets["ehull_raw"])
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    target_dict = {
        "bandgap":   torch.stack(batch_bandgap, dim=0),
        "gaptype":   torch.stack(batch_gaptype, dim=0),
        "eh_label":  torch.stack(batch_eh_label, dim=0),
        "ehull_raw": torch.stack(batch_ehull_raw, dim=0),
    }
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), target_dict, batch_cif_ids



def eh_to_label(eh_val):
    """Convert raw EH value to binary stability label."""
    return 0 if float(eh_val) < STABLE_THRESH else 1


# ---------------------------------------------------------------------------
# Cached graph dataset v7 — adds EH binary label
# ---------------------------------------------------------------------------
class CachedGraphDatasetV7(Dataset):
    """Load pre-computed graph tensors; EH is binary stability label."""
    def __init__(self, cache_dir, id_prop_data, merge_metal_indirect=False,
                 augment_noise=0.0):
        self.cache_dir = cache_dir
        self.id_prop_data = id_prop_data
        self.merge_metal_indirect = merge_metal_indirect
        self.augment_noise = augment_noise
        self.augment = False

        sample_id = id_prop_data[0][0]
        assert os.path.isfile(os.path.join(cache_dir, f'{sample_id}.pt')), \
            f'Cache not found: {cache_dir}/{sample_id}.pt'

    def __len__(self):
        return len(self.id_prop_data)

    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx][0]
        data = torch.load(os.path.join(self.cache_dir, f'{cif_id}.pt'),
                          weights_only=True)

        atom_fea = data['atom_fea']
        nbr_fea = data['nbr_fea']
        nbr_fea_idx = data['nbr_fea_idx']

        if self.augment and self.augment_noise > 0:
            atom_fea = atom_fea + torch.randn_like(atom_fea) * self.augment_noise

        gaptype = _map_gaptype(self.id_prop_data[idx][2],
                               self.merge_metal_indirect)
        eh_label = eh_to_label(self.id_prop_data[idx][3])

        targets = {
            'bandgap': data['bandgap'],
            'gaptype': torch.LongTensor([gaptype]),
            'eh_label': torch.LongTensor([eh_label]),
            'ehull_raw': data['ehull'],   # kept for reporting only
        }
        return (atom_fea, nbr_fea, nbr_fea_idx), targets, cif_id

    def get_gt_class_weights(self):
        counts = {}
        for row in self.id_prop_data:
            label = _map_gaptype(row[2], self.merge_metal_indirect)
            counts[label] = counts.get(label, 0) + 1
        n_classes = max(counts.keys()) + 1
        total = sum(counts.values())
        weights = [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)]
        return torch.FloatTensor(weights)

    def get_eh_class_weights(self):
        counts = {0: 0, 1: 0}
        for row in self.id_prop_data:
            counts[eh_to_label(row[3])] += 1
        total = sum(counts.values())
        w0 = total / (2 * counts[0])
        w1 = total / (2 * counts[1])
        return torch.FloatTensor([w0, w1])

    def get_sample_weights_for_gt(self, indices):
        """Per-sample weights for WeightedRandomSampler to balance GT classes."""
        cw = self.get_gt_class_weights()
        w = []
        for i in indices:
            label = _map_gaptype(self.id_prop_data[i][2], self.merge_metal_indirect)
            w.append(cw[label].item())
        return w

    @property
    def n_gap_classes(self):
        return 2 if self.merge_metal_indirect else 3


# ---------------------------------------------------------------------------
# Focal Loss with Label Smoothing (same as v6)
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


# ---------------------------------------------------------------------------
# Dynamic multi-task loss (Kendall et al. 2018) — adapted for 3 tasks
# ---------------------------------------------------------------------------
class DynamicMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_bg = nn.Parameter(torch.tensor(0.0))
        self.log_var_gt = nn.Parameter(torch.tensor(0.0))
        self.log_var_eh = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_bg, loss_gt, loss_eh):
        w_bg = torch.exp(-self.log_var_bg)
        w_gt = torch.exp(-self.log_var_gt)
        w_eh = torch.exp(-self.log_var_eh)
        total = (w_bg * loss_bg + 0.5 * self.log_var_bg +
                 w_gt * loss_gt + 0.5 * self.log_var_gt +
                 w_eh * loss_eh + 0.5 * self.log_var_eh)
        return total, (w_bg.item(), w_gt.item(), w_eh.item())

    def get_weights(self):
        return (torch.exp(-self.log_var_bg).item(),
                torch.exp(-self.log_var_gt).item(),
                torch.exp(-self.log_var_eh).item())


# ---------------------------------------------------------------------------
# Normalizer (BG only — EH is now classification)
# ---------------------------------------------------------------------------
class Normalizer:
    def __init__(self, tensor=None, log_transform=False, robust=False):
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
        self.robust = d.get('robust', False)


# ---------------------------------------------------------------------------
# Weighted Huber loss for BG
# ---------------------------------------------------------------------------
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.8, reweight_scale=0.5):
        super().__init__()
        self.delta = delta
        self.reweight_scale = reweight_scale

    def forward(self, pred, target, raw_bg=None):
        diff = pred - target
        abs_diff = diff.abs()
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        if raw_bg is not None and self.reweight_scale > 0:
            weights = 1.0 + self.reweight_scale * (raw_bg.abs() / 5.0).clamp(0, 1)
            loss = loss * weights.view_as(loss)
        return loss.mean()


# ---------------------------------------------------------------------------
# Rule-based post-processing for GT
# ---------------------------------------------------------------------------
def apply_rules(bg_pred, gt_log_probs, merge_metal_indirect=True):
    gt_pred = (gt_log_probs.argmax(dim=1) if gt_log_probs.dim() > 1
               else gt_log_probs)
    bg = bg_pred.squeeze(-1)
    metal_mask = bg < 0.05
    indirect_class = 1 if merge_metal_indirect else 2
    gt_pred = gt_pred.clone()
    gt_pred[metal_mask] = indirect_class
    return gt_pred


# ---------------------------------------------------------------------------
# Physical consistency loss (BG ↔ GT)
# ---------------------------------------------------------------------------
def physical_consistency_loss(bg_denorm, gt_log_probs, indirect_class=1,
                              threshold=0.1):
    probs = torch.exp(gt_log_probs)
    indirect_prob = probs[:, indirect_class]
    bg = bg_denorm.squeeze(-1)
    should_indirect = torch.sigmoid(-10.0 * (bg - threshold))
    loss = (should_indirect * (1 - indirect_prob) +
            (1 - should_indirect) * indirect_prob)
    return loss.mean()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Multi-task CGCNN v7')
    p.add_argument('data_dir', help='Path to data directory')
    p.add_argument('--cache-dir', default='',
                   help='Cached graph dir. If empty, falls back to CIF loading.')
    p.add_argument('--epochs', default=600, type=int)
    p.add_argument('-b', '--batch-size', default=64, type=int)
    p.add_argument('--lr', default=0.001, type=float)
    p.add_argument('--weight-decay', default=1e-3, type=float)
    p.add_argument('--optim', default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
    p.add_argument('--atom-fea-len', default=128, type=int)
    p.add_argument('--h-fea-len', default=256, type=int)
    p.add_argument('--n-conv', default=6, type=int)
    p.add_argument('--n-h', default=1, type=int)
    p.add_argument('--n-attn-heads', default=8, type=int)
    p.add_argument('--dropout', default=0.3, type=float)
    p.add_argument('-j', '--workers', default=0, type=int)
    p.add_argument('--print-freq', default=10, type=int)
    p.add_argument('--disable-cuda', action='store_true')
    p.add_argument('--patience', default=100, type=int)
    p.add_argument('--grad-clip', default=2.0, type=float)
    p.add_argument('--task-weights', nargs=3, type=float, default=[3.0, 5.0, 2.0],
                   help='Static weights [BG, GT, EH]')
    p.add_argument('--dynamic-weights', action='store_true', default=True)
    p.add_argument('--no-dynamic-weights', dest='dynamic_weights',
                   action='store_false')
    p.add_argument('--focal-gamma-gt', default=3.0, type=float,
                   help='Focal gamma for gap type')
    p.add_argument('--focal-gamma-eh', default=2.0, type=float,
                   help='Focal gamma for EH stability')
    p.add_argument('--label-smoothing', default=0.05, type=float)
    p.add_argument('--consistency-weight', default=0.1, type=float)
    p.add_argument('--augment-noise', default=0.05, type=float)
    p.add_argument('--warmup-epochs', default=15, type=int)
    p.add_argument('--cosine-T0', default=100, type=int,
                   help='CosineAnnealingWarmRestarts T0')
    p.add_argument('--cosine-Tmult', default=2, type=int,
                   help='CosineAnnealingWarmRestarts T_mult')
    p.add_argument('--merge-metal-indirect', action='store_true', default=True)
    p.add_argument('--log-bg', action='store_true', default=False)
    p.add_argument('--robust-norm', action='store_true', default=True)
    p.add_argument('--no-robust-norm', dest='robust_norm', action='store_false')
    p.add_argument('--huber-delta', default=0.8, type=float)
    p.add_argument('--bg-reweight', default=0.5, type=float)
    p.add_argument('--eh-stable-thresh', default=STABLE_THRESH, type=float,
                   help='EH threshold for stable (class 0) classification')
    p.add_argument('--use-weighted-sampler', action='store_true', default=True,
                   help='Use WeightedRandomSampler to balance GT classes')
    p.add_argument('--no-weighted-sampler', dest='use_weighted_sampler',
                   action='store_false')
    p.add_argument('--k-folds', default=3, type=int)
    p.add_argument('--n-seeds', default=3, type=int)
    p.add_argument('--checkpoint-dir', default='checkpoints/multitask_v7')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------
def run_epoch(loader, model, criterion_bg, criterion_gt, criterion_eh,
              normalizer_bg, args, device, dataset,
              optimizer=None, mt_loss_module=None, is_train=True):
    model.train() if is_train else model.eval()
    if hasattr(dataset, 'augment'):
        dataset.augment = is_train

    tot_loss, n = 0.0, 0
    all_bg_p, all_bg_t = [], []
    all_gt_logp, all_gt_t = [], []
    all_eh_logp, all_eh_t = [], []
    indirect_class = 1 if args.merge_metal_indirect else 2

    for inputs, targets, _ in loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        t_bg = normalizer_bg.norm(targets['bandgap']).to(device)
        t_gt = targets['gaptype'].squeeze(-1).to(device)
        t_eh = targets['eh_label'].squeeze(-1).to(device)
        raw_bg = targets['bandgap'].to(device)

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            p_bg, p_gt, p_eh = model(atom_fea, nbr_fea, nbr_fea_idx,
                                     crystal_atom_idx)
            loss_bg = criterion_bg(p_bg, t_bg, raw_bg)
            loss_gt = criterion_gt(p_gt, t_gt)
            loss_eh = criterion_eh(p_eh, t_eh)

            if mt_loss_module is not None and args.dynamic_weights:
                loss, _ = mt_loss_module(loss_bg, loss_gt, loss_eh)
            else:
                w_bg, w_gt, w_eh = args.task_weights
                loss = w_bg * loss_bg + w_gt * loss_gt + w_eh * loss_eh

            if args.consistency_weight > 0:
                bg_denorm = normalizer_bg.denorm(p_bg.detach())
                cons = physical_consistency_loss(
                    bg_denorm, p_gt, indirect_class=indirect_class)
                loss = loss + args.consistency_weight * cons

        if is_train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                params = list(model.parameters())
                if mt_loss_module is not None:
                    params += list(mt_loss_module.parameters())
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

        tot_loss += loss.item()
        n += 1
        all_bg_p.append(normalizer_bg.denorm(p_bg.detach().cpu()))
        all_bg_t.append(targets['bandgap'])
        all_gt_logp.append(p_gt.detach().cpu())
        all_gt_t.append(targets['gaptype'].squeeze(-1))
        all_eh_logp.append(p_eh.detach().cpu())
        all_eh_t.append(targets['eh_label'].squeeze(-1))

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_logp = torch.cat(all_gt_logp)
    gt_t_tensor = torch.cat(all_gt_t)
    eh_logp = torch.cat(all_eh_logp)
    eh_t_arr = torch.cat(all_eh_t).numpy()

    bg_p_tensor = torch.from_numpy(bg_p)
    gt_p_ruled = apply_rules(bg_p_tensor, gt_logp,
                             merge_metal_indirect=args.merge_metal_indirect)
    gt_p = gt_p_ruled.numpy()
    gt_t = gt_t_tensor.numpy()
    eh_p = eh_logp.argmax(dim=1).numpy()

    avg_type = 'binary' if args.merge_metal_indirect else 'macro'
    return tot_loss / max(n, 1), {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_p)),
        'gt_f1':  float(f1_score(gt_t, gt_p, average=avg_type, zero_division=0)),
        'eh_acc': float(accuracy_score(eh_t_arr, eh_p)),
        'eh_f1':  float(f1_score(eh_t_arr, eh_p, average='binary', zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Load a completed fold (skip training if all checkpoints exist & valid)
# ---------------------------------------------------------------------------
def _fold_is_complete(fold_dir, min_size_mb=8):
    """Return True if fold_dir has all 3 valid checkpoint files."""
    required = ['best_composite.pth.tar', 'best_bg_mae.pth.tar', 'best_gt_f1.pth.tar']
    for fname in required:
        p = os.path.join(fold_dir, fname)
        if not os.path.isfile(p) or os.path.getsize(p) < min_size_mb * 1024 * 1024:
            return False
    return True


def load_completed_fold(dataset, test_idx, fold_idx, seed, args, device):
    """Load results from an already-completed fold without retraining."""
    fold_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}_seed_{seed}')
    print(f'  [SKIP] fold_{fold_idx}_seed_{seed} already complete — loading from checkpoint')

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in test_idx])
    n_bg = Normalizer(tr_bg, log_transform=args.log_bg, robust=args.robust_norm)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV7(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=dataset.n_gap_classes, n_eh_classes=2,
        dropout=args.dropout, n_attn_heads=args.n_attn_heads,
    ).to(device)

    kw = dict(batch_size=args.batch_size, collate_fn=collate_pool_mt_v7,
              num_workers=args.workers, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    gt_cw = dataset.get_gt_class_weights().to(device)
    gt_cw[0] = gt_cw[0] * 1.5
    criterion_gt = FocalLossWithSmoothing(alpha=gt_cw, gamma=args.focal_gamma_gt,
                                          smoothing=args.label_smoothing)
    eh_cw = dataset.get_eh_class_weights().to(device)
    criterion_eh = FocalLossWithSmoothing(alpha=eh_cw, gamma=args.focal_gamma_eh,
                                          smoothing=args.label_smoothing)
    criterion_bg = WeightedHuberLoss(delta=args.huber_delta,
                                     reweight_scale=args.bg_reweight)

    results = {}
    for tag in ['composite', 'gt_f1', 'bg_mae']:
        ckpt_path = os.path.join(fold_dir, f'best_{tag}.pth.tar')
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        n_bg.load_state_dict(ckpt['normalizer_bg'])
        _, test_m = run_epoch(
            test_loader, model, criterion_bg, criterion_gt, criterion_eh,
            n_bg, args, device, dataset, is_train=False)
        results[tag] = test_m

    return results, fold_dir, model, n_bg


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

    kw = dict(batch_size=args.batch_size, collate_fn=collate_pool_mt_v7,
              num_workers=args.workers, pin_memory=device.type == 'cuda')

    if args.use_weighted_sampler:
        sample_w = dataset.get_sample_weights_for_gt(train_idx)
        sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)
        train_loader = DataLoader(dataset, sampler=sampler, **kw)
    else:
        train_loader = DataLoader(dataset,
                                  sampler=SubsetRandomSampler(train_idx), **kw)

    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **kw)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in train_idx])
    n_bg = Normalizer(tr_bg, log_transform=args.log_bg, robust=args.robust_norm)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV7(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=dataset.n_gap_classes, n_eh_classes=2,
        dropout=args.dropout, n_attn_heads=args.n_attn_heads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model params: {n_params:,}')

    criterion_bg = WeightedHuberLoss(delta=args.huber_delta,
                                     reweight_scale=args.bg_reweight)

    gt_cw = dataset.get_gt_class_weights().to(device)
    gt_cw[0] = gt_cw[0] * 1.5  # extra boost to minority direct-gap class
    print(f'  GT class weights: {gt_cw.tolist()}')
    criterion_gt = FocalLossWithSmoothing(
        alpha=gt_cw, gamma=args.focal_gamma_gt,
        smoothing=args.label_smoothing)

    eh_cw = dataset.get_eh_class_weights().to(device)
    print(f'  EH class weights: {eh_cw.tolist()}')
    criterion_eh = FocalLossWithSmoothing(
        alpha=eh_cw, gamma=args.focal_gamma_eh,
        smoothing=args.label_smoothing)

    mt_loss = None
    opt_params = list(model.parameters())
    if args.dynamic_weights:
        mt_loss = DynamicMultiTaskLoss().to(device)
        opt_params += list(mt_loss.parameters())
        print('  Dynamic task weighting (Kendall)')
    else:
        print(f'  Static task weights: {args.task_weights}')

    optimizer = optim.AdamW(opt_params, args.lr,
                            weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.01,
                      total_iters=args.warmup_epochs)
    cosine = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult,
        eta_min=args.lr * 0.002)
    scheduler = SequentialLR(optimizer, [warmup, cosine],
                             [args.warmup_epochs])

    fold_dir = os.path.join(args.checkpoint_dir,
                            f'fold_{fold_idx}_seed_{seed}')
    os.makedirs(fold_dir, exist_ok=True)

    best_composite = -float('inf')
    best_gt_f1 = -float('inf')
    best_bg_mae = float('inf')
    best_ep_comp = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t_loss, t_m = run_epoch(
            train_loader, model, criterion_bg, criterion_gt, criterion_eh,
            n_bg, args, device, dataset,
            optimizer=optimizer, mt_loss_module=mt_loss, is_train=True)
        v_loss, v_m = run_epoch(
            val_loader, model, criterion_bg, criterion_gt, criterion_eh,
            n_bg, args, device, dataset, is_train=False)
        scheduler.step()

        # v7 composite: BG (-2), GT F1 (+3), EH accuracy (+1)
        composite = (-2.0 * v_m['bg_mae'] + 3.0 * v_m['gt_f1']
                     + 1.0 * v_m['eh_acc'])

        ckpt = {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer_bg': n_bg.state_dict(),
                'args': vars(args)}
        if mt_loss is not None:
            ckpt['mt_loss'] = mt_loss.state_dict()

        if composite > best_composite:
            best_composite = composite
            best_ep_comp = epoch
            no_improve = 0
            torch.save(ckpt, os.path.join(fold_dir, 'best_composite.pth.tar'))
        else:
            no_improve += 1

        if v_m['gt_f1'] > best_gt_f1:
            best_gt_f1 = v_m['gt_f1']
            torch.save(ckpt, os.path.join(fold_dir, 'best_gt_f1.pth.tar'))

        if v_m['bg_mae'] < best_bg_mae:
            best_bg_mae = v_m['bg_mae']
            torch.save(ckpt, os.path.join(fold_dir, 'best_bg_mae.pth.tar'))

        if epoch % args.print_freq == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            dw_str = ''
            if mt_loss is not None:
                wb, wg, we = mt_loss.get_weights()
                dw_str = f' dw=[{wb:.2f},{wg:.2f},{we:.2f}]'
            print(f'  Ep {epoch:03d} lr={lr_now:.5f}{dw_str} | '
                  f'TrL {t_loss:.3f} VL {v_loss:.3f} | '
                  f'BG {v_m["bg_mae"]:.4f} GT {v_m["gt_acc"]:.3f}/'
                  f'{v_m["gt_f1"]:.3f} '
                  f'EH_acc {v_m["eh_acc"]:.3f} | comp={composite:.3f}')

        if args.patience > 0 and no_improve >= args.patience:
            print(f'  Early stop at epoch {epoch} '
                  f'(best comp ep {best_ep_comp})')
            break

    # Test all checkpoints
    results = {}
    for tag in ['composite', 'gt_f1', 'bg_mae']:
        ckpt_path = os.path.join(fold_dir, f'best_{tag}.pth.tar')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            n_bg.load_state_dict(ckpt['normalizer_bg'])
            _, test_m = run_epoch(
                test_loader, model, criterion_bg, criterion_gt, criterion_eh,
                n_bg, args, device, dataset, is_train=False)
            results[tag] = test_m
    return results, fold_dir, model, n_bg


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------
def eval_ensemble(models, norm_bg_list, test_loader, args, device):
    all_bg_p, all_gt_logp, all_eh_logp = [], [], []
    all_bg_t, all_gt_t, all_eh_t = [], [], []

    for inputs, targets, _ in test_loader:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        bg_preds, gt_logp_preds, eh_logp_preds = [], [], []
        for m, nb in zip(models, norm_bg_list):
            m.eval()
            with torch.no_grad():
                p_bg, p_gt, p_eh = m(atom_fea, nbr_fea, nbr_fea_idx,
                                     crystal_atom_idx)
            bg_preds.append(nb.denorm(p_bg.cpu()))
            gt_logp_preds.append(p_gt.cpu())
            eh_logp_preds.append(p_eh.cpu())

        avg_bg = torch.stack(bg_preds).mean(dim=0)
        avg_gt_logp = torch.stack(gt_logp_preds).mean(dim=0)
        avg_eh_logp = torch.stack(eh_logp_preds).mean(dim=0)
        gt_pred = apply_rules(avg_bg, avg_gt_logp,
                              merge_metal_indirect=args.merge_metal_indirect)

        all_bg_p.append(avg_bg)
        all_gt_logp.append(gt_pred)
        all_eh_logp.append(avg_eh_logp.argmax(dim=1))
        all_bg_t.append(targets['bandgap'])
        all_gt_t.append(targets['gaptype'].squeeze(-1))
        all_eh_t.append(targets['eh_label'].squeeze(-1))

    bg_p = torch.cat(all_bg_p).numpy().flatten()
    bg_t = torch.cat(all_bg_t).numpy().flatten()
    gt_p = torch.cat(all_gt_logp).numpy()
    gt_t = torch.cat(all_gt_t).numpy()
    eh_p = torch.cat(all_eh_logp).numpy()
    eh_t = torch.cat(all_eh_t).numpy()

    avg_type = 'binary' if args.merge_metal_indirect else 'macro'
    return {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_p)),
        'gt_f1':  float(f1_score(gt_t, gt_p, average=avg_type, zero_division=0)),
        'eh_acc': float(accuracy_score(eh_t, eh_p)),
        'eh_f1':  float(f1_score(eh_t, eh_p, average='binary', zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')
    print(f'EH stable threshold: {args.eh_stable_thresh} eV/atom')
    print(f'Dynamic weights: {args.dynamic_weights}')
    print(f'Weighted sampler: {args.use_weighted_sampler}')

    cache_dir = args.cache_dir
    if cache_dir and os.path.isdir(cache_dir):
        print(f'Using cached graphs from: {cache_dir}')
        id_prop_file = os.path.join(args.data_dir, 'id_prop.csv')
        with open(id_prop_file) as f:
            id_prop_data = list(csv.reader(f))
        random.seed(123)
        random.shuffle(id_prop_data)
        dataset = CachedGraphDatasetV7(
            cache_dir, id_prop_data,
            merge_metal_indirect=args.merge_metal_indirect,
            augment_noise=args.augment_noise)
    else:
        raise RuntimeError('v7 requires cached graphs. Run preprocess_graphs.py first.')

    print(f'Dataset: {len(dataset)} samples')

    # Class distribution summary
    gt_counts = {}
    eh_counts = {0: 0, 1: 0}
    for row in dataset.id_prop_data:
        l = _map_gaptype(row[2], args.merge_metal_indirect)
        gt_counts[l] = gt_counts.get(l, 0) + 1
        eh_counts[eh_to_label(row[3])] += 1
    print(f'GT class dist: {dict(sorted(gt_counts.items()))}')
    print(f'EH class dist (stable=0,unstable=1): {eh_counts}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()

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
            batch_size=args.batch_size, collate_fn=collate_pool_mt_v7,
            num_workers=args.workers)

        seeds = [42 + s * 137 for s in range(args.n_seeds)]
        for seed in seeds:
            print('\n--- Seed', seed, '---')
            fold_dir_check = os.path.join(args.checkpoint_dir, f'fold_{fi}_seed_{seed}')
            if _fold_is_complete(fold_dir_check):
                results, fold_dir, model, n_bg = load_completed_fold(
                    dataset, test_idx, fi, seed, args, device)
            else:
                if os.path.isdir(fold_dir_check):
                    import shutil; shutil.rmtree(fold_dir_check)
                    print(f'  [WARN] Removed incomplete: {fold_dir_check}')
                results, fold_dir, model, n_bg = train_fold(
                    dataset, train_idx, val_idx, test_idx, fi, seed,
                    args, device)
            for tag in all_results:
                if tag in results:
                    all_results[tag].append(results[tag])
            print(f'  Test composite: '
                  f'BG {results["composite"]["bg_mae"]:.4f} '
                  f'GT {results["composite"]["gt_acc"]:.4f}/'
                  f'{results["composite"]["gt_f1"]:.4f} '
                  f'EH_acc {results["composite"]["eh_acc"]:.4f}')

            ckpt = torch.load(
                os.path.join(fold_dir, 'best_composite.pth.tar'),
                weights_only=False)
            model.load_state_dict(ckpt['state_dict'])
            n_bg.load_state_dict(ckpt['normalizer_bg'])
            fold_models.append(model)
            fold_norms.append(n_bg)

        if len(fold_models) > 1:
            ens = eval_ensemble(fold_models, fold_norms, test_loader, args, device)
            ensemble_results.append(ens)
            print(f'\n  Ensemble: BG {ens["bg_mae"]:.4f} '
                  f'GT {ens["gt_acc"]:.4f}/{ens["gt_f1"]:.4f} '
                  f'EH_acc {ens["eh_acc"]:.4f}/{ens["eh_f1"]:.4f}')

    # Summary
    print(f'\n{"="*60}')
    print(f'{args.k_folds}-Fold CV Summary (v7)')
    print(f'{"="*60}')
    for tag in ['composite', 'gt_f1', 'bg_mae']:
        if all_results[tag]:
            print(f'\n  Checkpoint: best_{tag}')
            for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']:
                vals = [m[key] for m in all_results[tag]]
                unit = ' eV' if key == 'bg_mae' else ''
                print(f'    {key:8s}: {np.mean(vals):.4f} '
                      f'+/- {np.std(vals):.4f}{unit}')

    if ensemble_results:
        print(f'\n  Ensemble ({args.n_seeds}-seed avg):')
        for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']:
            vals = [m[key] for m in ensemble_results]
            unit = ' eV' if key == 'bg_mae' else ''
            print(f'    {key:8s}: {np.mean(vals):.4f} '
                  f'+/- {np.std(vals):.4f}{unit}')

    # Target check
    print(f'\n{"="*60}')
    print('Target check (ensemble if available, else best_composite):')
    if ensemble_results:
        ref = {k: np.mean([m[k] for m in ensemble_results])
               for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']}
    else:
        ref = {k: np.mean([m[k] for m in all_results['composite']])
               for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']}
    targets_check = [
        ('BG MAE < 0.5 eV',    ref['bg_mae'] < 0.5,  f'{ref["bg_mae"]:.4f}'),
        ('GT Acc > 0.8',        ref['gt_acc'] > 0.8,  f'{ref["gt_acc"]:.4f}'),
        ('GT F1  > 0.85',       ref['gt_f1']  > 0.85, f'{ref["gt_f1"]:.4f}'),
        ('EH Acc > 0.80',       ref['eh_acc'] > 0.80, f'{ref["eh_acc"]:.4f}'),
        ('EH F1  > 0.80',       ref['eh_f1']  > 0.80, f'{ref["eh_f1"]:.4f}'),
    ]
    for desc, met, val in targets_check:
        status = 'PASS' if met else 'FAIL'
        print(f'  [{status}] {desc}: {val}')
    print(f'{"="*60}')
    print(f'\nTotal time: {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
