"""
Multi-task CGCNN v10 — improvements over v9:

1. 5-fold CV with single seed (parallelize via SLURM array)
2. GradNorm MTL weighting (replaces Kendall uncertainty)
3. CosineClassifier for GT head (reduces class imbalance bias)
4. Increased capacity: 160/320/8conv
5. Structure augmentation enabled by default
6. Consistency loss logged
7. --fold N flag to run a specific fold (for array jobs)

Fixes over CodeX draft:
- Fixed indentation bug in main() (fatal SyntaxError)
- GradNorm: skip expensive autograd.grad on non-update steps
- Parameterized gt_cw minority boost (--gt-minority-boost)
- CosineClassifier temp default 0.05 -> 0.1
- Cached shared_params list per fold instead of per step
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import Counter

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
    stratified_kfold_v4,
    stratified_split_from_indices,
    _map_gaptype,
)
from model_mt_v10 import CrystalGraphConvNetMTV10

# EH stability threshold: EH < STABLE_THRESH -> class 0 (stable)
STABLE_THRESH = 0.1  # eV/atom


def load_aligned_split_indices(id_prop_data, data_dir, fold):
    split_dir = os.path.join(data_dir, "splits_mace_alignn")
    paths = {name: os.path.join(split_dir, f"fold{fold}_{name}.txt")
             for name in ("train", "val", "test")}
    if not all(os.path.exists(path) for path in paths.values()):
        return None
    id_to_idx = {row[0]: i for i, row in enumerate(id_prop_data)}
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

# ---------------------------------------------------------------------------
# GradNorm — task-specific gradient normalization for MTL
# ---------------------------------------------------------------------------
class GradNorm(nn.Module):
    def __init__(self, n_tasks, alpha=1.5):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(n_tasks))
        self.initial_losses = None

    def get_task_weights(self):
        """Return detached softplus weights for use in non-update steps."""
        return F.softplus(self.weights).detach()

    def forward(self, losses, shared_params):
        """Full GradNorm forward — only call on update steps."""
        w = F.softplus(self.weights)
        weighted_loss = (w * losses).sum()

        if self.initial_losses is None:
            self.initial_losses = losses.detach()
            return weighted_loss, w, torch.tensor(0.0, device=losses.device)

        grad_norms = []
        for i in range(self.n_tasks):
            grad = torch.autograd.grad(
                w[i] * losses[i], list(shared_params),
                retain_graph=True, create_graph=True, allow_unused=True
            )
            valid = [g for g in grad if g is not None]
            if valid:
                grad_norms.append(torch.norm(torch.stack([g.norm() for g in valid])))
            else:
                grad_norms.append(torch.tensor(1.0, device=losses.device))

        grad_norms = torch.stack(grad_norms)
        mean_norm = grad_norms.mean().detach()
        loss_ratios = losses.detach() / (self.initial_losses + 1e-8)
        mean_ratio = loss_ratios.mean()
        target_norms = mean_norm * (loss_ratios / (mean_ratio + 1e-8)) ** self.alpha
        grad_norm_loss = F.l1_loss(grad_norms, target_norms.detach())
        return weighted_loss, w, grad_norm_loss


# ---------------------------------------------------------------------------
# Collate, helpers — same as v9
# ---------------------------------------------------------------------------
def collate_pool_mt_v7(dataset_list):
    """Collate for v7/v9/v10: targets include bandgap, gaptype, eh_label, ehull_raw."""
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
    return 0 if float(eh_val) < STABLE_THRESH else 1


class WeightedSubsetSampler:
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


def _build_graph_from_structure(crystal, ari, gdf, max_num_nbr=12, radius=8, cif_id='?'):
    import warnings
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
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr)))
            nbr_fea.append(
                list(map(lambda x: x[1], nbr)) + [radius + 1.] * (max_num_nbr - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

    nbr_fea = torch.Tensor(gdf.expand(np.array(nbr_fea)))
    nbr_fea_idx = torch.LongTensor(np.array(nbr_fea_idx))
    return atom_fea, nbr_fea, nbr_fea_idx


class CachedGraphDatasetV7(Dataset):
    """Load pre-computed graph tensors; optional structure-aware augmentation."""
    def __init__(self, cache_dir, id_prop_data, merge_metal_indirect=False,
                 augment_noise=0.0, data_dir=None, augment_perturb=0.02,
                 augment_scale=0.01, augment_structure_prob=0.3,
                 radius=8, max_num_nbr=12):
        self.cache_dir = cache_dir
        self.id_prop_data = id_prop_data
        self.merge_metal_indirect = merge_metal_indirect
        self.augment_noise = augment_noise
        self.augment = False
        self.data_dir = data_dir
        self.augment_perturb = augment_perturb
        self.augment_scale = augment_scale
        self.augment_structure_prob = augment_structure_prob
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self._ari = None
        self._gdf = None
        if data_dir:
            import torch as _torch
            _sample_pt = os.path.join(cache_dir, id_prop_data[0][0] + '.pt')
            _cache_dim = _torch.load(_sample_pt, weights_only=False)['atom_fea'].shape[1]
            atom_init_92 = os.path.join(data_dir, 'atom_init_original_92dim.json')
            atom_init_def = os.path.join(data_dir, 'atom_init.json')
            if _cache_dim == 92 and os.path.exists(atom_init_92):
                atom_init = atom_init_92
            elif os.path.exists(atom_init_def):
                atom_init = atom_init_def
            else:
                atom_init = atom_init_92
            if os.path.exists(atom_init):
                from cgcnn.data import AtomCustomJSONInitializer, GaussianDistance
                self._ari = AtomCustomJSONInitializer(atom_init)
                self._gdf = GaussianDistance(dmin=0, dmax=radius, step=0.2)
                print('Structure aug atom_init: ' + os.path.basename(atom_init)
                      + ' (' + str(_cache_dim) + '-dim, matched cache)')

        sample_id = id_prop_data[0][0]
        assert os.path.isfile(os.path.join(cache_dir, f'{sample_id}.pt')), \
            f'Cache not found: {cache_dir}/{sample_id}.pt'

    def __len__(self):
        return len(self.id_prop_data)

    def _load_from_cache(self, cif_id):
        data = torch.load(os.path.join(self.cache_dir, f'{cif_id}.pt'),
                          weights_only=True)
        return data['atom_fea'], data['nbr_fea'], data['nbr_fea_idx'], data['bandgap'], data['ehull']

    def _load_from_cif_augmented(self, cif_id):
        from pymatgen.core.structure import Structure
        cif_path = os.path.join(self.data_dir, f'{cif_id}.cif')
        s = Structure.from_file(cif_path)
        s = s.copy()
        if self.augment_perturb > 0:
            s.perturb(distance=self.augment_perturb)
        if self.augment_scale > 0:
            factor = (1.0 + random.uniform(-self.augment_scale, self.augment_scale)) ** 3
            s.scale_lattice(s.volume * factor)
        atom_fea, nbr_fea, nbr_fea_idx = _build_graph_from_structure(
            s, self._ari, self._gdf, self.max_num_nbr, self.radius, cif_id)
        row = next(r for r in self.id_prop_data if r[0] == cif_id)
        bandgap = torch.Tensor([float(row[1])])
        ehull = torch.Tensor([float(row[3])])
        return atom_fea, nbr_fea, nbr_fea_idx, bandgap, ehull

    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx][0]

        use_structure_aug = (
            self.augment and self.data_dir and self._ari is not None
            and self.augment_structure_prob > 0
            and random.random() < self.augment_structure_prob
        )
        if use_structure_aug:
            atom_fea, nbr_fea, nbr_fea_idx, bandgap, ehull = self._load_from_cif_augmented(cif_id)
        else:
            atom_fea, nbr_fea, nbr_fea_idx, bandgap, ehull = self._load_from_cache(cif_id)

        if self.augment and self.augment_noise > 0 and not use_structure_aug:
            atom_fea = atom_fea + torch.randn_like(atom_fea) * self.augment_noise

        gaptype = _map_gaptype(self.id_prop_data[idx][2],
                               self.merge_metal_indirect)
        eh_label = eh_to_label(self.id_prop_data[idx][3])

        targets = {
            'bandgap': bandgap,
            'gaptype': torch.LongTensor([gaptype]),
            'eh_label': torch.LongTensor([eh_label]),
            'ehull_raw': ehull,
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

    def get_gt_targets_for_indices(self, indices=None):
        if indices is None:
            indices = range(len(self.id_prop_data))
        return [_map_gaptype(self.id_prop_data[i][2], self.merge_metal_indirect)
                for i in indices]

    @property
    def n_gap_classes(self):
        return 2 if self.merge_metal_indirect else 3


# ---------------------------------------------------------------------------
# Losses — same as v9
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


class DynamicMultiTaskLoss(nn.Module):
    """Kendall MTL with optional log-var clamping."""
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


def apply_rules(bg_pred, gt_log_probs, merge_metal_indirect=True):
    gt_pred = (gt_log_probs.argmax(dim=1) if gt_log_probs.dim() > 1
               else gt_log_probs)
    bg = bg_pred.squeeze(-1)
    metal_mask = bg < 0.05
    indirect_class = 1 if merge_metal_indirect else 2
    gt_pred = gt_pred.clone()
    gt_pred[metal_mask] = indirect_class
    return gt_pred


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
# Args — v10 defaults
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Multi-task CGCNN v10')
    p.add_argument('data_dir', help='Path to data directory')
    p.add_argument('--cache-dir', default='',
                   help='Cached graph dir. If empty, falls back to CIF loading.')
    p.add_argument('--epochs', default=800, type=int)
    p.add_argument('-b', '--batch-size', default=48, type=int)
    p.add_argument('--lr', default=0.0008, type=float)
    p.add_argument('--weight-decay', default=1e-3, type=float)
    p.add_argument('--atom-fea-len', default=160, type=int)
    p.add_argument('--h-fea-len', default=320, type=int)
    p.add_argument('--n-conv', default=8, type=int)
    p.add_argument('--n-h', default=1, type=int)
    p.add_argument('--n-attn-heads', default=8, type=int)
    p.add_argument('--dropout', default=0.35, type=float)
    p.add_argument('-j', '--workers', default=0, type=int)
    p.add_argument('--print-freq', default=10, type=int)
    p.add_argument('--disable-cuda', action='store_true')
    p.add_argument('--patience', default=150, type=int)
    p.add_argument('--grad-clip', default=2.0, type=float)
    p.add_argument('--task-weights', nargs=3, type=float, default=[3.0, 5.0, 2.0],
                   help='Static weights [BG, GT, EH]')
    p.add_argument('--dynamic-weights', action='store_true', default=True)
    p.add_argument('--no-dynamic-weights', dest='dynamic_weights',
                   action='store_false')
    p.add_argument('--dw-warmup', default=0, type=int)
    p.add_argument('--mtl-clamp-min', default=-3.0, type=float)
    p.add_argument('--mtl-clamp-max', default=5.0, type=float)
    p.add_argument('--focal-gamma-gt', default=4.0, type=float)
    p.add_argument('--focal-gamma-eh', default=2.0, type=float)
    p.add_argument('--label-smoothing', default=0.08, type=float)
    p.add_argument('--consistency-weight', default=0.05, type=float)
    p.add_argument('--augment-noise', default=0.03, type=float)
    p.add_argument('--augment-structure', action='store_true', default=True)
    p.add_argument('--no-augment-structure', dest='augment_structure',
                   action='store_false')
    p.add_argument('--augment-perturb', default=0.03, type=float)
    p.add_argument('--augment-scale', default=0.02, type=float)
    p.add_argument('--augment-structure-prob', default=0.4, type=float)
    p.add_argument('--warmup-epochs', default=20, type=int)
    p.add_argument('--cosine-T0', default=150, type=int)
    p.add_argument('--cosine-Tmult', default=2, type=int)
    p.add_argument('--merge-metal-indirect', action='store_true', default=True)
    p.add_argument('--log-bg', action='store_true', default=True)
    p.add_argument('--robust-norm', action='store_true', default=True)
    p.add_argument('--no-robust-norm', dest='robust_norm', action='store_false')
    p.add_argument('--huber-delta', default=0.4, type=float)
    p.add_argument('--bg-reweight', default=0.3, type=float)
    p.add_argument('--eh-stable-thresh', default=STABLE_THRESH, type=float)
    p.add_argument('--use-weighted-sampler', action='store_true', default=True)
    p.add_argument('--no-weighted-sampler', dest='use_weighted_sampler',
                   action='store_false')
    p.add_argument('--k-folds', default=5, type=int)
    p.add_argument('--val-ratio', default=0.1, type=float)
    p.add_argument('--seed', default=42, type=int, help='Single random seed')
    p.add_argument('--fold', default=-1, type=int,
                   help='Run specific fold only (0-indexed). -1 = all folds.')
    p.add_argument('--checkpoint-dir', default='checkpoints/multitask_v10')
    p.add_argument('--use-gradnorm', action='store_true', default=False,
                   help='Use GradNorm MTL weighting instead of Kendall')
    p.add_argument('--gradnorm-lr', default=0.001, type=float,
                   help='Learning rate for GradNorm weight optimizer')
    p.add_argument('--gradnorm-alpha', default=1.5, type=float,
                   help='GradNorm alpha (asymmetry parameter)')
    p.add_argument('--gradnorm-update-freq', default=10, type=int,
                   help='Update GradNorm weights every N steps')
    p.add_argument('--use-cosine-classifier', action='store_true', default=False,
                   help='Use CosineClassifier for GT head')
    p.add_argument('--cosine-temp', default=0.1, type=float,
                   help='Initial temperature for CosineClassifier')
    p.add_argument('--gt-minority-boost', default=1.5, type=float,
                   help='Extra weight multiplier for minority GT class (Direct). '
                        'Set to 1.0 to disable.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single epoch — v10: GradNorm integration + consistency loss logging
# Efficiency fix: only compute autograd.grad on GradNorm update steps
# ---------------------------------------------------------------------------
def run_epoch(loader, model, criterion_bg, criterion_gt, criterion_eh,
              normalizer_bg, args, device, dataset,
              optimizer=None, mt_loss_module=None, gradnorm_module=None,
              gradnorm_optimizer=None, shared_params_list=None,
              is_train=True, epoch=0):
    model.train() if is_train else model.eval()
    if hasattr(dataset, 'augment'):
        dataset.augment = is_train

    tot_loss, n = 0.0, 0
    tot_cons_loss = 0.0
    tot_gradnorm_loss = 0.0
    all_bg_p, all_bg_t = [], []
    all_gt_logp, all_gt_t = [], []
    all_eh_logp, all_eh_t = [], []
    indirect_class = 1 if args.merge_metal_indirect else 2
    step_count = 0

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

            # --- MTL weighting ---
            gradnorm_loss_val = 0.0
            _do_gradnorm_step = False

            if is_train and gradnorm_module is not None and args.use_gradnorm:
                # Only run full GradNorm (expensive autograd) on update steps
                is_update_step = (step_count % args.gradnorm_update_freq == 0)
                if is_update_step and shared_params_list is not None:
                    losses_vec = torch.stack([loss_bg, loss_gt, loss_eh])
                    weighted_loss, gn_weights, gn_loss = gradnorm_module(
                        losses_vec, shared_params_list)
                    w_detached = gn_weights.detach()
                    loss = (w_detached * losses_vec).sum()
                    gradnorm_loss_val = gn_loss.item()
                    if gradnorm_optimizer is not None and gn_loss.requires_grad:
                        _do_gradnorm_step = True
                else:
                    # Non-update step: use cached weights, no autograd
                    w_detached = gradnorm_module.get_task_weights()
                    losses_vec = torch.stack([loss_bg, loss_gt, loss_eh])
                    loss = (w_detached * losses_vec).sum()

            elif mt_loss_module is not None and args.dynamic_weights:
                dw_warmup = getattr(args, 'dw_warmup', 0)
                if epoch > dw_warmup:
                    dyn_loss, _ = mt_loss_module(loss_bg, loss_gt, loss_eh)
                    if dw_warmup > 0:
                        alpha = min((epoch - dw_warmup) / 20.0, 1.0)
                        static_loss = (args.task_weights[0] * loss_bg +
                                       args.task_weights[1] * loss_gt +
                                       args.task_weights[2] * loss_eh)
                        loss = (1 - alpha) * static_loss + alpha * dyn_loss
                    else:
                        loss = dyn_loss
                else:
                    w_bg, w_gt, w_eh = args.task_weights
                    loss = w_bg * loss_bg + w_gt * loss_gt + w_eh * loss_eh
            else:
                w_bg, w_gt, w_eh = args.task_weights
                loss = w_bg * loss_bg + w_gt * loss_gt + w_eh * loss_eh

            # Consistency loss
            cons_val = 0.0
            if args.consistency_weight > 0:
                bg_denorm = normalizer_bg.denorm(p_bg.detach())
                cons = physical_consistency_loss(
                    bg_denorm, p_gt, indirect_class=indirect_class)
                loss = loss + args.consistency_weight * cons
                cons_val = cons.item()

        if is_train and optimizer is not None:
            # GradNorm weight update FIRST (before model backward frees graph)
            if _do_gradnorm_step:
                gradnorm_optimizer.zero_grad()
                gn_loss.backward(retain_graph=True)
                gradnorm_optimizer.step()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                params = list(model.parameters())
                if mt_loss_module is not None:
                    params += list(mt_loss_module.parameters())
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

        tot_loss += loss.item()
        tot_cons_loss += cons_val
        tot_gradnorm_loss += gradnorm_loss_val
        n += 1
        step_count += 1
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
    metrics = {
        'bg_mae': float(np.mean(np.abs(bg_p - bg_t))),
        'gt_acc': float(accuracy_score(gt_t, gt_p)),
        'gt_f1':  float(f1_score(gt_t, gt_p, average=avg_type, zero_division=0)),
        'eh_acc': float(accuracy_score(eh_t_arr, eh_p)),
        'eh_f1':  float(f1_score(eh_t_arr, eh_p, average='binary', zero_division=0)),
        'cons_loss': tot_cons_loss / max(n, 1),
        'gradnorm_loss': tot_gradnorm_loss / max(n, 1),
    }
    return tot_loss / max(n, 1), metrics


# ---------------------------------------------------------------------------
# Train fold
# ---------------------------------------------------------------------------
def _fold_is_complete(fold_dir, min_size_mb=8):
    required = ['best_composite.pth.tar', 'best_bg_mae.pth.tar', 'best_gt_f1.pth.tar']
    for fname in required:
        p = os.path.join(fold_dir, fname)
        if not os.path.isfile(p) or os.path.getsize(p) < min_size_mb * 1024 * 1024:
            return False
    return True


def load_completed_fold(dataset, test_idx, fold_idx, seed, args, device):
    fold_dir = os.path.join(args.checkpoint_dir, f'fold_{fold_idx}_seed_{seed}')
    print(f'  [SKIP] fold_{fold_idx}_seed_{seed} already complete')

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in test_idx])
    n_bg = Normalizer(tr_bg, log_transform=args.log_bg, robust=args.robust_norm)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV10(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=dataset.n_gap_classes, n_eh_classes=2,
        dropout=args.dropout, n_attn_heads=args.n_attn_heads,
        use_cosine_classifier=args.use_cosine_classifier,
        cosine_temp=args.cosine_temp,
    ).to(device)

    kw = dict(batch_size=args.batch_size, collate_fn=collate_pool_mt_v7,
              num_workers=args.workers, pin_memory=device.type == 'cuda')
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    gt_cw = dataset.get_gt_class_weights().to(device)
    if args.gt_minority_boost != 1.0:
        gt_cw[0] = gt_cw[0] * args.gt_minority_boost
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
        gt_targets = [None] * len(dataset)
        for i in range(len(dataset)):
            gt_targets[i] = _map_gaptype(dataset.id_prop_data[i][2],
                                         args.merge_metal_indirect)
        sampler = WeightedSubsetSampler(train_idx, gt_targets)
        train_loader = DataLoader(dataset, sampler=sampler, **kw)
    else:
        train_loader = DataLoader(dataset,
                                  sampler=SubsetRandomSampler(train_idx), **kw)

    val_loader = DataLoader(dataset, sampler=SubsetRandomSampler(val_idx), **kw)
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx), **kw)

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in train_idx])
    n_bg = Normalizer(tr_bg, log_transform=args.log_bg, robust=args.robust_norm)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV10(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=args.atom_fea_len, n_conv=args.n_conv,
        h_fea_len=args.h_fea_len, n_h=args.n_h,
        n_gap_classes=dataset.n_gap_classes, n_eh_classes=2,
        dropout=args.dropout, n_attn_heads=args.n_attn_heads,
        use_cosine_classifier=args.use_cosine_classifier,
        cosine_temp=args.cosine_temp,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Model params: {n_params:,}')

    criterion_bg = WeightedHuberLoss(delta=args.huber_delta,
                                     reweight_scale=args.bg_reweight)
    gt_cw = dataset.get_gt_class_weights().to(device)
    if args.gt_minority_boost != 1.0:
        gt_cw[0] = gt_cw[0] * args.gt_minority_boost
    print(f'  GT class weights (with boost={args.gt_minority_boost}): {gt_cw.tolist()}')
    criterion_gt = FocalLossWithSmoothing(
        alpha=gt_cw, gamma=args.focal_gamma_gt,
        smoothing=args.label_smoothing)

    eh_cw = dataset.get_eh_class_weights().to(device)
    print(f'  EH class weights: {eh_cw.tolist()}')
    criterion_eh = FocalLossWithSmoothing(
        alpha=eh_cw, gamma=args.focal_gamma_eh,
        smoothing=args.label_smoothing)

    # --- MTL weighting setup ---
    mt_loss = None
    gradnorm_module = None
    gradnorm_optimizer = None
    shared_params_list = None
    opt_params = list(model.parameters())

    if args.use_gradnorm:
        gradnorm_module = GradNorm(
            n_tasks=3, alpha=args.gradnorm_alpha).to(device)
        gradnorm_optimizer = optim.Adam(
            gradnorm_module.parameters(), lr=args.gradnorm_lr)
        # Cache shared params list once per fold (not per step)
        shared_params_list = list(model.get_shared_params())
        print(f'  GradNorm MTL (alpha={args.gradnorm_alpha}, '
              f'lr={args.gradnorm_lr}, update_freq={args.gradnorm_update_freq})')
    elif args.dynamic_weights:
        mt_loss = DynamicMultiTaskLoss(
            log_var_min=args.mtl_clamp_min,
            log_var_max=args.mtl_clamp_max,
        ).to(device)
        opt_params += list(mt_loss.parameters())
        print(f'  Kendall MTL (clamp=[{args.mtl_clamp_min},{args.mtl_clamp_max}])')
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
            optimizer=optimizer, mt_loss_module=mt_loss,
            gradnorm_module=gradnorm_module,
            gradnorm_optimizer=gradnorm_optimizer,
            shared_params_list=shared_params_list,
            is_train=True, epoch=epoch)
        v_loss, v_m = run_epoch(
            val_loader, model, criterion_bg, criterion_gt, criterion_eh,
            n_bg, args, device, dataset, is_train=False, epoch=epoch)
        scheduler.step()

        composite = (-2.0 * v_m['bg_mae'] + 3.0 * v_m['gt_f1']
                     + 1.0 * v_m['eh_acc'])

        ckpt = {'epoch': epoch, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'normalizer_bg': n_bg.state_dict(),
                'args': vars(args)}
        if mt_loss is not None:
            ckpt['mt_loss'] = mt_loss.state_dict()
        if gradnorm_module is not None:
            ckpt['gradnorm'] = gradnorm_module.state_dict()

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
            if gradnorm_module is not None:
                gw = F.softplus(gradnorm_module.weights).detach().cpu()
                dw_str = f' gn=[{gw[0]:.2f},{gw[1]:.2f},{gw[2]:.2f}]'
            elif mt_loss is not None:
                wb, wg, we = mt_loss.get_weights()
                dw_str = f' dw=[{wb:.2f},{wg:.2f},{we:.2f}]'
            cons_str = f' cons={v_m["cons_loss"]:.4f}' if args.consistency_weight > 0 else ''
            gn_str = f' gnL={t_m["gradnorm_loss"]:.4f}' if args.use_gradnorm else ''
            print(f'  Ep {epoch:03d} lr={lr_now:.5f}{dw_str} | '
                  f'TrL {t_loss:.3f} VL {v_loss:.3f} | '
                  f'BG {v_m["bg_mae"]:.4f} GT {v_m["gt_acc"]:.3f}/'
                  f'{v_m["gt_f1"]:.3f} '
                  f'EH_acc {v_m["eh_acc"]:.3f} | '
                  f'comp={composite:.3f}{cons_str}{gn_str}')

        if args.patience > 0 and no_improve >= args.patience:
            print(f'  Early stop at epoch {epoch} '
                  f'(best comp ep {best_ep_comp})')
            break

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
# Main — v10: single seed, --fold for array jobs
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Device: {device}')
    print(f'ChalcoGNN v10 | seed={args.seed} | k_folds={args.k_folds} | '
          f'fold={args.fold if args.fold >= 0 else "all"}')
    print(f'GradNorm: {args.use_gradnorm} | CosineClassifier: {args.use_cosine_classifier}')
    print(f'Structure aug: {args.augment_structure} (prob={args.augment_structure_prob})')

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
            augment_noise=args.augment_noise,
            data_dir=args.data_dir if args.augment_structure else None,
            augment_perturb=args.augment_perturb,
            augment_scale=args.augment_scale,
            augment_structure_prob=args.augment_structure_prob)
    else:
        raise RuntimeError('v10 requires cached graphs. Run preprocess_graphs.py first.')

    print(f'Dataset: {len(dataset)} samples')

    gt_counts = {}
    eh_counts = {0: 0, 1: 0}
    for row in dataset.id_prop_data:
        l = _map_gaptype(row[2], args.merge_metal_indirect)
        gt_counts[l] = gt_counts.get(l, 0) + 1
        eh_counts[eh_to_label(row[3])] += 1
    print(f'GT class dist: {dict(sorted(gt_counts.items()))}')
    print(f'EH class dist: {eh_counts}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()

    folds = stratified_kfold_v4(
        dataset.id_prop_data, args.k_folds,
        merge_metal_indirect=args.merge_metal_indirect)

    # Determine which folds to run
    if args.fold >= 0:
        fold_range = [args.fold]
    else:
        fold_range = list(range(args.k_folds))

    all_results = {'composite': [], 'gt_f1': [], 'bg_mae': []}
    seed = args.seed

    for fi in fold_range:
        aligned_split = load_aligned_split_indices(dataset.id_prop_data, args.data_dir, fi)
        if aligned_split is not None:
            train_idx, val_idx, test_idx = aligned_split
        else:
            test_idx = folds[fi]
            train_val_idx = []
            for j in range(args.k_folds):
                if j != fi:
                    train_val_idx.extend(folds[j])

            train_ratio = 1.0 - args.val_ratio
            train_idx, val_idx = stratified_split_from_indices(
                dataset.id_prop_data, train_val_idx,
                train_ratio=train_ratio,
                random_seed=123 + fi,
                merge_metal_indirect=args.merge_metal_indirect)

        print(f'\n{"="*60}')
        print(f'Fold {fi}/{args.k_folds-1} (seed {seed}) | '
              f'Train {len(train_idx)} Val {len(val_idx)} Test {len(test_idx)}')
        print(f'{"="*60}')

        fold_dir_check = os.path.join(args.checkpoint_dir,
                                      f'fold_{fi}_seed_{seed}')
        if _fold_is_complete(fold_dir_check):
            results, fold_dir, model, n_bg = load_completed_fold(
                dataset, test_idx, fi, seed, args, device)
        else:
            if os.path.isdir(fold_dir_check):
                import shutil
                shutil.rmtree(fold_dir_check)
                print(f'  [WARN] Removed incomplete: {fold_dir_check}')
            results, fold_dir, model, n_bg = train_fold(
                dataset, train_idx, val_idx, test_idx, fi, seed,
                args, device)

        for tag in all_results:
            if tag in results:
                all_results[tag].append(results[tag])
        r = results["composite"]
        print(f'  Test composite: '
              f'BG {r["bg_mae"]:.4f} '
              f'GT {r["gt_acc"]:.4f}/{r["gt_f1"]:.4f} '
              f'EH {r["eh_acc"]:.4f}/{r["eh_f1"]:.4f}')

        results_path = os.path.join(args.checkpoint_dir, f'fold{fi}_results.csv')
        with open(results_path, 'w', newline='') as _f:
            writer = csv.writer(_f)
            writer.writerow(['metric', 'value'])
            for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']:
                writer.writerow([k, f'{r[k]:.6f}'])
        print(f'  Results saved to {results_path}')

    # Summary (only meaningful when running all folds)
    if len(fold_range) > 1:
        print(f'\n{"="*60}')
        print(f'{args.k_folds}-Fold CV Summary (v10, seed={seed})')
        print(f'{"="*60}')
        for tag in ['composite', 'gt_f1', 'bg_mae']:
            if all_results[tag]:
                print(f'\n  Checkpoint: best_{tag}')
                for key in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']:
                    vals = [m[key] for m in all_results[tag]]
                    unit = ' eV' if key == 'bg_mae' else ''
                    print(f'    {key:8s}: {np.mean(vals):.4f} '
                          f'+/- {np.std(vals):.4f}{unit}')

        ref = {k: np.mean([m[k] for m in all_results['composite']])
               for k in ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']}
        print(f'\n{"="*60}')
        print('Target check (best_composite):')
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
