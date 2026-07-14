"""
Training script for Dual-Backbone MACE + ALIGNN fusion (Plan A).

Loads pre-trained MACE (from MP bandgap pre-training) and pre-trained ALIGNN
(from chalcohalide fold-specific training), fuses their representations via
gated fusion, and trains shared task heads.

Both backbones start frozen; progressive unfreezing enables fine-tuning.
"""
from __future__ import print_function, division

import argparse
import os
import sys
import time
import csv
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from model_dual_backbone import DualBackboneMultiTask
from model_alignn_pyg import ALIGNNMultiTaskPyG
from data_dual_backbone import DualBackboneDataset, stratified_kfold_split


# ───────────────────── Loss functions ─────────────────────

class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.4):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, weights=None):
        diff = pred.squeeze() - target.squeeze()
        abs_diff = diff.abs()
        quadratic = torch.clamp(abs_diff, max=self.delta)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        if weights is not None:
            loss = loss * weights
        return loss.mean()


class FocalNLLLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, log_probs, targets):
        targets = targets.reshape(-1)
        log_probs = log_probs.view(targets.size(0), -1)
        n_classes = log_probs.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(log_probs,
                                         self.label_smoothing / max(n_classes - 1, 1))
                smooth.scatter_(1, targets.unsqueeze(1),
                                1.0 - self.label_smoothing)
            nll = -(smooth * log_probs).sum(dim=1)
            pt = (smooth * log_probs.exp()).sum(dim=1)
        else:
            nll = F.nll_loss(log_probs, targets, reduction='none')
            pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1 - pt).pow(self.gamma) * nll
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


class KendallMTL(nn.Module):
    def __init__(self, n_tasks=3, clamp_min=-3.0, clamp_max=5.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, losses):
        log_vars = self.log_vars.clamp(self.clamp_min, self.clamp_max)
        precisions = torch.exp(-log_vars)
        weighted = (precisions * losses + log_vars).sum()
        return weighted, precisions.detach()


# ───────────────────── Normalizer & Metrics ─────────────────────

class Normalizer:
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = tensor.mean().item()
            self.std = tensor.std().item()
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, t):
        return (t - self.mean) / (self.std + 1e-8)

    def denorm(self, t):
        return t * (self.std + 1e-8) + self.mean


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mae_metric(pred, target):
    return (pred - target).abs().mean().item()


def accuracy(log_probs, targets):
    return (log_probs.argmax(dim=1) == targets).float().mean().item()


# ───────────────────── Training ─────────────────────

def train_epoch(model, loader, optimizer, criterion_bg, criterion_gt,
                criterion_eh, mtl_module, normalizer, consistency_weight,
                device, log_bg=True):
    model.train()
    losses_m = AverageMeter()
    bg_mae_m = AverageMeter()
    gt_acc_m = AverageMeter()
    eh_acc_m = AverageMeter()

    for batch in loader:
        batch = batch.to(device)
        bg_pred, gt_pred, eh_pred = model(batch)

        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).view(-1)
        eh_target = batch.eh.to(device).view(-1)

        loss_bg = criterion_bg(bg_pred.squeeze(), bg_target.squeeze())
        gt_known = gt_target >= 0
        if gt_known.any():
            loss_gt = criterion_gt(gt_pred[gt_known], gt_target[gt_known])
        else:
            loss_gt = torch.zeros(1, device=device, requires_grad=True).squeeze()
        loss_eh = criterion_eh(eh_pred, eh_target)

        # Consistency
        bg_raw = normalizer.denorm(bg_pred.squeeze())
        bg_ev = torch.expm1(bg_raw.detach()) if log_bg else bg_raw.detach()
        mask_small = bg_ev < 0.1
        if mask_small.any() and consistency_weight > 0:
            cons_loss = consistency_weight * gt_pred.exp()[mask_small, 1].mean()
        else:
            cons_loss = torch.tensor(0.0, device=device)

        losses_vec = torch.stack([loss_bg, loss_gt, loss_eh])
        total_loss, task_weights = mtl_module(losses_vec)
        total_loss = total_loss + cons_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        with torch.no_grad():
            bg_denorm = normalizer.denorm(bg_pred.squeeze())
            bg_target_denorm = normalizer.denorm(bg_target.squeeze())
            if log_bg:
                bg_ev_pred = torch.expm1(bg_denorm)
                bg_ev_true = torch.expm1(bg_target_denorm)
            else:
                bg_ev_pred, bg_ev_true = bg_denorm, bg_target_denorm
            mae = mae_metric(bg_ev_pred, bg_ev_true)

        n = batch.bg.size(0)
        losses_m.update(total_loss.item(), n)
        bg_mae_m.update(mae, n)
        gt_known_cpu = gt_target >= 0
        if gt_known_cpu.any():
            gt_acc_m.update(accuracy(gt_pred[gt_known_cpu], gt_target[gt_known_cpu]),
                            gt_known_cpu.sum().item())
        else:
            gt_acc_m.update(0.0, 1)
        eh_acc_m.update(accuracy(eh_pred, eh_target), n)

    return losses_m.avg, bg_mae_m.avg, gt_acc_m.avg, eh_acc_m.avg, task_weights


@torch.no_grad()
def validate(model, loader, criterion_bg, criterion_gt, criterion_eh,
             normalizer, device, log_bg=True):
    model.eval()
    bg_mae_m = AverageMeter()
    gt_preds_all, gt_targets_all = [], []
    eh_preds_all, eh_targets_all = [], []
    losses_total = AverageMeter()

    for batch in loader:
        batch = batch.to(device)
        bg_pred, gt_pred, eh_pred = model(batch)

        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).view(-1)
        eh_target = batch.eh.to(device).view(-1)

        n = batch.bg.size(0)
        l_bg = criterion_bg(bg_pred.squeeze(), bg_target.squeeze()).item()
        gt_known_v = gt_target >= 0
        l_gt = criterion_gt(gt_pred[gt_known_v], gt_target[gt_known_v]).item() if gt_known_v.any() else 0.0
        l_eh = criterion_eh(eh_pred, eh_target).item()
        losses_total.update(l_bg + l_gt + l_eh, n)

        bg_denorm = normalizer.denorm(bg_pred.squeeze())
        bg_target_denorm = normalizer.denorm(bg_target.squeeze())
        if log_bg:
            bg_ev_pred = torch.expm1(bg_denorm)
            bg_ev_true = torch.expm1(bg_target_denorm)
        else:
            bg_ev_pred, bg_ev_true = bg_denorm, bg_target_denorm
        bg_mae_m.update(mae_metric(bg_ev_pred, bg_ev_true), n)

        gt_known_v = gt_target >= 0
        if gt_known_v.any():
            gt_preds_all.append(gt_pred.argmax(dim=1)[gt_known_v].cpu())
            gt_targets_all.append(gt_target[gt_known_v].cpu())
        eh_preds_all.append(eh_pred.argmax(dim=1).cpu())
        eh_targets_all.append(eh_target.cpu())

    gt_p = torch.cat(gt_preds_all).numpy()
    gt_t = torch.cat(gt_targets_all).numpy()
    eh_p = torch.cat(eh_preds_all).numpy()
    eh_t = torch.cat(eh_targets_all).numpy()

    return {
        "val_loss": losses_total.avg,
        "bg_mae": bg_mae_m.avg,
        "gt_acc": (gt_p == gt_t).mean(),
        "gt_f1": f1_score(gt_t, gt_p, average='macro', zero_division=0),
        "eh_acc": (eh_p == eh_t).mean(),
        "eh_f1": f1_score(eh_t, eh_p, average='macro', zero_division=0),
    }


def load_alignn_model(checkpoint_path, device, hidden_dim=256, dropout=0.3):
    """Load pre-trained ALIGNN PyG model from checkpoint."""
    model = ALIGNNMultiTaskPyG(
        atom_input_dim=94, edge_input_dim=80, angle_input_dim=40,
        hidden_dim=hidden_dim, n_alignn_layers=4, n_gcn_layers=4,
        n_gap_classes=2, n_eh_classes=2, dropout=dropout,
    )
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    # Handle different checkpoint formats
    state_dict = ckpt.get("model_state_dict",
                          ckpt.get("state_dict", ckpt))
    if isinstance(state_dict, dict) and not any(
            k.startswith("atom_proj") for k in state_dict):
        # Might be a raw state dict
        pass
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Dual-Backbone MACE+ALIGNN Training (Plan A)")
    parser.add_argument("data_dir")
    parser.add_argument("--mace-cache-dir", default=None)
    parser.add_argument("--crystal-cache-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/dual_backbone")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Pre-trained checkpoints
    parser.add_argument("--mace-pretrained", required=True,
                        help="Path to pre-trained MACE checkpoint")
    parser.add_argument("--alignn-checkpoint-dir", required=True,
                        help="Dir with ALIGNN fold{N}_best.pt checkpoints")

    # Architecture
    parser.add_argument("--h-fea-len", type=int, default=256)
    parser.add_argument("--n-attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-cosine-classifier", action="store_true",
                        default=True)
    parser.add_argument("--cosine-temp", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mace-backbone-lr", type=float, default=5e-6)
    parser.add_argument("--alignn-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-epoch", type=int, default=30,
                        help="Epoch to unfreeze MACE backbone (last N layers)")
    parser.add_argument("--unfreeze-epoch-2", type=int, default=50,
                        help="Epoch to unfreeze all MACE + ALIGNN backbone")
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--cosine-T0", type=int, default=120)
    parser.add_argument("--cosine-Tmult", type=int, default=2)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    # Model selection
    parser.add_argument("--min-save-epoch", type=int, default=30)
    parser.add_argument("--gt-composite-weight", type=float, default=0.5)
    parser.add_argument("--eh-composite-weight", type=float, default=0.3)

    # GT conditioning
    parser.add_argument("--cond-anneal-epochs", type=int, default=40)

    # Loss
    parser.add_argument("--huber-delta", type=float, default=0.4)
    parser.add_argument("--focal-gamma-gt", type=float, default=3.0)
    parser.add_argument("--focal-gamma-eh", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--gt-minority-boost", type=float, default=2.0)

    # Data
    parser.add_argument("--log-bg", action="store_true", default=True)
    parser.add_argument("--merge-metal-indirect", action="store_true",
                        default=True)

    # MTL
    parser.add_argument("--mtl-clamp-min", type=float, default=-3.0)
    parser.add_argument("--mtl-clamp-max", type=float, default=5.0)

    # Misc
    parser.add_argument("--eval-only", action="store_true", default=False,
                        help="skip training, run test evaluation only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--mace-model", default="small")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Plan A: Dual-Backbone MACE+ALIGNN | seed={args.seed} | fold={args.fold}")

    # ─── Load MACE backbone ───
    print(f"Loading MACE-MP-0 ({args.mace_model})...")
    from mace.calculators import mace_mp
    calc = mace_mp(model=args.mace_model, default_dtype="float32",
                   device=str(device))
    mace_model = calc.models[0]

    # ─── Load ALIGNN model ───
    alignn_ckpt = os.path.join(args.alignn_checkpoint_dir,
                               f"fold{args.fold}_best.pt")
    print(f"Loading ALIGNN checkpoint: {alignn_ckpt}")
    alignn_model = load_alignn_model(alignn_ckpt, device, dropout=args.dropout)
    print(f"  ALIGNN loaded: {sum(p.numel() for p in alignn_model.parameters()):,} params")

    # ─── Dataset ───
    mace_cache = args.mace_cache_dir or \
        os.path.join(args.data_dir, "mace_cached_graphs")
    crystal_cache = args.crystal_cache_dir or \
        os.path.join(args.data_dir, "crystal_cached_graphs")

    dataset = DualBackboneDataset(
        root_dir=args.data_dir,
        mace_cache_dir=mace_cache,
        crystal_cache_dir=crystal_cache,
        merge_metal_indirect=args.merge_metal_indirect,
        log_bg=args.log_bg,
    )

    gt_dist = dataset.get_gt_class_dist()
    eh_dist = dataset.get_eh_class_dist()
    print(f"GT dist: {dict(gt_dist)}, EH dist: {dict(eh_dist)}")

    train_idx, val_idx, test_idx = stratified_kfold_split(
        dataset, k_folds=args.k_folds, fold=args.fold,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"Fold {args.fold}: Train {len(train_idx)} Val {len(val_idx)} "
          f"Test {len(test_idx)}")

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers,
                            pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    # ─── Normalizer ───
    train_bg = torch.tensor([dataset.samples[i]["bg"] for i in train_idx])
    normalizer = Normalizer(train_bg)

    # ─── Model ───
    model = DualBackboneMultiTask(
        mace_model, alignn_model,
        h_fea_len=args.h_fea_len,
        n_attn_heads=args.n_attn_heads,
        dropout=args.dropout,
        use_cosine_classifier=args.use_cosine_classifier,
        cosine_temp=args.cosine_temp,
    ).to(device)

    # Load pre-trained MACE weights (backbone + attention pool)
    if args.mace_pretrained:
        print(f"Loading MACE pre-trained: {args.mace_pretrained}")
        ckpt = torch.load(args.mace_pretrained, weights_only=False,
                          map_location=device)
        model_dict = model.state_dict()
        src_dict = ckpt.get("model_state_dict", ckpt)
        loaded = 0
        for k, v in src_dict.items():
            # Map from MACEMultiTask keys to DualBackbone keys
            # backbone.X → mace_backbone.X, attn_pool.X → mace_pool.X
            new_k = k
            if k.startswith("backbone."):
                new_k = "mace_" + k  # backbone.X → mace_backbone.X
            elif k.startswith("attn_pool."):
                new_k = "mace_pool." + k[len("attn_pool."):]
            if new_k in model_dict and v.shape == model_dict[new_k].shape:
                model_dict[new_k] = v
                loaded += 1
        model.load_state_dict(model_dict)
        print(f"  Loaded {loaded} params from MACE pre-trained")

    # Freeze both backbones initially
    model.freeze_mace_backbone()
    model.freeze_alignn_backbone()
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total, {n_train:,} trainable (both frozen)")

    # ─── Loss ───
    n0, n1 = gt_dist.get(0, 1), gt_dist.get(1, 1)
    total_gt = n0 + n1
    w_gt = torch.tensor([total_gt / (2 * n0) * args.gt_minority_boost,
                         total_gt / (2 * n1)], device=device)
    n0_eh, n1_eh = eh_dist.get(0, 1), eh_dist.get(1, 1)
    total_eh = n0_eh + n1_eh
    w_eh = torch.tensor([total_eh / (2 * n0_eh),
                         total_eh / (2 * n1_eh)], device=device)

    criterion_bg = WeightedHuberLoss(delta=args.huber_delta)
    criterion_gt = FocalNLLLoss(gamma=args.focal_gamma_gt, weight=w_gt,
                                label_smoothing=args.label_smoothing)
    criterion_eh = FocalNLLLoss(gamma=args.focal_gamma_eh, weight=w_eh,
                                label_smoothing=args.label_smoothing)

    mtl_module = KendallMTL(n_tasks=3, clamp_min=args.mtl_clamp_min,
                            clamp_max=args.mtl_clamp_max).to(device)

    # ─── Optimizer ───
    head_params = model.get_head_params()
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr, "weight_decay": 5e-4},
        {"params": mtl_module.parameters(), "lr": args.lr * 0.1},
    ])

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult)
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_epochs)

    # ─── Training loop ───
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_composite = float("-inf")
    patience_counter = 0
    backbone_phase = 0  # 0=frozen, 1=MACE partial, 2=all unfrozen

    for epoch in (range(1, args.epochs + 1) if not args.eval_only else []):
        # GT conditioning annealing
        cw = min(1.0, epoch / args.cond_anneal_epochs) \
            if args.cond_anneal_epochs > 0 else 1.0
        model.set_cond_weight(cw)

        # Progressive unfreezing:
        # Phase 1 (epoch 30): MACE last layer + ALIGNN last 2 layers
        # Phase 2 (epoch 50): all MACE + all ALIGNN
        if epoch == args.unfreeze_epoch and backbone_phase == 0:
            print(f"\n>>> Phase 1: Unfreezing MACE last {args.unfreeze_layers} "
                  f"+ ALIGNN last 2 layers...")
            model.unfreeze_mace_backbone(last_n_layers=args.unfreeze_layers)
            model.unfreeze_alignn_backbone(last_n_layers=2)

            mace_bp = [p for p in model.get_mace_backbone_params()
                       if p.requires_grad]
            alignn_bp = [p for p in model.get_alignn_backbone_params()
                         if p.requires_grad]
            if mace_bp:
                optimizer.add_param_group({
                    "params": mace_bp, "lr": args.mace_backbone_lr,
                    "weight_decay": 1e-5,
                })
            if alignn_bp:
                optimizer.add_param_group({
                    "params": alignn_bp, "lr": args.alignn_backbone_lr,
                    "weight_decay": 1e-5,
                })
            n_train = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
            print(f"    Trainable: {n_train:,}")
            backbone_phase = 1

        if epoch == args.unfreeze_epoch_2 and backbone_phase == 1:
            print(f"\n>>> Phase 2: Unfreezing all backbone layers...")
            model.unfreeze_mace_backbone()
            model.unfreeze_alignn_backbone()

            already = {id(p) for pg in optimizer.param_groups
                       for p in pg["params"]}
            new_mace = [p for p in model.get_mace_backbone_params()
                        if p.requires_grad and id(p) not in already]
            new_alignn = [p for p in model.get_alignn_backbone_params()
                          if p.requires_grad and id(p) not in already]
            if new_mace:
                optimizer.add_param_group({
                    "params": new_mace,
                    "lr": args.mace_backbone_lr * 0.4,
                    "weight_decay": 1e-5,
                })
            if new_alignn:
                optimizer.add_param_group({
                    "params": new_alignn,
                    "lr": args.alignn_backbone_lr * 0.4,
                    "weight_decay": 1e-5,
                })
            n_train = sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
            print(f"    Trainable: {n_train:,}")
            backbone_phase = 2

        t0 = time.time()
        train_loss, train_mae, train_gt, train_eh, tw = train_epoch(
            model, train_loader, optimizer, criterion_bg, criterion_gt,
            criterion_eh, mtl_module, normalizer, args.consistency_weight,
            device, log_bg=args.log_bg,
        )

        if epoch <= args.warmup_epochs and warmup_scheduler:
            warmup_scheduler.step()
        else:
            scheduler.step()

        val_metrics = validate(
            model, val_loader, criterion_bg, criterion_gt, criterion_eh,
            normalizer, device, log_bg=args.log_bg)

        elapsed = time.time() - t0
        dw = tw.cpu().numpy()

        if epoch % args.print_freq == 0 or epoch <= 5 or \
           epoch in (args.unfreeze_epoch, args.unfreeze_epoch_2):
            print(f"  Ep {epoch:03d} cw={cw:.2f} "
                  f"dw=[{dw[0]:.2f},{dw[1]:.2f},{dw[2]:.2f}] "
                  f"| TrL {train_loss:.3f} VL {val_metrics['val_loss']:.3f} "
                  f"| BG {val_metrics['bg_mae']:.4f} "
                  f"GT {val_metrics['gt_acc']:.3f}/{val_metrics['gt_f1']:.3f} "
                  f"EH {val_metrics['eh_acc']:.3f} "
                  f"| {elapsed:.1f}s")

        composite = (-val_metrics["bg_mae"]
                     + args.gt_composite_weight * val_metrics["gt_acc"]
                     + args.eh_composite_weight * val_metrics["eh_acc"])

        if epoch >= args.min_save_epoch and composite > best_composite:
            best_composite = composite
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mtl_state_dict": mtl_module.state_dict(),
                "normalizer_mean": normalizer.mean,
                "normalizer_std": normalizer.std,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt"))
        elif epoch >= args.min_save_epoch:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ─── Test ───
    print(f"\n{'='*60}")
    ckpt_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        vm = ckpt["val_metrics"]
        print(f"Best model ep {ckpt['epoch']} "
              f"(BG={vm['bg_mae']:.4f} GT={vm['gt_acc']:.3f} EH={vm['eh_acc']:.3f})")
    else:
        print("WARNING: No checkpoint saved, using final model state")

    model.set_cond_weight(1.0)
    test_metrics = validate(
        model, test_loader, criterion_bg, criterion_gt, criterion_eh,
        normalizer, device, log_bg=args.log_bg)

    print(f"\n  TEST RESULTS (Fold {args.fold}):")
    print(f"  BG MAE:  {test_metrics['bg_mae']:.4f} eV")
    print(f"  GT Acc:  {test_metrics['gt_acc']:.4f}  F1: {test_metrics['gt_f1']:.4f}")
    print(f"  EH Acc:  {test_metrics['eh_acc']:.4f}  F1: {test_metrics['eh_f1']:.4f}")

    results_path = os.path.join(args.checkpoint_dir,
                                f"fold{args.fold}_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in test_metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
