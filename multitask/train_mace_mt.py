"""
Training script for MACE multi-task chalcohalide model (v3).

Fixes over v2:
  1. Model selection: min_save_epoch + balanced composite (GT weight 0.5)
  2. Progressive backbone unfreezing (2-stage)
  3. Regularization: lower LR, stronger weight decay, grad clip 2.0
  4. GT conditioning annealing (0→1 over first 40 epochs)
  5. Larger val set (15%)
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

from model_mace_mt import MACEMultiTask
from data_mace_mt import MACECrystalDataset, stratified_kfold_split


# ───────────────────── Loss functions ─────────────────────

class WeightedHuberLoss(nn.Module):
    """Huber loss with optional sample weighting."""
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
    """Focal loss on log-probabilities for class imbalance."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, log_probs, targets):
        n_classes = log_probs.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.label_smoothing / max(n_classes - 1, 1))
                targets = targets.view(-1)
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            nll = -(smooth * log_probs).sum(dim=1)
            pt = (smooth * log_probs.exp()).sum(dim=1)
        else:
            nll = F.nll_loss(log_probs, targets, reduction='none')
            pt = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        focal = (1 - pt).pow(self.gamma) * nll

        if self.weight is not None:
            w = self.weight[targets]
            focal = focal * w

        return focal.mean()


class KendallMTL(nn.Module):
    """Kendall uncertainty-based multi-task loss weighting."""
    def __init__(self, n_tasks=3, clamp_min=-3.0, clamp_max=5.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, losses):
        log_vars = self.log_vars.clamp(self.clamp_min, self.clamp_max)
        precisions = torch.exp(-log_vars)
        weighted = (precisions * losses + log_vars).sum()
        weights = precisions.detach()
        return weighted, weights

    def get_task_weights(self):
        log_vars = self.log_vars.detach().clamp(self.clamp_min, self.clamp_max)
        return torch.exp(-log_vars)


# ───────────────────── Normalizer ─────────────────────

class Normalizer:
    """Running mean/std normalizer for bandgap targets."""
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = tensor.mean().item()
            self.std = tensor.std().item()
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)

    def denorm(self, tensor):
        return tensor * (self.std + 1e-8) + self.mean


# ───────────────────── Metrics ─────────────────────

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def mae_metric(pred, target):
    return (pred - target).abs().mean().item()


def accuracy(log_probs, targets):
    preds = log_probs.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ───────────────────── Training ─────────────────────

def train_epoch(model, loader, optimizer, criterion_bg, criterion_gt, criterion_eh,
                mtl_module, normalizer, consistency_weight, device,
                log_bg=True, print_freq=20):
    model.train()
    losses_m = AverageMeter()
    bg_mae_m = AverageMeter()
    gt_acc_m = AverageMeter()
    eh_acc_m = AverageMeter()

    for i, batch in enumerate(loader):
        batch = batch.to(device)
        data_dict = batch.to_dict()

        bg_pred, gt_pred, eh_pred = model(data_dict)

        # Targets
        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).squeeze()
        eh_target = batch.eh.to(device).squeeze()

        # Task losses
        loss_bg = criterion_bg(bg_pred.squeeze(), bg_target.squeeze())
        loss_gt = criterion_gt(gt_pred, gt_target)
        loss_eh = criterion_eh(eh_pred, eh_target)

        # Consistency loss: BG < 0.1 eV → should be indirect/metal (class 1)
        bg_raw = normalizer.denorm(bg_pred.squeeze())
        if log_bg:
            bg_ev = torch.expm1(bg_raw.detach())
        else:
            bg_ev = bg_raw.detach()
        mask_small = bg_ev < 0.1
        if mask_small.any() and consistency_weight > 0:
            gt_probs = gt_pred.exp()
            cons_loss = consistency_weight * gt_probs[mask_small, 1].mean()
        else:
            cons_loss = torch.tensor(0.0, device=device)

        # MTL weighting
        losses_vec = torch.stack([loss_bg, loss_gt, loss_eh])
        total_loss, task_weights = mtl_module(losses_vec)
        total_loss = total_loss + cons_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Fix 3: tighter clip
        optimizer.step()

        # Metrics
        with torch.no_grad():
            bg_denorm = normalizer.denorm(bg_pred.squeeze())
            bg_target_denorm = normalizer.denorm(bg_target.squeeze())
            if log_bg:
                bg_ev_pred = torch.expm1(bg_denorm)
                bg_ev_true = torch.expm1(bg_target_denorm)
            else:
                bg_ev_pred = bg_denorm
                bg_ev_true = bg_target_denorm
            mae = mae_metric(bg_ev_pred, bg_ev_true)

        n = batch.bg.size(0)
        losses_m.update(total_loss.item(), n)
        bg_mae_m.update(mae, n)
        gt_acc_m.update(accuracy(gt_pred, gt_target), n)
        eh_acc_m.update(accuracy(eh_pred, eh_target), n)

    return losses_m.avg, bg_mae_m.avg, gt_acc_m.avg, eh_acc_m.avg, task_weights


@torch.no_grad()
def validate(model, loader, criterion_bg, criterion_gt, criterion_eh,
             normalizer, device, log_bg=True):
    model.eval()
    losses_bg = AverageMeter()
    losses_gt = AverageMeter()
    losses_eh = AverageMeter()
    bg_mae_m = AverageMeter()
    gt_preds_all, gt_targets_all = [], []
    eh_preds_all, eh_targets_all = [], []

    for batch in loader:
        batch = batch.to(device)
        data_dict = batch.to_dict()
        bg_pred, gt_pred, eh_pred = model(data_dict)

        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).squeeze()
        eh_target = batch.eh.to(device).squeeze()

        n = batch.bg.size(0)
        losses_bg.update(criterion_bg(bg_pred.squeeze(), bg_target.squeeze()).item(), n)
        losses_gt.update(criterion_gt(gt_pred, gt_target).item(), n)
        losses_eh.update(criterion_eh(eh_pred, eh_target).item(), n)

        bg_denorm = normalizer.denorm(bg_pred.squeeze())
        bg_target_denorm = normalizer.denorm(bg_target.squeeze())
        if log_bg:
            bg_ev_pred = torch.expm1(bg_denorm)
            bg_ev_true = torch.expm1(bg_target_denorm)
        else:
            bg_ev_pred = bg_denorm
            bg_ev_true = bg_target_denorm
        bg_mae_m.update(mae_metric(bg_ev_pred, bg_ev_true), n)

        gt_preds_all.append(gt_pred.argmax(dim=1).cpu().reshape(-1))
        gt_targets_all.append(gt_target.cpu().reshape(-1))
        eh_preds_all.append(eh_pred.argmax(dim=1).cpu().reshape(-1))
        eh_targets_all.append(eh_target.cpu().reshape(-1))

    gt_preds = torch.cat(gt_preds_all).numpy()
    gt_targets = torch.cat(gt_targets_all).numpy()
    eh_preds = torch.cat(eh_preds_all).numpy()
    eh_targets = torch.cat(eh_targets_all).numpy()

    gt_acc = (gt_preds == gt_targets).mean()
    gt_f1 = f1_score(gt_targets, gt_preds, average='macro', zero_division=0)
    eh_acc = (eh_preds == eh_targets).mean()
    eh_f1 = f1_score(eh_targets, eh_preds, average='macro', zero_division=0)

    val_loss = losses_bg.avg + losses_gt.avg + losses_eh.avg

    return {
        "val_loss": val_loss,
        "bg_mae": bg_mae_m.avg,
        "gt_acc": gt_acc,
        "gt_f1": gt_f1,
        "eh_acc": eh_acc,
        "eh_f1": eh_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="MACE Multi-task Training v3")
    parser.add_argument("data_dir", help="Path to data directory with CIFs and id_prop.csv")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for MACE graphs")
    parser.add_argument("--checkpoint-dir", default="checkpoints/mace_mt", help="Checkpoint dir")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Architecture
    parser.add_argument("--h-fea-len", type=int, default=256)
    parser.add_argument("--n-attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use-cosine-classifier", action="store_true", default=True)
    parser.add_argument("--cosine-temp", type=float, default=0.1)
    parser.add_argument("--pretrained-checkpoint", default=None,
                        help="Path to pre-trained checkpoint for transfer learning")

    # Training
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4, help="Head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=5e-6, help="MACE backbone LR")
    parser.add_argument("--unfreeze-epoch", type=int, default=30,
                        help="Epoch to start unfreezing backbone (phase 1)")
    parser.add_argument("--unfreeze-epoch-2", type=int, default=50,
                        help="Epoch for second unfreeze phase (all layers)")
    parser.add_argument("--unfreeze-layers", type=int, default=1,
                        help="Number of MACE layers to unfreeze in phase 1")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--cosine-T0", type=int, default=100)
    parser.add_argument("--cosine-Tmult", type=int, default=2)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--val-ratio", type=float, default=0.15)

    # Model selection (Fix 1)
    parser.add_argument("--min-save-epoch", type=int, default=30,
                        help="Minimum epoch before model can be saved")
    parser.add_argument("--gt-composite-weight", type=float, default=0.5,
                        help="Weight for GT acc in composite metric")
    parser.add_argument("--eh-composite-weight", type=float, default=0.3,
                        help="Weight for EH acc in composite metric")

    # GT conditioning annealing (Fix 4)
    parser.add_argument("--cond-anneal-epochs", type=int, default=40,
                        help="Epochs over which GT conditioning anneals from 0 to 1")

    # Loss
    parser.add_argument("--huber-delta", type=float, default=0.4)
    parser.add_argument("--focal-gamma-gt", type=float, default=3.0)
    parser.add_argument("--focal-gamma-eh", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--gt-minority-boost", type=float, default=2.0)

    # Data
    parser.add_argument("--log-bg", action="store_true", default=True)
    parser.add_argument("--merge-metal-indirect", action="store_true", default=True)

    # MTL
    parser.add_argument("--mtl-clamp-min", type=float, default=-3.0)
    parser.add_argument("--mtl-clamp-max", type=float, default=5.0)

    # Misc
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--mace-model", default="small", choices=["small", "medium", "large"])

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MACE Multi-task v3 | seed={args.seed} | k_folds={args.k_folds} | fold={args.fold}")
    print(f"  min_save_epoch={args.min_save_epoch} gt_w={args.gt_composite_weight} "
          f"eh_w={args.eh_composite_weight} cond_anneal={args.cond_anneal_epochs}")

    # ─── Load MACE backbone ───
    print(f"Loading MACE-MP-0 ({args.mace_model})...")
    from mace.calculators import mace_mp
    calc = mace_mp(model=args.mace_model, default_dtype="float32", device=str(device))
    mace_model = calc.models[0]
    z_table = calc.z_table
    r_max = calc.r_max
    print(f"MACE-MP-0 loaded: {sum(p.numel() for p in mace_model.parameters()):,} params, r_max={r_max}")

    # ─── Dataset ───
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(args.data_dir, "mace_cached_graphs")

    dataset = MACECrystalDataset(
        root_dir=args.data_dir,
        z_table=z_table,
        r_max=r_max,
        cache_dir=cache_dir,
        merge_metal_indirect=args.merge_metal_indirect,
        log_bg=args.log_bg,
    )

    gt_dist = dataset.get_gt_class_dist()
    eh_dist = dataset.get_eh_class_dist()
    print(f"GT class dist: {dict(gt_dist)}")
    print(f"EH class dist: {dict(eh_dist)}")

    # ─── Split (Fix 5: larger val set) ───
    train_idx, val_idx, test_idx = stratified_kfold_split(
        dataset, k_folds=args.k_folds, fold=args.fold,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"\n{'='*60}")
    print(f"Fold {args.fold}/{args.k_folds-1} (seed {args.seed}) | "
          f"Train {len(train_idx)} Val {len(val_idx)} Test {len(test_idx)}")
    print(f"{'='*60}")

    # Subsets
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # ─── Normalizer ───
    train_bg = torch.tensor([dataset.samples[i]["bg"] for i in train_idx])
    normalizer = Normalizer(train_bg)
    print(f"Normalizer center={normalizer.mean:.3f} scale={normalizer.std:.3f}")

    # ─── Model ───
    model = MACEMultiTask(
        mace_model,
        h_fea_len=args.h_fea_len,
        n_gap_classes=2,
        n_eh_classes=2,
        dropout=args.dropout,
        n_attn_heads=args.n_attn_heads,
        use_cosine_classifier=args.use_cosine_classifier,
        cosine_temp=args.cosine_temp,
    ).to(device)

    # Load pre-trained weights if provided
    if args.pretrained_checkpoint:
        print(f"Loading pre-trained checkpoint: {args.pretrained_checkpoint}")
        ckpt = torch.load(args.pretrained_checkpoint, weights_only=False, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in ckpt["model_state_dict"].items()
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        loaded = len(pretrained_dict)
        total_keys = len(model_dict)
        print(f"  Loaded {loaded}/{total_keys} parameters from pre-trained model")
        if "val_metrics" in ckpt:
            pm = ckpt["val_metrics"]
            print(f"  Pre-train metrics: BG MAE={pm.get('bg_mae', '?'):.4f} "
                  f"GT={pm.get('gt_acc', '?'):.3f} EH={pm.get('eh_acc', '?'):.3f}")

    # Freeze backbone initially
    model.freeze_backbone()
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_total:,} total, {n_trainable:,} trainable (backbone frozen)")

    # ─── Loss functions ───
    n0, n1 = gt_dist.get(0, 1), gt_dist.get(1, 1)
    total_gt = n0 + n1
    w_gt = torch.tensor([total_gt / (2 * n0) * args.gt_minority_boost,
                         total_gt / (2 * n1)], device=device)
    print(f"GT class weights (boost={args.gt_minority_boost}): {w_gt.tolist()}")

    n0_eh, n1_eh = eh_dist.get(0, 1), eh_dist.get(1, 1)
    total_eh = n0_eh + n1_eh
    w_eh = torch.tensor([total_eh / (2 * n0_eh), total_eh / (2 * n1_eh)], device=device)
    print(f"EH class weights: {w_eh.tolist()}")

    criterion_bg = WeightedHuberLoss(delta=args.huber_delta)
    criterion_gt = FocalNLLLoss(gamma=args.focal_gamma_gt, weight=w_gt,
                                label_smoothing=args.label_smoothing)
    criterion_eh = FocalNLLLoss(gamma=args.focal_gamma_eh, weight=w_eh,
                                label_smoothing=args.label_smoothing)

    # ─── MTL module ───
    mtl_module = KendallMTL(n_tasks=3, clamp_min=args.mtl_clamp_min,
                            clamp_max=args.mtl_clamp_max).to(device)

    # ─── Optimizer (Fix 3: stronger weight decay) ───
    head_params = model.get_head_params()
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr, "weight_decay": 5e-4},
        {"params": mtl_module.parameters(), "lr": args.lr * 0.1},
    ])

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult)

    # Warmup
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=args.warmup_epochs
        )

    # ─── Training loop ───
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_composite = float("-inf")
    patience_counter = 0
    backbone_phase = 0  # 0=frozen, 1=last N layers, 2=all layers

    for epoch in range(1, args.epochs + 1):
        # Fix 4: GT conditioning annealing (0→1 over cond_anneal_epochs)
        if args.cond_anneal_epochs > 0:
            cw = min(1.0, epoch / args.cond_anneal_epochs)
        else:
            cw = 1.0
        model.set_cond_weight(cw)

        # Fix 2: Progressive backbone unfreezing (2-stage)
        if epoch == args.unfreeze_epoch and backbone_phase == 0:
            print(f"\n>>> Phase 1: Unfreezing backbone (last {args.unfreeze_layers} layers)...")
            if args.unfreeze_layers == 0:
                model.unfreeze_backbone()
            else:
                model.unfreeze_backbone(last_n_layers=args.unfreeze_layers)

            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Trainable params: {n_trainable:,}")

            backbone_params = [p for p in model.get_backbone_params() if p.requires_grad]
            optimizer.add_param_group({
                "params": backbone_params,
                "lr": args.backbone_lr,
                "weight_decay": 1e-5,
            })
            backbone_phase = 1

        if epoch == args.unfreeze_epoch_2 and backbone_phase == 1 and args.unfreeze_layers > 0:
            print(f"\n>>> Phase 2: Unfreezing all backbone layers...")
            model.unfreeze_backbone()  # unfreeze all

            # Collect newly unfrozen params (not already in optimizer)
            already_in_opt = set()
            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    already_in_opt.add(id(p))
            new_params = [p for p in model.get_backbone_params()
                         if p.requires_grad and id(p) not in already_in_opt]
            if new_params:
                optimizer.add_param_group({
                    "params": new_params,
                    "lr": args.backbone_lr * 0.4,  # Even lower LR for earlier layers
                    "weight_decay": 1e-5,
                })

            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Trainable params: {n_trainable:,}")
            backbone_phase = 2

        t0 = time.time()
        train_loss, train_mae, train_gt_acc, train_eh_acc, task_weights = train_epoch(
            model, train_loader, optimizer, criterion_bg, criterion_gt, criterion_eh,
            mtl_module, normalizer, args.consistency_weight, device,
            log_bg=args.log_bg, print_freq=args.print_freq,
        )

        # Scheduler step
        if epoch <= args.warmup_epochs and warmup_scheduler is not None:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # Validation
        val_metrics = validate(model, val_loader, criterion_bg, criterion_gt, criterion_eh,
                               normalizer, device, log_bg=args.log_bg)

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
        dw = task_weights.cpu().numpy()

        if epoch % args.print_freq == 0 or epoch <= 5 or epoch in (args.unfreeze_epoch, args.unfreeze_epoch_2):
            print(f"  Ep {epoch:03d} lr={lr_current:.5f} cw={cw:.2f} dw=[{dw[0]:.2f},{dw[1]:.2f},{dw[2]:.2f}] "
                  f"| TrL {train_loss:.3f} VL {val_metrics['val_loss']:.3f} "
                  f"| BG {val_metrics['bg_mae']:.4f} "
                  f"GT {val_metrics['gt_acc']:.3f}/{val_metrics['gt_f1']:.3f} "
                  f"EH {val_metrics['eh_acc']:.3f} "
                  f"| {elapsed:.1f}s")

        # Fix 1: Model selection with min_save_epoch + balanced composite
        composite = (-val_metrics["bg_mae"]
                     + args.gt_composite_weight * val_metrics["gt_acc"]
                     + args.eh_composite_weight * val_metrics["eh_acc"])

        if epoch >= args.min_save_epoch and composite > best_composite:
            best_composite = composite
            patience_counter = 0

            ckpt_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "mtl_state_dict": mtl_module.state_dict(),
                "normalizer_mean": normalizer.mean,
                "normalizer_std": normalizer.std,
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
        elif epoch >= args.min_save_epoch:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # ─── Test ───
    print(f"\n{'='*60}")
    print(f"Testing fold {args.fold} (loading best model)...")
    ckpt_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"WARNING: No checkpoint saved (min_save_epoch={args.min_save_epoch} not reached?)")
        print(f"Using final model state for testing.")
    else:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        vm = ckpt["val_metrics"]
        print(f"Best model from epoch {ckpt['epoch']} "
              f"(BG={vm['bg_mae']:.4f} GT={vm['gt_acc']:.3f} EH={vm['eh_acc']:.3f})")

    # Set full conditioning for test
    model.set_cond_weight(1.0)
    test_metrics = validate(model, test_loader, criterion_bg, criterion_gt, criterion_eh,
                            normalizer, device, log_bg=args.log_bg)

    print(f"\n  TEST RESULTS (Fold {args.fold}):")
    print(f"  BG MAE:  {test_metrics['bg_mae']:.4f} eV")
    print(f"  GT Acc:  {test_metrics['gt_acc']:.4f}  F1: {test_metrics['gt_f1']:.4f}")
    print(f"  EH Acc:  {test_metrics['eh_acc']:.4f}  F1: {test_metrics['eh_f1']:.4f}")

    # Save test results
    results_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in test_metrics.items():
            writer.writerow([k, f"{v:.6f}"])
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
