"""
Pre-train MACE multi-task model on Materials Project bandgap dataset (~50k structures).

Stage 1 of the two-stage transfer learning pipeline:
  1. Pre-train on MP (this script) → learn general structure-property relationships
  2. Fine-tune on chalcohalide (train_mace_mt.py --pretrained-checkpoint) → adapt to specific family

Architecture is identical to the fine-tuning model so weights transfer directly.
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
from sklearn.model_selection import train_test_split

from model_mace_mt import MACEMultiTask
from data_mace_mt import MACECrystalDataset

warnings.filterwarnings("ignore")


# ───────────────────── Loss functions (same as train_mace_mt.py) ─────────────────────

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
        n_classes = log_probs.size(1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.label_smoothing / (n_classes - 1))
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

    def get_task_weights(self):
        log_vars = self.log_vars.detach().clamp(self.clamp_min, self.clamp_max)
        return torch.exp(-log_vars)


class Normalizer:
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

def train_epoch(model, loader, optimizer, criterion_bg, criterion_gt, criterion_eh,
                mtl_module, normalizer, device, log_bg=True):
    model.train()
    losses_m = AverageMeter()
    bg_mae_m = AverageMeter()
    gt_acc_m = AverageMeter()
    eh_acc_m = AverageMeter()

    for batch in loader:
        batch = batch.to(device)
        data_dict = batch.to_dict()
        bg_pred, gt_pred, eh_pred = model(data_dict)

        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).squeeze()
        eh_target = batch.eh.to(device).squeeze()

        loss_bg = criterion_bg(bg_pred.squeeze(), bg_target.squeeze())
        loss_gt = criterion_gt(gt_pred, gt_target)
        loss_eh = criterion_eh(eh_pred, eh_target)

        losses_vec = torch.stack([loss_bg, loss_gt, loss_eh])
        total_loss, task_weights = mtl_module(losses_vec)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        with torch.no_grad():
            bg_denorm = normalizer.denorm(bg_pred.squeeze())
            bg_target_denorm = normalizer.denorm(bg_target.squeeze())
            if log_bg:
                mae = mae_metric(torch.expm1(bg_denorm), torch.expm1(bg_target_denorm))
            else:
                mae = mae_metric(bg_denorm, bg_target_denorm)

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

        bg_denorm = normalizer.denorm(bg_pred.squeeze())
        bg_target_denorm = normalizer.denorm(bg_target.squeeze())
        if log_bg:
            bg_mae_m.update(mae_metric(torch.expm1(bg_denorm), torch.expm1(bg_target_denorm)),
                            batch.bg.size(0))
        else:
            bg_mae_m.update(mae_metric(bg_denorm, bg_target_denorm), batch.bg.size(0))

        gt_preds_all.append(gt_pred.argmax(dim=1).cpu())
        gt_targets_all.append(gt_target.cpu())
        eh_preds_all.append(eh_pred.argmax(dim=1).cpu())
        eh_targets_all.append(eh_target.cpu())

    gt_p = torch.cat(gt_preds_all).numpy()
    gt_t = torch.cat(gt_targets_all).numpy()
    eh_p = torch.cat(eh_preds_all).numpy()
    eh_t = torch.cat(eh_targets_all).numpy()

    return {
        "bg_mae": bg_mae_m.avg,
        "gt_acc": (gt_p == gt_t).mean(),
        "gt_f1": f1_score(gt_t, gt_p, average='macro', zero_division=0),
        "eh_acc": (eh_p == eh_t).mean(),
        "eh_f1": f1_score(eh_t, eh_p, average='macro', zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Pre-train MACE on MP bandgap data")
    parser.add_argument("data_dir", help="Path to MP bandgap data directory")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints/mace_pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--h-fea-len", type=int, default=256)
    parser.add_argument("--n-attn-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--unfreeze-epoch", type=int, default=10)
    parser.add_argument("--unfreeze-layers", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--log-bg", action="store_true", default=True)
    parser.add_argument("--merge-metal-indirect", action="store_true", default=True)
    parser.add_argument("--mace-model", default="small")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--print-freq", type=int, default=5)
    parser.add_argument("--use-cosine-classifier", action="store_true", default=True)
    parser.add_argument("--cosine-temp", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load MACE
    from mace.calculators import mace_mp
    calc = mace_mp(model=args.mace_model, default_dtype="float32", device=str(device))
    mace_model = calc.models[0]
    z_table = calc.z_table
    r_max = calc.r_max
    print(f"MACE-MP-0 loaded: {sum(p.numel() for p in mace_model.parameters()):,} params")

    # Dataset
    cache_dir = args.cache_dir or os.path.join(args.data_dir, "mace_cached_graphs")
    dataset = MACECrystalDataset(
        root_dir=args.data_dir, z_table=z_table, r_max=r_max,
        cache_dir=cache_dir, merge_metal_indirect=args.merge_metal_indirect,
        log_bg=args.log_bg,
    )

    gt_dist = dataset.get_gt_class_dist()
    eh_dist = dataset.get_eh_class_dist()
    print(f"GT dist: {dict(gt_dist)}")
    print(f"EH dist: {dict(eh_dist)}")

    # Train/val split
    indices = np.arange(len(dataset))
    labels = dataset.get_stratify_labels()
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_ratio, random_state=args.seed,
        stratify=labels,
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Normalizer
    train_bg = torch.tensor([dataset.samples[i]["bg"] for i in train_idx])
    normalizer = Normalizer(train_bg)
    print(f"Normalizer center={normalizer.mean:.3f} scale={normalizer.std:.3f}")

    # Model
    model = MACEMultiTask(
        mace_model, h_fea_len=args.h_fea_len, dropout=args.dropout,
        n_attn_heads=args.n_attn_heads,
        use_cosine_classifier=args.use_cosine_classifier,
        cosine_temp=args.cosine_temp,
    ).to(device)
    model.freeze_backbone()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    # Class weights
    n0_gt, n1_gt = gt_dist.get(0, 1), gt_dist.get(1, 1)
    total_gt = n0_gt + n1_gt
    w_gt = torch.tensor([total_gt / (2 * n0_gt), total_gt / (2 * n1_gt)], device=device)
    n0_eh, n1_eh = eh_dist.get(0, 1), eh_dist.get(1, 1)
    total_eh = n0_eh + n1_eh
    w_eh = torch.tensor([total_eh / (2 * n0_eh), total_eh / (2 * n1_eh)], device=device)
    print(f"GT weights: {w_gt.tolist()}")
    print(f"EH weights: {w_eh.tolist()}")

    criterion_bg = WeightedHuberLoss(delta=0.4)
    criterion_gt = FocalNLLLoss(gamma=3.0, weight=w_gt, label_smoothing=0.05)
    criterion_eh = FocalNLLLoss(gamma=2.0, weight=w_eh, label_smoothing=0.05)
    mtl_module = KendallMTL(n_tasks=3).to(device)

    # Optimizer
    head_params = model.get_head_params()
    optimizer = torch.optim.AdamW([
        {"params": head_params, "lr": args.lr, "weight_decay": 1e-4},
        {"params": mtl_module.parameters(), "lr": args.lr * 0.1},
    ])

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_bg_mae = float("inf")
    backbone_unfrozen = False

    for epoch in range(1, args.epochs + 1):
        if epoch == args.unfreeze_epoch and not backbone_unfrozen:
            print(f"\n>>> Unfreezing backbone (last {args.unfreeze_layers} layers)...")
            if args.unfreeze_layers == 0:
                model.unfreeze_backbone()
            else:
                model.unfreeze_backbone(last_n_layers=args.unfreeze_layers)
            n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"    Trainable: {n_t:,}")
            backbone_params = [p for p in model.get_backbone_params() if p.requires_grad]
            optimizer.add_param_group({
                "params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-5,
            })
            backbone_unfrozen = True

        t0 = time.time()
        tr_loss, tr_mae, tr_gt, tr_eh, tw = train_epoch(
            model, train_loader, optimizer, criterion_bg, criterion_gt, criterion_eh,
            mtl_module, normalizer, device, log_bg=args.log_bg,
        )

        if epoch <= args.warmup_epochs and warmup_scheduler:
            warmup_scheduler.step()
        else:
            scheduler.step()

        val_m = validate(model, val_loader, criterion_bg, criterion_gt, criterion_eh,
                         normalizer, device, log_bg=args.log_bg)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        dw = tw.cpu().numpy()

        if epoch % args.print_freq == 0 or epoch <= 3 or epoch == args.unfreeze_epoch:
            print(f"  Ep {epoch:03d} lr={lr:.5f} dw=[{dw[0]:.2f},{dw[1]:.2f},{dw[2]:.2f}] "
                  f"| BG {val_m['bg_mae']:.4f} GT {val_m['gt_acc']:.3f}/{val_m['gt_f1']:.3f} "
                  f"EH {val_m['eh_acc']:.3f}/{val_m['eh_f1']:.3f} | {elapsed:.1f}s")

        if val_m["bg_mae"] < best_bg_mae:
            best_bg_mae = val_m["bg_mae"]
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "mtl_state_dict": mtl_module.state_dict(),
                "normalizer_mean": normalizer.mean,
                "normalizer_std": normalizer.std,
                "val_metrics": val_m,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "pretrained_best.pt"))

        # Also save periodic checkpoints
        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
            }, os.path.join(args.checkpoint_dir, f"pretrained_ep{epoch}.pt"))

    print(f"\nPre-training done. Best BG MAE: {best_bg_mae:.4f}")
    print(f"Checkpoint: {args.checkpoint_dir}/pretrained_best.pt")


if __name__ == "__main__":
    main()
