"""Train the PyTorch Geometric ALIGNN backbone on the aligned folds."""
from __future__ import print_function, division

import argparse
import csv
import math
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader

warnings.filterwarnings("ignore")

from data_crystal_mt import CrystalMultiTaskDataset, stratified_kfold_split
from model_alignn_pyg import ALIGNNMultiTaskPyG


# ─── Loss functions (same as train_mace_mt.py) ─────────────────────────────

class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.4):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, weights=None):
        diff    = pred.squeeze() - target.squeeze()
        abs_d   = diff.abs()
        quad    = torch.clamp(abs_d, max=self.delta)
        lin     = abs_d - quad
        loss    = 0.5 * quad.pow(2) + self.delta * lin
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
            pt  = (smooth * log_probs.exp()).sum(dim=1)
        else:
            nll = F.nll_loss(log_probs, targets, reduction="none")
            pt  = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)

        focal = (1 - pt).pow(self.gamma) * nll
        if self.weight is not None:
            focal = focal * self.weight[targets]
        return focal.mean()


class KendallMTL(nn.Module):
    def __init__(self, n_tasks=3, clamp_min=-3.0, clamp_max=5.0):
        super().__init__()
        self.log_vars  = nn.Parameter(torch.zeros(n_tasks))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, losses):
        lv = self.log_vars.clamp(self.clamp_min, self.clamp_max)
        precisions = torch.exp(-lv)
        total = (precisions * losses + lv).sum()
        return total, precisions.detach()


# ─── Normalizer ──────────────────────────────────────────────────────────────

class Normalizer:
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = tensor.mean().item()
            self.std  = tensor.std().item()
        else:
            self.mean, self.std = 0.0, 1.0

    def norm(self, t):   return (t - self.mean) / (self.std + 1e-8)
    def denorm(self, t): return t * (self.std + 1e-8) + self.mean


# ─── Metrics ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = self.count = 0
    def update(self, val, n=1): self.sum += val * n; self.count += n
    @property
    def avg(self): return self.sum / max(self.count, 1)


def mae_metric(pred, target):
    return (pred - target).abs().mean().item()


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler,
                criterion_bg, criterion_gt, criterion_eh,
                mtl, normalizer, cons_weight, device, log_bg):
    model.train()
    losses_m = AverageMeter()
    bg_mae_m = AverageMeter()
    gt_acc_m = AverageMeter()
    eh_acc_m = AverageMeter()

    for batch in loader:
        batch = batch.to(device)
        bg_pred, gt_pred, eh_pred = model(batch)

        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).long()
        eh_target = batch.eh.to(device).long()

        loss_bg = criterion_bg(bg_pred, bg_target)
        gt_known = gt_target >= 0
        if gt_known.any():
            loss_gt = criterion_gt(gt_pred[gt_known], gt_target[gt_known])
        else:
            loss_gt = torch.zeros(1, device=device, requires_grad=True).squeeze()
        loss_eh = criterion_eh(eh_pred, eh_target)

        # Consistency: small BG → should be indirect/metal
        bg_ev = torch.expm1(normalizer.denorm(bg_pred).detach()) if log_bg \
                else normalizer.denorm(bg_pred).detach()
        mask  = bg_ev < 0.1
        cons  = torch.tensor(0.0, device=device)
        if mask.any() and cons_weight > 0:
            cons = cons_weight * gt_pred.exp()[mask, 1].mean()

        losses_v = torch.stack([loss_bg, loss_gt, loss_eh])
        total, _ = mtl(losses_v)
        total = total + cons

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            bg_d = normalizer.denorm(bg_pred)
            bg_t = normalizer.denorm(bg_target)
            if log_bg:
                bg_d, bg_t = torch.expm1(bg_d), torch.expm1(bg_t)
            mae = mae_metric(bg_d, bg_t)
            n = batch.bg.size(0)
            losses_m.update(total.item(), n)
            bg_mae_m.update(mae, n)
            gt_known_cpu = gt_target >= 0
            gt_acc_val = (gt_pred.argmax(1)[gt_known_cpu] == gt_target[gt_known_cpu]).float().mean().item() if gt_known_cpu.any() else 0.0
            gt_acc_m.update(gt_acc_val, gt_known_cpu.sum().item() or 1)
            eh_acc_m.update((eh_pred.argmax(1) == eh_target).float().mean().item(), n)

    return losses_m.avg, bg_mae_m.avg, gt_acc_m.avg, eh_acc_m.avg


@torch.no_grad()
def evaluate(model, loader, criterion_bg, criterion_gt, criterion_eh,
             normalizer, device, log_bg):
    model.eval()
    bg_mae_m = AverageMeter()
    gt_preds, gt_tgts, eh_preds, eh_tgts = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        bg_pred, gt_pred, eh_pred = model(batch)

        bg_target = normalizer.norm(batch.bg.to(device))
        gt_target = batch.gt.to(device).long()
        eh_target = batch.eh.to(device).long()

        bg_d = normalizer.denorm(bg_pred)
        bg_t = normalizer.denorm(bg_target)
        if log_bg:
            bg_d, bg_t = torch.expm1(bg_d), torch.expm1(bg_t)
        bg_mae_m.update(mae_metric(bg_d, bg_t), batch.bg.size(0))

        gt_known_mask = gt_target >= 0
        if gt_known_mask.any():
            gt_preds.append(gt_pred.argmax(1)[gt_known_mask].cpu())
            gt_tgts.append(gt_target[gt_known_mask].cpu())
        eh_preds.append(eh_pred.argmax(1).cpu())
        eh_tgts.append(eh_target.cpu())

    gt_p = torch.cat(gt_preds).numpy()
    gt_t = torch.cat(gt_tgts).numpy()
    eh_p = torch.cat(eh_preds).numpy()
    eh_t = torch.cat(eh_tgts).numpy()

    return {
        "bg_mae": bg_mae_m.avg,
        "gt_acc": (gt_p == gt_t).mean(),
        "gt_f1":  f1_score(gt_t, gt_p, average="macro", zero_division=0),
        "eh_acc": (eh_p == eh_t).mean(),
        "eh_f1":  f1_score(eh_t, eh_p, average="macro", zero_division=0),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--checkpoint-dir", default="Checkpoint/ALIGNN")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Architecture
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-alignn-layers", type=int, default=4)
    parser.add_argument("--n-gcn-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Graph construction
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--max-num-nbr", type=int, default=12)
    parser.add_argument("--dist-bins", type=int, default=80)
    parser.add_argument("--angle-bins", type=int, default=40)

    # Training
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)

    # Loss
    parser.add_argument("--huber-delta", type=float, default=0.4)
    parser.add_argument("--focal-gamma-gt", type=float, default=3.0)
    parser.add_argument("--focal-gamma-eh", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.06)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--gt-minority-boost", type=float, default=2.0)
    parser.add_argument("--mtl-clamp-min", type=float, default=-3.0)
    parser.add_argument("--mtl-clamp-max", type=float, default=5.0)

    # Data
    parser.add_argument("--log-bg", action="store_true", default=True)
    parser.add_argument("--merge-metal-indirect", action="store_true", default=True)
    parser.add_argument("--eh-threshold", type=float, default=0.1)

    parser.add_argument("--print-freq", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: ALIGNN | fold={args.fold} | hidden={args.hidden_dim}")

    # ── Dataset ──────────────────────────────────────────────────────
    cache_dir = args.cache_dir or os.path.join(args.data_dir, "crystal_cached_graphs")
    dataset = CrystalMultiTaskDataset(
        root_dir=args.data_dir,
        cache_dir=cache_dir,
        radius=args.radius,
        max_num_nbr=args.max_num_nbr,
        dist_bins=args.dist_bins,
        angle_bins=args.angle_bins,
        log_bg=args.log_bg,
        merge_metal_indirect=args.merge_metal_indirect,
        eh_threshold=args.eh_threshold,
    )
    gt_dist = dataset.get_gt_dist()
    eh_dist = dataset.get_eh_dist()
    print(f"GT dist: {dict(gt_dist)}")
    print(f"EH dist: {dict(eh_dist)}")

    # ── Split ────────────────────────────────────────────────────────
    train_idx, val_idx, test_idx = stratified_kfold_split(
        dataset, k_folds=args.k_folds, fold=args.fold,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"\n{'='*60}")
    print(f"Fold {args.fold}/{args.k_folds-1} | "
          f"Train {len(train_idx)} Val {len(val_idx)} Test {len(test_idx)}")
    print(f"{'='*60}")

    train_set = [dataset[i] for i in train_idx]
    val_set   = [dataset[i] for i in val_idx]
    test_set  = [dataset[i] for i in test_idx]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              follow_batch=["x"])
    val_loader   = DataLoader(val_set,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, follow_batch=["x"])
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, follow_batch=["x"])

    # ── Normalizer ───────────────────────────────────────────────────
    train_bg = torch.tensor([dataset.samples[i]["bg"] for i in train_idx])
    normalizer = Normalizer(train_bg)
    print(f"Normalizer center={normalizer.mean:.3f} scale={normalizer.std:.3f}")

    # ── Model ────────────────────────────────────────────────────────
    model = ALIGNNMultiTaskPyG(
        atom_input_dim=94,
        edge_input_dim=args.dist_bins,
        angle_input_dim=args.angle_bins,
        hidden_dim=args.hidden_dim,
        n_alignn_layers=args.n_alignn_layers,
        n_gcn_layers=args.n_gcn_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Loss ─────────────────────────────────────────────────────────
    n0_gt, n1_gt = gt_dist.get(0, 1), gt_dist.get(1, 1)
    tot_gt = n0_gt + n1_gt
    w_gt = torch.tensor([
        tot_gt / (2 * n0_gt) * args.gt_minority_boost,
        tot_gt / (2 * n1_gt),
    ], device=device)
    print(f"GT class weights (boost={args.gt_minority_boost}): {w_gt.tolist()}")

    n0_eh, n1_eh = eh_dist.get(0, 1), eh_dist.get(1, 1)
    tot_eh = n0_eh + n1_eh
    w_eh = torch.tensor([tot_eh / (2 * n0_eh), tot_eh / (2 * n1_eh)], device=device)

    criterion_bg = WeightedHuberLoss(delta=args.huber_delta)
    criterion_gt = FocalNLLLoss(args.focal_gamma_gt, w_gt, args.label_smoothing)
    criterion_eh = FocalNLLLoss(args.focal_gamma_eh, w_eh, args.label_smoothing)
    mtl = KendallMTL(3, args.mtl_clamp_min, args.mtl_clamp_max).to(device)

    # ── Optimizer + Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mtl.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos",
    )

    # ── Training loop ────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_bg_mae   = float("inf")
    best_composite = float("-inf")
    patience_counter = 0
    ckpt_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae, tr_gt, tr_eh = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_bg, criterion_gt, criterion_eh,
            mtl, normalizer, args.consistency_weight, device, args.log_bg,
        )
        val = evaluate(model, val_loader, criterion_bg, criterion_gt, criterion_eh,
                       normalizer, device, args.log_bg)
        elapsed = time.time() - t0

        if epoch % args.print_freq == 0 or epoch <= 5:
            lr = optimizer.param_groups[0]["lr"]
            dw = mtl.log_vars.detach().clamp(args.mtl_clamp_min, args.mtl_clamp_max)
            dw = torch.exp(-dw).cpu().numpy()
            print(f"  Ep {epoch:03d} lr={lr:.5f} dw=[{dw[0]:.2f},{dw[1]:.2f},{dw[2]:.2f}] "
                  f"| TrL {tr_loss:.3f} "
                  f"| BG {val['bg_mae']:.4f} "
                  f"GT {val['gt_acc']:.3f}/{val['gt_f1']:.3f} "
                  f"EH {val['eh_acc']:.3f} "
                  f"| {elapsed:.1f}s")

        composite = -val["bg_mae"] + 0.3 * (val["gt_acc"] + val["eh_acc"])
        improved = (val["bg_mae"] < best_bg_mae) or (composite > best_composite)
        if improved:
            best_bg_mae     = min(best_bg_mae, val["bg_mae"])
            best_composite  = max(best_composite, composite)
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "mtl_state_dict": mtl.state_dict(),
                "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std,
                "val_metrics": val, "args": vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Test ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Testing fold {args.fold} (loading best model)...")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Best model from epoch {ckpt['epoch']} "
          f"(val BG MAE={ckpt['val_metrics']['bg_mae']:.4f})")

    test_m = evaluate(model, test_loader, criterion_bg, criterion_gt, criterion_eh,
                      normalizer, device, args.log_bg)

    print(f"\n  TEST RESULTS (Fold {args.fold}):")
    print(f"  BG MAE:  {test_m['bg_mae']:.4f} eV")
    print(f"  GT Acc:  {test_m['gt_acc']:.4f}  F1: {test_m['gt_f1']:.4f}")
    print(f"  EH Acc:  {test_m['eh_acc']:.4f}  F1: {test_m['eh_f1']:.4f}")

    results_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_results.csv")
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in test_m.items():
            w.writerow([k, f"{v:.6f}"])
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
