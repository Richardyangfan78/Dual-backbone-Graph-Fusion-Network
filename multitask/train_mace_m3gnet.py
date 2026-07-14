"""
Training script for MACE + M3GNet dual-backbone fusion (ablation).

Mirrors train_dual_backbone.py exactly — only model class and
M3GNet checkpoint loading differ.
"""
from __future__ import print_function, division

import argparse, os, sys, time, csv, math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

from model_mace_m3gnet import DualBackboneMACEM3GNet
from model_m3gnet_pyg import M3GNetMultiTaskPyG
from data_dual_backbone import DualBackboneDataset, stratified_kfold_split


# ── Reuse loss/metric helpers from train_dual_backbone ───────────────────────

class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.4):
        super().__init__()
        self.delta = delta
    def forward(self, pred, target, weights=None):
        diff = pred.squeeze() - target.squeeze()
        abs_diff = diff.abs()
        quad = torch.clamp(abs_diff, max=self.delta)
        lin  = abs_diff - quad
        loss = 0.5 * quad.pow(2) + self.delta * lin
        if weights is not None: loss = loss * weights
        return loss.mean()


class FocalNLLLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ls = label_smoothing
    def forward(self, log_probs, targets):
        n = log_probs.size(1)
        if self.ls > 0:
            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.ls / max(n - 1, 1))
                targets = targets.view(-1)
                smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
            nll = -(smooth * log_probs).sum(dim=1)
            pt  = (smooth * log_probs.exp()).sum(dim=1)
        else:
            nll = F.nll_loss(log_probs, targets, reduction='none')
            pt  = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1 - pt).pow(self.gamma) * nll
        if self.weight is not None: focal = focal * self.weight[targets]
        return focal.mean()


class KendallMTL(nn.Module):
    def __init__(self, n_tasks=3, clamp_min=-3.0, clamp_max=5.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
        self.c_min = clamp_min; self.c_max = clamp_max
    def forward(self, losses):
        lv = self.log_vars.clamp(self.c_min, self.c_max)
        prec = torch.exp(-lv)
        return (prec * losses + lv).sum(), prec.detach()


class Normalizer:
    def __init__(self, tensor=None):
        self.mean = tensor.mean().item() if tensor is not None else 0.0
        self.std  = tensor.std().item()  if tensor is not None else 1.0
    def norm(self, t):   return (t - self.mean) / (self.std + 1e-8)
    def denorm(self, t): return t * (self.std + 1e-8) + self.mean


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self, val, n=1):
        self.val=val; self.sum+=val*n; self.count+=n; self.avg=self.sum/self.count


def mae_metric(p, t): return (p - t).abs().mean().item()
def accuracy(lp, t):  return (lp.argmax(dim=1) == t).float().mean().item()


def train_epoch(model, loader, optimizer, c_bg, c_gt, c_eh, mtl, norm,
                cons_w, device, log_bg):
    model.train()
    L=AverageMeter(); B=AverageMeter(); G=AverageMeter(); E=AverageMeter()
    for batch in loader:
        batch = batch.to(device)
        bg_p, gt_p, eh_p = model(batch)
        bg_t = norm.norm(batch.bg.to(device))
        gt_t = batch.gt.to(device).squeeze()
        eh_t = batch.eh.to(device).squeeze()

        l_bg = c_bg(bg_p.squeeze(), bg_t.squeeze())
        l_gt = c_gt(gt_p, gt_t)
        l_eh = c_eh(eh_p, eh_t)

        bg_raw = norm.denorm(bg_p.squeeze())
        bg_ev  = torch.expm1(bg_raw.detach()) if log_bg else bg_raw.detach()
        cons   = cons_w * gt_p.exp()[bg_ev < 0.1, 1].mean() \
                 if (bg_ev < 0.1).any() and cons_w > 0 \
                 else torch.tensor(0.0, device=device)

        total, tw = mtl(torch.stack([l_bg, l_gt, l_eh]))
        total = total + cons
        optimizer.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        with torch.no_grad():
            bd = norm.denorm(bg_p.squeeze()); btd = norm.denorm(bg_t.squeeze())
            bev = torch.expm1(bd) if log_bg else bd
            btev= torch.expm1(btd) if log_bg else btd
        n = batch.bg.size(0)
        L.update(total.item(), n); B.update(mae_metric(bev, btev), n)
        G.update(accuracy(gt_p, gt_t), n); E.update(accuracy(eh_p, eh_t), n)
    return L.avg, B.avg, G.avg, E.avg, tw


@torch.no_grad()
def validate(model, loader, c_bg, c_gt, c_eh, norm, device, log_bg):
    model.eval()
    B=AverageMeter(); VL=AverageMeter()
    gp_all=[]; gt_all=[]; ep_all=[]; et_all=[]
    for batch in loader:
        batch = batch.to(device)
        bg_p, gt_p, eh_p = model(batch)
        bg_t = norm.norm(batch.bg.to(device))
        n = batch.bg.size(0)
        VL.update(c_bg(bg_p.squeeze(),bg_t.squeeze()).item()
                  +c_gt(gt_p,batch.gt.to(device).squeeze()).item()
                  +c_eh(eh_p,batch.eh.to(device).squeeze()).item(), n)
        bd=norm.denorm(bg_p.squeeze()); btd=norm.denorm(bg_t.squeeze())
        B.update(mae_metric(torch.expm1(bd) if log_bg else bd,
                             torch.expm1(btd) if log_bg else btd), n)
        gp_all.append(gt_p.argmax(1).cpu().reshape(-1)); gt_all.append(batch.gt.cpu().reshape(-1))
        ep_all.append(eh_p.argmax(1).cpu().reshape(-1)); et_all.append(batch.eh.cpu().reshape(-1))
    gp=torch.cat(gp_all).numpy(); gt=torch.cat(gt_all).numpy()
    ep=torch.cat(ep_all).numpy(); et=torch.cat(et_all).numpy()
    return {"val_loss": VL.avg, "bg_mae": B.avg,
            "gt_acc": (gp==gt).mean(), "gt_f1": f1_score(gt,gp,average='macro',zero_division=0),
            "eh_acc": (ep==et).mean(), "eh_f1": f1_score(et,ep,average='macro',zero_division=0)}


def load_m3gnet(ckpt_path, device, dropout=0.3):
    model = M3GNetMultiTaskPyG(
        atom_input_dim=94, edge_input_dim=80, angle_input_dim=40,
        hidden_dim=128, n_blocks=3, dropout=dropout,
    )
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(sd, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="MACE+M3GNet ablation training")
    parser.add_argument("data_dir")
    parser.add_argument("--mace-cache-dir",     default=None)
    parser.add_argument("--crystal-cache-dir",  default=None)
    parser.add_argument("--checkpoint-dir",     default="checkpoints/mace_m3gnet")
    parser.add_argument("--fold",               type=int, default=0)
    parser.add_argument("--k-folds",            type=int, default=5)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--mace-pretrained",    required=True)
    parser.add_argument("--m3gnet-checkpoint-dir", required=True)
    parser.add_argument("--h-fea-len",          type=int,   default=256)
    parser.add_argument("--n-attn-heads",       type=int,   default=8)
    parser.add_argument("--dropout",            type=float, default=0.3)
    parser.add_argument("--use-cosine-classifier", action="store_true", default=True)
    parser.add_argument("--cosine-temp",        type=float, default=0.1)
    parser.add_argument("--epochs",             type=int,   default=300)
    parser.add_argument("--batch-size",         type=int,   default=32)
    parser.add_argument("--lr",                 type=float, default=5e-4)
    parser.add_argument("--mace-backbone-lr",   type=float, default=5e-6)
    parser.add_argument("--m3gnet-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze-epoch",     type=int,   default=30)
    parser.add_argument("--unfreeze-epoch-2",   type=int,   default=50)
    parser.add_argument("--unfreeze-layers",    type=int,   default=1)
    parser.add_argument("--warmup-epochs",      type=int,   default=10)
    parser.add_argument("--cosine-T0",          type=int,   default=120)
    parser.add_argument("--cosine-Tmult",       type=int,   default=2)
    parser.add_argument("--patience",           type=int,   default=80)
    parser.add_argument("--val-ratio",          type=float, default=0.15)
    parser.add_argument("--min-save-epoch",     type=int,   default=30)
    parser.add_argument("--gt-composite-weight",type=float, default=0.5)
    parser.add_argument("--eh-composite-weight",type=float, default=0.3)
    parser.add_argument("--cond-anneal-epochs", type=int,   default=40)
    parser.add_argument("--huber-delta",        type=float, default=0.4)
    parser.add_argument("--focal-gamma-gt",     type=float, default=3.0)
    parser.add_argument("--focal-gamma-eh",     type=float, default=2.0)
    parser.add_argument("--label-smoothing",    type=float, default=0.06)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--gt-minority-boost",  type=float, default=2.0)
    parser.add_argument("--log-bg",             action="store_true", default=True)
    parser.add_argument("--merge-metal-indirect",action="store_true", default=True)
    parser.add_argument("--workers",            type=int,   default=4)
    parser.add_argument("--print-freq",         type=int,   default=10)
    parser.add_argument("--mace-model",         default="small")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"MACE+M3GNet ablation | fold={args.fold}")

    # Load MACE
    from mace.calculators import mace_mp
    calc = mace_mp(model=args.mace_model, default_dtype="float32", device=str(device))
    mace_base = calc.models[0]

    # Load M3GNet (fold-specific)
    m3g_ckpt = os.path.join(args.m3gnet_checkpoint_dir, f"fold{args.fold}_best.pt")
    print(f"Loading M3GNet: {m3g_ckpt}")
    m3gnet_model = load_m3gnet(m3g_ckpt, device, dropout=args.dropout)

    # Dataset (reuse same DualBackboneDataset — no new pipeline needed)
    mace_cache   = args.mace_cache_dir   or os.path.join(args.data_dir, "mace_cached_graphs")
    crystal_cache= args.crystal_cache_dir or os.path.join(args.data_dir, "crystal_cached_graphs")
    dataset = DualBackboneDataset(
        root_dir=args.data_dir, mace_cache_dir=mace_cache,
        crystal_cache_dir=crystal_cache,
        merge_metal_indirect=args.merge_metal_indirect, log_bg=args.log_bg,
    )
    gt_dist = dataset.get_gt_class_dist(); eh_dist = dataset.get_eh_class_dist()
    print(f"GT: {dict(gt_dist)}  EH: {dict(eh_dist)}")

    train_idx, val_idx, test_idx = stratified_kfold_split(
        dataset, k_folds=args.k_folds, fold=args.fold,
        val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"Fold {args.fold}: Train {len(train_idx)} Val {len(val_idx)} Test {len(test_idx)}")

    train_loader = DataLoader([dataset[i] for i in train_idx],
        batch_size=args.batch_size, shuffle=True,  num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader([dataset[i] for i in val_idx],
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader([dataset[i] for i in test_idx],
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    train_bg   = torch.tensor([dataset.samples[i]["bg"] for i in train_idx])
    normalizer = Normalizer(train_bg)

    # Model
    model = DualBackboneMACEM3GNet(
        mace_base, m3gnet_model, h_fea_len=args.h_fea_len,
        n_attn_heads=args.n_attn_heads, dropout=args.dropout,
        use_cosine_classifier=args.use_cosine_classifier,
        cosine_temp=args.cosine_temp,
    ).to(device)

    # Load pre-trained MACE weights
    if args.mace_pretrained:
        ckpt = torch.load(args.mace_pretrained, weights_only=False, map_location=device)
        model_dict = model.state_dict()
        src_dict   = ckpt.get("model_state_dict", ckpt)
        loaded = 0
        for k, v in src_dict.items():
            new_k = ("mace_" + k  if k.startswith("backbone.") else
                     "mace_pool." + k[len("attn_pool."):] if k.startswith("attn_pool.") else k)
            if new_k in model_dict and v.shape == model_dict[new_k].shape:
                model_dict[new_k] = v; loaded += 1
        model.load_state_dict(model_dict)
        print(f"MACE pretrained: loaded {loaded} params")

    model.freeze_mace_backbone()
    model.freeze_m3gnet_backbone()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {n_tr:,} (both frozen)")

    # Loss
    n0,n1=gt_dist.get(0,1),gt_dist.get(1,1); tot_gt=n0+n1
    w_gt=torch.tensor([tot_gt/(2*n0)*args.gt_minority_boost, tot_gt/(2*n1)], device=device)
    n0e,n1e=eh_dist.get(0,1),eh_dist.get(1,1); tot_eh=n0e+n1e
    w_eh=torch.tensor([tot_eh/(2*n0e), tot_eh/(2*n1e)], device=device)
    c_bg = WeightedHuberLoss(delta=args.huber_delta)
    c_gt = FocalNLLLoss(gamma=args.focal_gamma_gt, weight=w_gt, label_smoothing=args.label_smoothing)
    c_eh = FocalNLLLoss(gamma=args.focal_gamma_eh, weight=w_eh, label_smoothing=args.label_smoothing)
    mtl  = KendallMTL(n_tasks=3).to(device)

    head_params = model.get_head_params()
    optimizer = torch.optim.AdamW([
        {"params": head_params,       "lr": args.lr,       "weight_decay": 5e-4},
        {"params": mtl.parameters(),  "lr": args.lr*0.1},
    ])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult)
    warmup_sch = (torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs)
        if args.warmup_epochs > 0 else None)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_composite = float("-inf"); patience_counter = 0; phase = 0

    for epoch in range(1, args.epochs + 1):
        cw = min(1.0, epoch / args.cond_anneal_epochs) if args.cond_anneal_epochs > 0 else 1.0
        model.set_cond_weight(cw)

        if epoch == args.unfreeze_epoch and phase == 0:
            print(f"\n>>> Phase 1: Unfreeze MACE last {args.unfreeze_layers} + M3GNet last 2 blocks")
            model.unfreeze_mace_backbone(last_n_layers=args.unfreeze_layers)
            model.unfreeze_m3gnet_backbone(last_n_layers=2)
            new_mp=[p for p in model.get_mace_backbone_params()   if p.requires_grad]
            new_m3=[p for p in model.get_m3gnet_backbone_params() if p.requires_grad]
            if new_mp: optimizer.add_param_group({"params":new_mp,"lr":args.mace_backbone_lr,  "weight_decay":1e-5})
            if new_m3: optimizer.add_param_group({"params":new_m3,"lr":args.m3gnet_backbone_lr,"weight_decay":1e-5})
            phase = 1

        if epoch == args.unfreeze_epoch_2 and phase == 1:
            print(f"\n>>> Phase 2: Unfreeze all backbones")
            model.unfreeze_mace_backbone(); model.unfreeze_m3gnet_backbone()
            have = {id(p) for pg in optimizer.param_groups for p in pg["params"]}
            new_mp=[p for p in model.get_mace_backbone_params()   if p.requires_grad and id(p) not in have]
            new_m3=[p for p in model.get_m3gnet_backbone_params() if p.requires_grad and id(p) not in have]
            if new_mp: optimizer.add_param_group({"params":new_mp,"lr":args.mace_backbone_lr*0.4,"weight_decay":1e-5})
            if new_m3: optimizer.add_param_group({"params":new_m3,"lr":args.m3gnet_backbone_lr*0.4,"weight_decay":1e-5})
            phase = 2

        t0 = time.time()
        tr_l, tr_b, tr_g, tr_e, tw = train_epoch(
            model, train_loader, optimizer, c_bg, c_gt, c_eh,
            mtl, normalizer, args.consistency_weight, device, args.log_bg)
        if epoch <= args.warmup_epochs and warmup_sch: warmup_sch.step()
        else: scheduler.step()
        vm = validate(model, val_loader, c_bg, c_gt, c_eh, normalizer, device, args.log_bg)
        elapsed = time.time() - t0

        if epoch % args.print_freq == 0 or epoch <= 5:
            dw = tw.cpu().numpy()
            print(f"  Ep {epoch:03d} cw={cw:.2f} dw=[{dw[0]:.2f},{dw[1]:.2f},{dw[2]:.2f}]"
                  f" | TrL {tr_l:.3f} VL {vm['val_loss']:.3f}"
                  f" | BG {vm['bg_mae']:.4f} GT {vm['gt_acc']:.3f}/{vm['gt_f1']:.3f}"
                  f" EH {vm['eh_acc']:.3f} | {elapsed:.1f}s")

        composite = (-vm["bg_mae"]
                     + args.gt_composite_weight * vm["gt_acc"]
                     + args.eh_composite_weight * vm["eh_acc"])
        if epoch >= args.min_save_epoch and composite > best_composite:
            best_composite = composite; patience_counter = 0
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),
                        "optimizer_state_dict":optimizer.state_dict(),
                        "mtl_state_dict":mtl.state_dict(),
                        "normalizer_mean":normalizer.mean,"normalizer_std":normalizer.std,
                        "val_metrics":vm,"args":vars(args)},
                       os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt"))
        elif epoch >= args.min_save_epoch:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}"); break

    # Test
    ckpt_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_best.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        vm = ckpt["val_metrics"]
        print(f"\nBest ep {ckpt['epoch']}: BG={vm['bg_mae']:.4f} GT={vm['gt_acc']:.3f} EH={vm['eh_acc']:.3f}")
    model.set_cond_weight(1.0)
    test_m = validate(model, test_loader, c_bg, c_gt, c_eh, normalizer, device, args.log_bg)
    print(f"\n  TEST (Fold {args.fold}):")
    for k,v in test_m.items(): print(f"    {k}: {v:.6f}")
    results_path = os.path.join(args.checkpoint_dir, f"fold{args.fold}_results.csv")
    with open(results_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric","value"])
        for k,v in test_m.items(): w.writerow([k, f"{v:.6f}"])
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
