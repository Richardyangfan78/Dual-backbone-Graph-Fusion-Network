#!/usr/bin/env python3
"""
Generate OOF predictions and plots for MACE+ALIGNN and MACE+M3GNet only.
(CGCNN, ALIGNN, M3GNet, MACE plots already generated.)
"""
import os, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, r2_score
import torch

PROJECT   = "/path/to/Dual-backbone-Graph-Fusion-Network"
MULTITASK = f"{PROJECT}/multitask"
CKPT_BASE = f"{PROJECT}/checkpoints"
OUT_DIR   = f"{PROJECT}/results/model_plots"
sys.path.insert(0, MULTITASK)
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class Normalizer:
    def __init__(self):
        self.mean = 0.0
        self.std  = 1.0
    def denorm(self, t):
        return t * (self.std + 1e-8) + self.mean


FOLD_COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2']

def plot_bg_scatter(bg_true, bg_pred, fold_arr, model_name):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for fi in range(5):
        mask = fold_arr == fi
        ax.scatter(bg_true[mask], bg_pred[mask],
                   s=10, alpha=0.45, color=FOLD_COLORS[fi],
                   label=f'Fold {fi}', linewidths=0)
    lim_lo = max(0, min(bg_true.min(), bg_pred.min()) - 0.1)
    lim_hi = max(bg_true.max(), bg_pred.max()) + 0.2
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', lw=1.2, label='Ideal')
    mae = np.mean(np.abs(bg_true - bg_pred))
    r2  = r2_score(bg_true, bg_pred)
    ax.text(0.04, 0.95, f'MAE = {mae:.3f} eV\n$R^2$ = {r2:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    ax.set_xlabel('DFT Bandgap (eV)', fontsize=11)
    ax.set_ylabel('Predicted Bandgap (eV)', fontsize=11)
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_title(f'{model_name} — Bandgap Fitting', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, markerscale=1.5, loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    path = f"{OUT_DIR}/{model_name}_bg_scatter.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")


def plot_clf_bars(true_arr, pred_arr, fold_arr, model_name, task):
    accs, f1s = [], []
    for fi in range(5):
        mask = fold_arr == fi
        accs.append(accuracy_score(true_arr[mask], pred_arr[mask]))
        f1s.append(f1_score(true_arr[mask], pred_arr[mask], average='macro', zero_division=0))
    x = np.arange(5); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    bars_acc = ax.bar(x - w/2, accs, w, label='Accuracy',   color='#4C72B0', alpha=0.85)
    bars_f1  = ax.bar(x + w/2, f1s,  w, label='F1 (macro)', color='#DD8452', alpha=0.85)
    mean_acc = np.mean(accs); mean_f1 = np.mean(f1s)
    ax.axhline(mean_acc, color='#4C72B0', lw=1.5, ls='--', alpha=0.7,
               label=f'Mean Acc={mean_acc:.3f}')
    ax.axhline(mean_f1,  color='#DD8452', lw=1.5, ls='--', alpha=0.7,
               label=f'Mean F1={mean_f1:.3f}')
    for bar in list(bars_acc) + list(bars_f1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f'Fold {i}' for i in range(5)])
    ax.set_ylim(0, 1.08); ax.set_ylabel('Score', fontsize=11)
    task_label = 'Gap Type (GT)' if task == 'gt' else 'E-hull (EH)'
    ax.set_title(f'{model_name} — {task_label} Accuracy & F1',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    path = f"{OUT_DIR}/{model_name}_{task}_bars.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  Saved {path}")


def make_all_plots(model_name, data):
    print(f"\n--- Plotting {model_name} ---")
    plot_bg_scatter(data['bg_true'], data['bg_pred'], data['fold'], model_name)
    plot_clf_bars(data['gt_true'], data['gt_pred'], data['fold'], model_name, 'gt')
    plot_clf_bars(data['eh_true'], data['eh_pred'], data['fold'], model_name, 'eh')


# ──────────────────────────────────────────────
# Load MACE foundation model (needed for both dual-backbone models)
# ──────────────────────────────────────────────
from mace.calculators import mace_mp

print("\nLoading MACE-MP-0 small...")
calc      = mace_mp(model='small', default_dtype='float32', device=str(device))
mace_base = calc.models[0]
z_table   = calc.z_table
r_max     = calc.r_max
print(f"MACE loaded (r_max={r_max})")


# ──────────────────────────────────────────────
# MACE+ALIGNN (dual_backbone)
# ──────────────────────────────────────────────
def run_mace_alignn():
    from train_dual_backbone import load_alignn_model
    from model_dual_backbone import DualBackboneMultiTask
    from data_dual_backbone import DualBackboneDataset, stratified_kfold_split
    from torch_geometric.loader import DataLoader as PygLoader

    results = {k: [] for k in ['bg_true','bg_pred','gt_true','gt_pred','eh_true','eh_pred','fold']}

    for fold in range(5):
        ckpt = torch.load(f"{CKPT_BASE}/dual_backbone/fold{fold}_best.pt",
                          map_location=device, weights_only=False)
        args = ckpt['args']

        alignn_ckpt  = os.path.join(args['alignn_checkpoint_dir'], f"fold{fold}_best.pt")
        alignn_model = load_alignn_model(alignn_ckpt, device, dropout=args['dropout'])

        # FIX: correct arg order — root_dir first, then mace_cache_dir, crystal_cache_dir
        dataset = DualBackboneDataset(
            args['data_dir'], args['mace_cache_dir'], args['crystal_cache_dir'],
            merge_metal_indirect=args['merge_metal_indirect'],
            log_bg=args['log_bg'],
        )
        _, _, test_idx = stratified_kfold_split(
            dataset, args['k_folds'], fold, args['val_ratio'], args['seed'])

        loader = PygLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)

        model = DualBackboneMultiTask(
            mace_base, alignn_model,
            h_fea_len=args['h_fea_len'],
            n_gap_classes=2,
            dropout=args['dropout'],
            n_attn_heads=args['n_attn_heads'],
            use_cosine_classifier=args['use_cosine_classifier'],
            cosine_temp=args['cosine_temp'],
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()

        norm = Normalizer()
        norm.mean = float(ckpt['normalizer_mean'])
        norm.std  = float(ckpt['normalizer_std'])

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_p, gt_l, eh_l = model(batch)
                bg_p = torch.expm1(norm.denorm(bg_p.squeeze().cpu().float())).numpy()
                bg_t = torch.expm1(batch.bg.cpu().float()).numpy()
                results['bg_true'].extend(bg_t.tolist())
                results['bg_pred'].extend(bg_p.tolist())
                results['gt_true'].extend(batch.gt.cpu().numpy().tolist())
                results['gt_pred'].extend(gt_l.argmax(1).cpu().numpy().tolist())
                results['eh_true'].extend(batch.eh.cpu().numpy().tolist())
                results['eh_pred'].extend(eh_l.argmax(1).cpu().numpy().tolist())
                results['fold'].extend([fold] * int(batch.bg.size(0)))
        print(f"  MACE+ALIGNN fold {fold}: {len(test_idx)} samples")

    return {k: np.array(v) for k, v in results.items()}

print("\n=== MACE+ALIGNN ===")
make_all_plots('MACE+ALIGNN', run_mace_alignn())


# ──────────────────────────────────────────────
# MACE+M3GNet (mace_m3gnet)
# ──────────────────────────────────────────────
def run_mace_m3gnet():
    from train_mace_m3gnet import load_m3gnet
    from model_mace_m3gnet import DualBackboneMACEM3GNet
    from data_dual_backbone import DualBackboneDataset, stratified_kfold_split
    from torch_geometric.loader import DataLoader as PygLoader

    results = {k: [] for k in ['bg_true','bg_pred','gt_true','gt_pred','eh_true','eh_pred','fold']}

    for fold in range(5):
        ckpt = torch.load(f"{CKPT_BASE}/mace_m3gnet/fold{fold}_best.pt",
                          map_location=device, weights_only=False)
        args = ckpt['args']

        m3g_ckpt     = os.path.join(args['m3gnet_checkpoint_dir'], f"fold{fold}_best.pt")
        m3gnet_model = load_m3gnet(m3g_ckpt, device, dropout=args['dropout'])

        # FIX: correct arg order — root_dir first, then mace_cache_dir, crystal_cache_dir
        dataset = DualBackboneDataset(
            args['data_dir'], args['mace_cache_dir'], args['crystal_cache_dir'],
            merge_metal_indirect=args['merge_metal_indirect'],
            log_bg=args['log_bg'],
        )
        _, _, test_idx = stratified_kfold_split(
            dataset, args['k_folds'], fold, args['val_ratio'], args['seed'])

        loader = PygLoader([dataset[i] for i in test_idx], batch_size=32, shuffle=False)

        model = DualBackboneMACEM3GNet(
            mace_base, m3gnet_model,
            h_fea_len=args['h_fea_len'],
            n_gap_classes=2,
            dropout=args['dropout'],
            n_attn_heads=args['n_attn_heads'],
            use_cosine_classifier=args['use_cosine_classifier'],
            cosine_temp=args['cosine_temp'],
        )
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device).eval()

        norm = Normalizer()
        norm.mean = float(ckpt['normalizer_mean'])
        norm.std  = float(ckpt['normalizer_std'])

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_p, gt_l, eh_l = model(batch)
                bg_p = torch.expm1(norm.denorm(bg_p.squeeze().cpu().float())).numpy()
                bg_t = torch.expm1(batch.bg.cpu().float()).numpy()
                results['bg_true'].extend(bg_t.tolist())
                results['bg_pred'].extend(bg_p.tolist())
                results['gt_true'].extend(batch.gt.cpu().numpy().tolist())
                results['gt_pred'].extend(gt_l.argmax(1).cpu().numpy().tolist())
                results['eh_true'].extend(batch.eh.cpu().numpy().tolist())
                results['eh_pred'].extend(eh_l.argmax(1).cpu().numpy().tolist())
                results['fold'].extend([fold] * int(batch.bg.size(0)))
        print(f"  MACE+M3GNet fold {fold}: {len(test_idx)} samples")

    return {k: np.array(v) for k, v in results.items()}

print("\n=== MACE+M3GNet ===")
make_all_plots('MACE+M3GNet', run_mace_m3gnet())

print(f"\nAll dual-backbone plots saved to {OUT_DIR}")
