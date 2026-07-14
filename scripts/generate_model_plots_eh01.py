#!/usr/bin/env python3
"""
Generate per-model OOF predictions and 3 plots for all 6 eh01 models:
  1. Bandgap scatter (predicted vs. true, coloured by fold)
  2. Gap Type Accuracy & F1 per fold bar chart
  3. E-hull Stability Accuracy & F1 per fold bar chart

Models: CGCNN, MACE, ALIGNN, M3GNet, MACE+M3GNet, MACE+ALIGNN  (all eh01)
"""
import os, sys, warnings
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, r2_score

warnings.filterwarnings("ignore")

PROJECT   = "/path/to/Dual-backbone-Graph-Fusion-Network"
MULTITASK = f"{PROJECT}/multitask"
CKPT_BASE = f"{PROJECT}/checkpoints"
OUT_DIR   = f"{PROJECT}/results/model_plots_eh01"
sys.path.insert(0, MULTITASK)
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_FOLDS = 5
FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

MODEL_COLORS = {
    "CGCNN":       "#B09C85",
    "MACE":        "#4DBBD5",
    "ALIGNN":      "#00A087",
    "M3GNet":      "#E64B35",
    "MACE+M3GNet": "#3C5488",
    "MACE+ALIGNN": "#DC0000",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
})


# ── Normalizer ────────────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean; self.std = std
    def denorm(self, t):
        return t * (self.std + 1e-8) + self.mean


# ── Plot helpers ──────────────────────────────────────────────────────────────
def plot_bg_scatter(bg_true, bg_pred, fold_arr, model_name, color):
    fig, ax = plt.subplots(figsize=(5, 5))
    for fi in range(N_FOLDS):
        mask = fold_arr == fi
        ax.scatter(bg_true[mask], bg_pred[mask], s=8, alpha=0.40,
                   color=FOLD_COLORS[fi], label=f"Fold {fi}", linewidths=0, zorder=3)
    lo = max(0, min(bg_true.min(), bg_pred.min()) - 0.15)
    hi = max(bg_true.max(), bg_pred.max()) + 0.25
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6, label="Ideal", zorder=2)
    mae = float(np.mean(np.abs(bg_true - bg_pred)))
    r2  = float(r2_score(bg_true, bg_pred))
    ax.text(0.05, 0.95,
            f"MAE = {mae:.3f} eV\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", alpha=0.9))
    ax.set_xlabel("DFT Bandgap (eV)", fontsize=11)
    ax.set_ylabel("Predicted Bandgap (eV)", fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(f"{model_name}  —  Bandgap Fitting", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, markerscale=1.8, loc="lower right", framealpha=0.85)
    ax.set_aspect("equal")
    plt.tight_layout()
    path = f"{OUT_DIR}/{model_name}_bg_scatter.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


def plot_clf_bars(true_arr, pred_arr, fold_arr, model_name, task, color):
    accs, f1s = [], []
    for fi in range(N_FOLDS):
        mask = fold_arr == fi
        if mask.sum() == 0:
            accs.append(float("nan")); f1s.append(float("nan")); continue
        accs.append(accuracy_score(true_arr[mask], pred_arr[mask]))
        f1s.append(f1_score(true_arr[mask], pred_arr[mask], average="macro", zero_division=0))

    x = np.arange(N_FOLDS); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))

    valid_acc = [a for a in accs if not np.isnan(a)]
    valid_f1  = [f for f in f1s  if not np.isnan(f)]

    b_acc = ax.bar(x - w/2, accs, w, label="Accuracy",   color=color,    alpha=0.85, edgecolor="white")
    b_f1  = ax.bar(x + w/2, f1s,  w, label="F1 (macro)", color=color,    alpha=0.45, edgecolor="white", hatch="///")

    if valid_acc:
        mean_acc = np.mean(valid_acc); mean_f1 = np.mean(valid_f1)
        ax.axhline(mean_acc, color=color, lw=1.8, ls="--", alpha=0.9,
                   label=f"Mean Acc = {mean_acc:.3f}")
        ax.axhline(mean_f1,  color=color, lw=1.8, ls=":",  alpha=0.9,
                   label=f"Mean F1  = {mean_f1:.3f}")

    for bar in list(b_acc) + list(b_f1):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.006,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i}" for i in range(N_FOLDS)])
    ax.set_ylim(0, 1.10); ax.set_ylabel("Score", fontsize=11)
    task_label = "Gap Type (GT)" if task == "gt" else "E-hull Stability (EH)"
    ax.set_title(f"{model_name}  —  {task_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.88)
    ax.grid(axis="y", ls="--", lw=0.5, alpha=0.3)
    plt.tight_layout()
    path = f"{OUT_DIR}/{model_name}_{task}_bars.png"
    plt.savefig(path, bbox_inches="tight"); plt.close()
    print(f"  Saved {path}")


def make_all_plots(model_name, data):
    print(f"\n--- Plotting {model_name} ---")
    color = MODEL_COLORS[model_name]
    plot_bg_scatter(data["bg_true"], data["bg_pred"], data["fold"], model_name, color)
    plot_clf_bars(data["gt_true"], data["gt_pred"], data["fold"], model_name, "gt", color)
    if data["eh_true"] is not None:
        plot_clf_bars(data["eh_true"], data["eh_pred"], data["fold"], model_name, "eh", color)


def collect(d, bg_t, bg_p, gt_t, gt_p, eh_t, eh_p, fi):
    d["bg_true"].extend(bg_t); d["bg_pred"].extend(bg_p)
    d["gt_true"].extend(gt_t); d["gt_pred"].extend(gt_p)
    d["eh_true"].extend(eh_t); d["eh_pred"].extend(eh_p)
    d["fold"].extend([fi] * len(bg_t))


def to_np(d):
    return {k: np.array(v) for k, v in d.items()}


def empty_results():
    return {k: [] for k in ["bg_true","bg_pred","gt_true","gt_pred","eh_true","eh_pred","fold"]}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CGCNN  (train_mt_v10 / CachedGraphDatasetV7 / stratified_kfold_v4)
# ═══════════════════════════════════════════════════════════════════════════════
def run_cgcnn():
    import csv as _csv, random as _random
    from train_mt_v10 import CachedGraphDatasetV7, collate_pool_mt_v7, Normalizer
    from data_mt_v4   import stratified_kfold_v4
    from model_mt_v10 import CrystalGraphConvNetMTV10
    from torch.utils.data import DataLoader, SubsetRandomSampler

    CKPT_DIR  = f"{CKPT_BASE}/cgcnn_mt_eh01"
    DATA_DIR  = f"{PROJECT}/Data/multitask"
    CACHE_DIR = f"{PROJECT}/Data/multitask/cached_graphs"
    res = empty_results()

    # Load id_prop.csv exactly as training did
    with open(f"{DATA_DIR}/id_prop.csv") as f:
        id_prop_data = list(_csv.reader(f))
    _random.seed(123)
    _random.shuffle(id_prop_data)

    dataset = CachedGraphDatasetV7(CACHE_DIR, id_prop_data, merge_metal_indirect=True)
    print(f"  CGCNN dataset: {len(dataset)} samples")

    # Get feature dims from first sample
    (s_af, s_nf, _), _, _ = dataset[0]

    # Reproduce same fold splits
    ck0 = torch.load(f"{CKPT_DIR}/fold_0_seed_42/best_composite.pth.tar",
                     map_location="cpu", weights_only=False)
    args = ck0["args"]
    folds = stratified_kfold_v4(
        dataset.id_prop_data, n_folds=args["k_folds"],
        merge_metal_indirect=args.get("merge_metal_indirect", True))

    for fold in range(args["k_folds"]):
        test_idx = folds[fold]
        ck = torch.load(f"{CKPT_DIR}/fold_{fold}_seed_42/best_composite.pth.tar",
                        map_location=device, weights_only=False)

        normalizer = Normalizer()
        normalizer.load_state_dict(ck["normalizer_bg"])

        model = CrystalGraphConvNetMTV10(
            s_af.shape[-1], s_nf.shape[-1],
            atom_fea_len=args["atom_fea_len"],
            n_conv=args["n_conv"],
            h_fea_len=args["h_fea_len"],
            n_h=args.get("n_h", 1),
            n_gap_classes=2,
            n_eh_classes=2,
            dropout=args["dropout"],
            n_attn_heads=args["n_attn_heads"],
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        )
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()

        loader = DataLoader(
            dataset, sampler=SubsetRandomSampler(test_idx),
            batch_size=64, collate_fn=collate_pool_mt_v7,
            num_workers=0, pin_memory=False,
        )

        bg_t, bg_p, gt_t, gt_p, eh_t, eh_p = [], [], [], [], [], []
        with torch.no_grad():
            for (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), targets, _ in loader:
                atom_fea        = atom_fea.to(device)
                nbr_fea         = nbr_fea.to(device)
                nbr_fea_idx     = nbr_fea_idx.to(device)
                crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

                p_bg, p_gt, p_eh = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

                bg_pred = normalizer.denorm(p_bg.squeeze().cpu().float())
                bg_true = targets["bandgap"].squeeze().float()
                bg_p.extend(bg_pred.numpy().tolist())
                bg_t.extend(bg_true.numpy().tolist())
                gt_p.extend(p_gt.argmax(1).cpu().numpy().tolist())
                gt_t.extend(targets["gaptype"].squeeze(-1).numpy().tolist())
                eh_p.extend(p_eh.argmax(1).cpu().numpy().tolist())
                eh_t.extend(targets["eh_label"].squeeze(-1).numpy().tolist())

        collect(res, bg_t, bg_p, gt_t, gt_p, eh_t, eh_p, fold)
        print(f"  CGCNN fold {fold}: {len(test_idx)} samples")

    return to_np(res)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ALIGNN / M3GNet  (train_crystal_mt / CrystalMultiTaskDataset)
# ═══════════════════════════════════════════════════════════════════════════════
def run_crystal_mt(model_key, model_class):
    """Shared runner for ALIGNN and M3GNet (both use CrystalMultiTaskDataset)."""
    from data_crystal_mt import CrystalMultiTaskDataset, stratified_kfold_split
    from torch_geometric.loader import DataLoader as PygLoader

    CKPT_DIR = f"{CKPT_BASE}/{model_key}_mt_eh01"
    res = empty_results()

    ck0   = torch.load(f"{CKPT_DIR}/fold0_best.pt", map_location="cpu", weights_only=False)
    args  = ck0["args"]

    dataset = CrystalMultiTaskDataset(
        root_dir=args["data_dir"],
        cache_dir=args["cache_dir"],
        merge_metal_indirect=args["merge_metal_indirect"],
        log_bg=args["log_bg"],
        eh_threshold=args.get("eh_threshold", 0.1),
        radius=args.get("radius", 8.0),
        max_num_nbr=args.get("max_num_nbr", 12),
        dist_bins=args.get("dist_bins", 80),
        angle_bins=args.get("angle_bins", 40),
    )
    print(f"  {model_key} dataset: {len(dataset)} samples")

    for fold in range(N_FOLDS):
        ck   = torch.load(f"{CKPT_DIR}/fold{fold}_best.pt",
                          map_location=device, weights_only=False)
        args = ck["args"]
        nm, ns = float(ck["normalizer_mean"]), float(ck["normalizer_std"])

        _, _, test_idx = stratified_kfold_split(
            dataset, k_folds=args["k_folds"], fold=fold,
            val_ratio=args.get("val_ratio", 0.1), seed=args["seed"])

        sample = dataset[0]
        edge_input_dim  = sample.edge_attr.shape[1]
        angle_input_dim = sample.line_attr.shape[1]

        model = model_class(
            hidden_dim=args.get("hidden_dim", 256),
            n_gap_classes=2,
            n_eh_classes=2,
            dropout=args.get("dropout", 0.3),
            edge_input_dim=edge_input_dim,
            angle_input_dim=angle_input_dim,
            **({} if model_key == "alignn" else {"n_blocks": args.get("n_blocks", 3)}),
            **({
                "n_alignn_layers": args.get("n_alignn_layers", 4),
                "n_gcn_layers":    args.get("n_gcn_layers",    4),
            } if model_key == "alignn" else {}),
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()

        loader = PygLoader([dataset[i] for i in test_idx],
                           batch_size=32, shuffle=False, num_workers=0)
        bg_t, bg_p, gt_t, gt_p, eh_t, eh_p = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_out, gt_out, eh_out = model(batch)
                bg_pred = torch.expm1(bg_out.squeeze().cpu().float() * (ns + 1e-8) + nm)
                bg_true = torch.expm1(batch.bg.cpu().float())
                bg_p.extend(bg_pred.numpy().tolist())
                bg_t.extend(bg_true.numpy().tolist())
                gt_p.extend(gt_out.argmax(1).cpu().numpy().tolist())
                gt_t.extend(batch.gt.cpu().numpy().tolist())
                eh_p.extend(eh_out.argmax(1).cpu().numpy().tolist())
                eh_t.extend(batch.eh.cpu().numpy().tolist())
        collect(res, bg_t, bg_p, gt_t, gt_p, eh_t, eh_p, fold)
        print(f"  {model_key} fold {fold}: {len(test_idx)} samples")

    return to_np(res)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MACE  (train_mace_mt / MACECrystalDataset)
# ═══════════════════════════════════════════════════════════════════════════════
def run_mace(mace_base, z_table, r_max):
    from data_mace_mt import MACECrystalDataset, stratified_kfold_split
    from model_mace_mt import MACEMultiTask
    from torch_geometric.loader import DataLoader as PygLoader

    CKPT_DIR = f"{CKPT_BASE}/mace_mt_eh01"
    res = empty_results()

    ck0  = torch.load(f"{CKPT_DIR}/fold0_best.pt", map_location="cpu", weights_only=False)
    args = ck0["args"]

    dataset = MACECrystalDataset(
        args["data_dir"], z_table, r_max,
        cache_dir=args["cache_dir"],
        merge_metal_indirect=args["merge_metal_indirect"],
        log_bg=args["log_bg"],
        eh_threshold=args.get("eh_threshold", 0.1),
    )
    print(f"  MACE dataset: {len(dataset)} samples")

    for fold in range(N_FOLDS):
        ck   = torch.load(f"{CKPT_DIR}/fold{fold}_best.pt",
                          map_location=device, weights_only=False)
        args = ck["args"]
        nm, ns = float(ck["normalizer_mean"]), float(ck["normalizer_std"])

        _, _, test_idx = stratified_kfold_split(
            dataset, k_folds=args["k_folds"], fold=fold,
            val_ratio=args.get("val_ratio", 0.1), seed=args["seed"])

        model = MACEMultiTask(
            mace_base,
            h_fea_len=args["h_fea_len"],
            n_gap_classes=2,
            n_eh_classes=2,
            dropout=args["dropout"],
            n_attn_heads=args["n_attn_heads"],
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()

        loader = PygLoader([dataset[i] for i in test_idx],
                           batch_size=32, shuffle=False, num_workers=0)
        bg_t, bg_p, gt_t, gt_p, eh_t, eh_p = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_out, gt_out, eh_out = model(batch.to_dict())
                bg_pred = torch.expm1(bg_out.squeeze().cpu().float() * (ns + 1e-8) + nm)
                bg_true = torch.expm1(batch.bg.cpu().float())
                bg_p.extend(bg_pred.numpy().tolist())
                bg_t.extend(bg_true.numpy().tolist())
                gt_p.extend(gt_out.argmax(1).cpu().numpy().tolist())
                gt_t.extend(batch.gt.cpu().numpy().tolist())
                eh_p.extend(eh_out.argmax(1).cpu().numpy().tolist())
                eh_t.extend(batch.eh.cpu().numpy().tolist())
        collect(res, bg_t, bg_p, gt_t, gt_p, eh_t, eh_p, fold)
        print(f"  MACE fold {fold}: {len(test_idx)} samples")

    return to_np(res)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MACE+ALIGNN  (train_dual_backbone / DualBackboneDataset)
# ═══════════════════════════════════════════════════════════════════════════════
def run_dual_backbone(mace_base):
    from data_dual_backbone import DualBackboneDataset, stratified_kfold_split
    from model_dual_backbone import DualBackboneMultiTask
    from model_alignn_pyg    import ALIGNNMultiTaskPyG
    from torch_geometric.loader import DataLoader as PygLoader

    CKPT_DIR       = f"{CKPT_BASE}/dual_backbone_eh01"
    ALIGNN_CKPT_DIR= f"{CKPT_BASE}/alignn_mt_eh01"
    res = empty_results()

    ck0  = torch.load(f"{CKPT_DIR}/fold0_best.pt", map_location="cpu", weights_only=False)
    args = ck0["args"]

    dataset = DualBackboneDataset(
        mace_cache_dir=args["mace_cache_dir"],
        crystal_cache_dir=args["crystal_cache_dir"],
        root_dir=args["data_dir"],
        merge_metal_indirect=args["merge_metal_indirect"],
        log_bg=args["log_bg"],
        eh_threshold=args.get("eh_threshold", 0.1),
    )
    print(f"  MACE+ALIGNN dataset: {len(dataset)} samples")

    sample = dataset[0]
    edge_dim  = sample.alignn_edge_attr.shape[1]
    angle_dim = sample.alignn_line_attr.shape[1]

    for fold in range(N_FOLDS):
        ck   = torch.load(f"{CKPT_DIR}/fold{fold}_best.pt",
                          map_location=device, weights_only=False)
        args = ck["args"]
        nm, ns = float(ck["normalizer_mean"]), float(ck["normalizer_std"])

        _, _, test_idx = stratified_kfold_split(
            dataset, k_folds=args["k_folds"], fold=fold,
            val_ratio=args.get("val_ratio", 0.15), seed=args["seed"])

        alignn_ck  = torch.load(f"{ALIGNN_CKPT_DIR}/fold{fold}_best.pt",
                                map_location=device, weights_only=False)
        alignn_model = ALIGNNMultiTaskPyG(
            hidden_dim=256, n_alignn_layers=4, n_gcn_layers=4,
            dropout=args["dropout"], edge_input_dim=edge_dim,
            angle_input_dim=angle_dim, n_gap_classes=2, n_eh_classes=2)
        alignn_model.load_state_dict(alignn_ck["model_state_dict"])

        model = DualBackboneMultiTask(
            mace_base, alignn_model,
            h_fea_len=args["h_fea_len"], n_attn_heads=args["n_attn_heads"],
            dropout=args["dropout"], n_gap_classes=2, n_eh_classes=2,
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()

        loader = PygLoader([dataset[i] for i in test_idx],
                           batch_size=16, shuffle=False, num_workers=0)
        bg_t, bg_p, gt_t, gt_p, eh_t, eh_p = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_out, gt_out, eh_out = model(batch)
                bg_pred = torch.expm1(bg_out.squeeze().cpu().float() * (ns + 1e-8) + nm)
                bg_true = torch.expm1(batch.bg.cpu().float())
                bg_p.extend(bg_pred.numpy().tolist())
                bg_t.extend(bg_true.numpy().tolist())
                gt_p.extend(gt_out.argmax(1).cpu().numpy().tolist())
                gt_t.extend(batch.gt.cpu().numpy().tolist())
                eh_p.extend(eh_out.argmax(1).cpu().numpy().tolist())
                eh_t.extend(batch.eh.cpu().numpy().tolist())
        collect(res, bg_t, bg_p, gt_t, gt_p, eh_t, eh_p, fold)
        print(f"  MACE+ALIGNN fold {fold}: {len(test_idx)} samples")

    return to_np(res)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MACE+M3GNet  (train_mace_m3gnet / DualBackboneDataset)
# ═══════════════════════════════════════════════════════════════════════════════
def run_mace_m3gnet(mace_base):
    from data_dual_backbone import DualBackboneDataset, stratified_kfold_split
    from model_mace_m3gnet import DualBackboneMACEM3GNet
    from model_m3gnet_pyg  import M3GNetMultiTaskPyG
    from torch_geometric.loader import DataLoader as PygLoader

    CKPT_DIR      = f"{CKPT_BASE}/mace_m3gnet_eh01"
    M3G_CKPT_DIR  = f"{CKPT_BASE}/m3gnet_mt_eh01"
    res = empty_results()

    ck0  = torch.load(f"{CKPT_DIR}/fold0_best.pt", map_location="cpu", weights_only=False)
    args = ck0["args"]

    dataset = DualBackboneDataset(
        mace_cache_dir=args["mace_cache_dir"],
        crystal_cache_dir=args["crystal_cache_dir"],
        root_dir=args["data_dir"],
        merge_metal_indirect=args["merge_metal_indirect"],
        log_bg=args["log_bg"],
        eh_threshold=args.get("eh_threshold", 0.1),
    )
    print(f"  MACE+M3GNet dataset: {len(dataset)} samples")

    sample = dataset[0]
    edge_dim  = sample.alignn_edge_attr.shape[1]
    angle_dim = sample.alignn_line_attr.shape[1]

    for fold in range(N_FOLDS):
        ck   = torch.load(f"{CKPT_DIR}/fold{fold}_best.pt",
                          map_location=device, weights_only=False)
        args = ck["args"]
        nm, ns = float(ck["normalizer_mean"]), float(ck["normalizer_std"])

        _, _, test_idx = stratified_kfold_split(
            dataset, k_folds=args["k_folds"], fold=fold,
            val_ratio=args.get("val_ratio", 0.15), seed=args["seed"])

        m3g_ck = torch.load(f"{M3G_CKPT_DIR}/fold{fold}_best.pt",
                             map_location=device, weights_only=False)
        m3g_args = m3g_ck["args"]
        m3gnet_model = M3GNetMultiTaskPyG(
            hidden_dim=m3g_args.get("hidden_dim", 128),
            n_blocks=m3g_args.get("n_blocks", 3),
            dropout=m3g_args.get("dropout", 0.3),
            edge_input_dim=edge_dim,
            angle_input_dim=angle_dim,
            n_gap_classes=2, n_eh_classes=2)
        m3gnet_model.load_state_dict(m3g_ck["model_state_dict"])

        model = DualBackboneMACEM3GNet(
            mace_base, m3gnet_model,
            h_fea_len=args["h_fea_len"], n_attn_heads=args["n_attn_heads"],
            dropout=args["dropout"], n_gap_classes=2, n_eh_classes=2,
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()

        loader = PygLoader([dataset[i] for i in test_idx],
                           batch_size=16, shuffle=False, num_workers=0)
        bg_t, bg_p, gt_t, gt_p, eh_t, eh_p = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_out, gt_out, eh_out = model(batch)
                bg_pred = torch.expm1(bg_out.squeeze().cpu().float() * (ns + 1e-8) + nm)
                bg_true = torch.expm1(batch.bg.cpu().float())
                bg_p.extend(bg_pred.numpy().tolist())
                bg_t.extend(bg_true.numpy().tolist())
                gt_p.extend(gt_out.argmax(1).cpu().numpy().tolist())
                gt_t.extend(batch.gt.cpu().numpy().tolist())
                eh_p.extend(eh_out.argmax(1).cpu().numpy().tolist())
                eh_t.extend(batch.eh.cpu().numpy().tolist())
        collect(res, bg_t, bg_p, gt_t, gt_p, eh_t, eh_p, fold)
        print(f"  MACE+M3GNet fold {fold}: {len(test_idx)} samples")

    return to_np(res)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Load MACE base once (shared across MACE, MACE+ALIGNN, MACE+M3GNet)
    print("\nLoading MACE-MP-0 small ...")
    import os
    os.environ["HF_HOME"]    = f"{PROJECT}/.cache"
    os.environ["MACE_CACHE"] = f"{PROJECT}/.cache"
    from mace.calculators import mace_mp
    calc      = mace_mp(model="small", default_dtype="float32", device=str(device))
    mace_base = calc.models[0]
    z_table   = calc.z_table
    r_max     = float(calc.r_max)
    print(f"MACE loaded  (r_max={r_max})")

    print("\n=== CGCNN ===")
    make_all_plots("CGCNN", run_cgcnn())

    print("\n=== MACE ===")
    make_all_plots("MACE", run_mace(mace_base, z_table, r_max))

    print("\n=== ALIGNN ===")
    from model_alignn_pyg import ALIGNNMultiTaskPyG
    make_all_plots("ALIGNN", run_crystal_mt("alignn", ALIGNNMultiTaskPyG))

    print("\n=== M3GNet ===")
    from model_m3gnet_pyg import M3GNetMultiTaskPyG
    make_all_plots("M3GNet", run_crystal_mt("m3gnet", M3GNetMultiTaskPyG))

    print("\n=== MACE+M3GNet ===")
    make_all_plots("MACE+M3GNet", run_mace_m3gnet(mace_base))

    print("\n=== MACE+ALIGNN ===")
    make_all_plots("MACE+ALIGNN", run_dual_backbone(mace_base))

    print(f"\n✓ All plots saved to {OUT_DIR}")
