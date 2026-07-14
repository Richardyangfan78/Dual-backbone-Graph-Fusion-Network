"""
Gate analysis for MACE+ALIGNN dual-backbone model.

For each fold's test set:
  1. Extract gate weight g ∈ [0,1] per crystal (g→1 = prefer MACE, g→0 = prefer ALIGNN)
  2. Extract crystal metadata: formula, crystal system, BG, GT label, EH label
  3. Save all to CSV for plotting
  4. Generate figures:
     a. Gate distribution histogram (all data)
     b. Gate by crystal system (boxplot)
     c. Gate by GT class (violin)
     d. Gate by EH class (violin)
     e. Gate vs BG scatter (coloured by GT)
     f. Task-conditional gate: per-task preferred backbone
"""
import os
import sys
import csv
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ── project root on PYTHONPATH ──────────────────────────────────────────────
PROJECT = "/path/to/Dual-backbone-Graph-Fusion-Network"
sys.path.insert(0, os.path.join(PROJECT, "multitask"))

from model_dual_backbone import DualBackboneMultiTask, GatedFusion
from model_alignn_pyg import ALIGNNMultiTaskPyG
from data_dual_backbone import DualBackboneDataset, stratified_kfold_split
from torch_geometric.loader import DataLoader


# ── colour palette ───────────────────────────────────────────────────────────
CRYSTAL_SYSTEM_ORDER = [
    "triclinic", "monoclinic", "orthorhombic",
    "tetragonal", "trigonal", "hexagonal", "cubic",
]
CRYSTAL_SYSTEM_COLORS = dict(zip(
    CRYSTAL_SYSTEM_ORDER,
    sns.color_palette("husl", len(CRYSTAL_SYSTEM_ORDER)),
))
GT_LABELS  = {0: "Direct", 1: "Indirect"}   # after merge_metal_indirect
EH_LABELS  = {0: "Stable", 1: "Unstable"}


# ── Patch GatedFusion to expose gate values ──────────────────────────────────

class GatedFusionWithHook(GatedFusion):
    """Subclass that stores the last gate tensor for inspection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_gate: torch.Tensor | None = None

    def forward(self, f_mace, f_alignn):
        concat = torch.cat([f_mace, f_alignn], dim=1)
        g = self.gate(concat)                       # [B, D]
        self._last_gate = g.detach().cpu()          # save
        gated = g * f_mace + (1 - g) * f_alignn
        projected = self.proj(concat)
        return gated + projected


def patch_fusion(model: DualBackboneMultiTask) -> GatedFusionWithHook:
    """Replace model.fusion with a GatedFusionWithHook in-place."""
    orig = model.fusion
    dim = orig.gate[0].in_features // 2    # gate: Linear(2D, D)
    dropout = 0.0                           # inference only
    hook = GatedFusionWithHook(dim, dropout=dropout).to(
        next(model.parameters()).device
    )
    # Copy weights
    hook.load_state_dict(orig.state_dict())
    model.fusion = hook
    return hook


# ── Crystal system from CIF ──────────────────────────────────────────────────

def get_crystal_systems(mp_ids, cif_dir):
    """Returns dict mp_id → crystal_system string (lowercase)."""
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        print("WARNING: pymatgen not available; crystal_system = 'unknown'")
        return {mid: "unknown" for mid in mp_ids}

    result = {}
    for mid in mp_ids:
        cif_path = os.path.join(cif_dir, f"{mid}.cif")
        if not os.path.exists(cif_path):
            result[mid] = "unknown"
            continue
        try:
            struct = Structure.from_file(cif_path)
            sga = SpacegroupAnalyzer(struct, symprec=0.1)
            result[mid] = sga.get_crystal_system()
        except Exception:
            result[mid] = "unknown"
    return result


# ── Load model ───────────────────────────────────────────────────────────────

def load_model(ckpt_path, device, dropout=0.3):
    """Load DualBackboneMultiTask from checkpoint (no MACE calc needed)."""
    from mace.calculators import mace_mp
    calc = mace_mp(model="small", default_dtype="float32", device=str(device))
    mace_model_base = calc.models[0]

    alignn_model = ALIGNNMultiTaskPyG(
        atom_input_dim=94, edge_input_dim=80, angle_input_dim=40,
        hidden_dim=256, n_alignn_layers=4, n_gcn_layers=4,
        n_gap_classes=2, n_eh_classes=2, dropout=dropout,
    )
    model = DualBackboneMultiTask(
        mace_model_base, alignn_model,
        h_fea_len=256, n_attn_heads=8, dropout=dropout,
        use_cosine_classifier=True, cosine_temp=0.1,
    ).to(device)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.set_cond_weight(1.0)

    mean = ckpt.get("normalizer_mean", 0.0)
    std  = ckpt.get("normalizer_std",  1.0)
    return model, mean, std


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_gates(model, loader, fusion_hook, normalizer_mean, normalizer_std,
                  device, log_bg=True):
    """Run inference; collect per-sample gate means, predictions, labels."""
    records = []
    for batch in loader:
        batch = batch.to(device)
        bg_pred, gt_pred, eh_pred = model(batch)

        # Gate: [B, D] → mean over feature dim → scalar per sample
        g = fusion_hook._last_gate                  # [B, D]
        g_mean = g.mean(dim=1).numpy()              # [B]
        g_mace = g_mean                             # >0.5 → MACE wins
        g_alignn = 1.0 - g_mean                    # <0.5 → ALIGNN wins

        # BG in eV
        bg_denorm = bg_pred.squeeze().cpu() * (normalizer_std + 1e-8) + normalizer_mean
        if log_bg:
            bg_ev = torch.expm1(bg_denorm).numpy()
        else:
            bg_ev = bg_denorm.numpy()

        gt_true  = batch.gt.cpu().squeeze().numpy()
        eh_true  = batch.eh.cpu().squeeze().numpy()
        gt_hat   = gt_pred.argmax(dim=1).cpu().numpy()
        eh_hat   = eh_pred.argmax(dim=1).cpu().numpy()
        bg_true  = batch.bg.cpu().numpy()

        mp_ids = getattr(batch, "mp_id", [None] * len(g_mean))
        if not isinstance(mp_ids, list):
            mp_ids = [None] * len(g_mean)

        for i in range(len(g_mean)):
            records.append({
                "mp_id":     mp_ids[i],
                "gate_mean": float(g_mean[i]),
                "gate_mace": float(g_mace[i]),
                "gate_alignn": float(g_alignn[i]),
                "bg_true":   float(bg_true[i]),
                "bg_pred":   float(bg_ev[i]),
                "gt_true":   int(gt_true[i]),
                "gt_pred":   int(gt_hat[i]),
                "eh_true":   int(eh_true[i]),
                "eh_pred":   int(eh_hat[i]),
                "gt_correct": int(gt_true[i] == gt_hat[i]),
                "eh_correct": int(eh_true[i] == eh_hat[i]),
            })
    return records


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_all(records, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    gates  = np.array([r["gate_mean"] for r in records])
    bg     = np.array([r["bg_true"]   for r in records])
    gt_true = np.array([r["gt_true"]  for r in records])
    eh_true = np.array([r["eh_true"]  for r in records])
    cs      = np.array([r["crystal_system"] for r in records])
    gt_corr = np.array([r["gt_correct"] for r in records])
    eh_corr = np.array([r["eh_correct"] for r in records])

    sns.set_theme(style="whitegrid", font_scale=1.15)
    fig_size = (7, 4.5)

    # ── Fig 1: Overall gate distribution ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=fig_size)
    ax.hist(gates, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(0.5, ls="--", color="gray", lw=1.2, label="g = 0.5 (equal weight)")
    ax.axvline(gates.mean(), ls="-", color="#DD4444", lw=1.5,
               label=f"mean = {gates.mean():.3f}")
    ax.set_xlabel("Gate weight  g  (→ 1 = prefer MACE,  → 0 = prefer ALIGNN)")
    ax.set_ylabel("Count")
    ax.set_title("MACE-ALIGNN Gate Distribution (all test folds)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig1_gate_distribution.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig1_gate_distribution.png")

    # ── Fig 2: Gate by crystal system ─────────────────────────────────────────
    cs_present = [c for c in CRYSTAL_SYSTEM_ORDER if c in cs]
    data_cs = {c: gates[cs == c] for c in cs_present if (cs == c).sum() > 0}
    if "unknown" in set(cs):
        data_cs["unknown"] = gates[cs == "unknown"]
        cs_present.append("unknown")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = range(len(cs_present))
    bp = ax.boxplot(
        [data_cs[c] for c in cs_present],
        positions=list(positions),
        widths=0.55,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    colors = [CRYSTAL_SYSTEM_COLORS.get(c, "#AAAAAA") for c in cs_present]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(0.5, ls="--", color="gray", lw=1.0)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [f"{c}\n(N={len(data_cs[c])})" for c in cs_present],
        fontsize=10,
    )
    ax.set_ylabel("Gate weight  g")
    ax.set_title("Gate Weight by Crystal System")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_gate_by_crystal_system.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig2_gate_by_crystal_system.png")

    # ── Fig 3: Gate by GT class (violin) ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for ax, (correct_only, title_suffix) in zip(
        axes, [(False, "all"), (True, "correct preds only")]
    ):
        for cls_idx, cls_name in GT_LABELS.items():
            mask = gt_true == cls_idx
            if correct_only:
                mask = mask & (gt_corr == 1)
            g_sub = gates[mask]
            if len(g_sub) < 5:
                continue
            color = "#4C72B0" if cls_idx == 0 else "#DD8452"
            parts = ax.violinplot(g_sub, positions=[cls_idx],
                                  showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.65)
            parts["cmedians"].set_color("black")
        ax.axhline(0.5, ls="--", color="gray", lw=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([GT_LABELS[0], GT_LABELS[1]])
        ax.set_title(f"Gate by Gap Type ({title_suffix})")
        ax.set_ylabel("Gate weight  g")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig3_gate_by_gap_type.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig3_gate_by_gap_type.png")

    # ── Fig 4: Gate by EH class (violin) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for cls_idx, cls_name in EH_LABELS.items():
        g_sub = gates[eh_true == cls_idx]
        if len(g_sub) < 5:
            continue
        color = "#55A868" if cls_idx == 0 else "#C44E52"
        parts = ax.violinplot(g_sub, positions=[cls_idx],
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        ax.text(cls_idx, g_sub.mean() + 0.03, f"μ={g_sub.mean():.3f}",
                ha="center", fontsize=10, color="black")
    ax.axhline(0.5, ls="--", color="gray", lw=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([EH_LABELS[0], EH_LABELS[1]])
    ax.set_title("Gate Weight by Stability (E_hull)")
    ax.set_ylabel("Gate weight  g")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_gate_by_stability.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig4_gate_by_stability.png")

    # ── Fig 5: Gate vs BG scatter (coloured by GT) ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = {0: "#4C72B0", 1: "#DD8452"}
    for cls_idx, cls_name in GT_LABELS.items():
        mask = gt_true == cls_idx
        ax.scatter(bg[mask], gates[mask], c=cmap[cls_idx], alpha=0.35,
                   s=18, label=cls_name, linewidths=0)
    ax.axhline(0.5, ls="--", color="gray", lw=1.0, label="g = 0.5")
    ax.set_xlabel("Bandgap (eV)  [DFT]")
    ax.set_ylabel("Gate weight  g")
    ax.set_title("Gate Weight vs Bandgap (coloured by Gap Type)")
    ax.legend(title="Gap type", framealpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig5_gate_vs_bandgap.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig5_gate_vs_bandgap.png")

    # ── Fig 6: Heatmap gate mean by (crystal system, GT class) ───────────────
    from collections import defaultdict
    cs_gt_gate = defaultdict(list)
    for r in records:
        key = (r["crystal_system"], r["gt_true"])
        cs_gt_gate[key].append(r["gate_mean"])

    cs_list = [c for c in CRYSTAL_SYSTEM_ORDER if c in set(cs)]
    heatmap_data = np.full((len(cs_list), 2), np.nan)
    for i, c in enumerate(cs_list):
        for j in [0, 1]:
            vals = cs_gt_gate[(c, j)]
            if vals:
                heatmap_data[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(5.5, len(cs_list) * 0.75 + 1.5))
    im = ax.imshow(heatmap_data, cmap="RdBu_r", vmin=0.3, vmax=0.7, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Indirect", "Direct"])
    ax.set_yticks(range(len(cs_list)))
    ax.set_yticklabels(cs_list)
    for i in range(len(cs_list)):
        for j in range(2):
            v = heatmap_data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, color="black" if 0.35 < v < 0.65 else "white")
    plt.colorbar(im, ax=ax, label="Mean gate weight  g\n(→1=MACE, →0=ALIGNN)")
    ax.set_title("Mean Gate by Crystal System × Gap Type")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_heatmap_cs_gt.png"), dpi=150)
    plt.close(fig)
    print("  Saved fig6_heatmap_cs_gt.png")

    # ── Fig 7: Feature-dimension gate distribution (mean per-dim, sorted) ────
    # Collect all gate vectors [B, D] across folds
    all_gate_vecs = [r.get("gate_vec") for r in records if r.get("gate_vec") is not None]
    if all_gate_vecs:
        gate_mat = np.stack(all_gate_vecs, axis=0)   # [N, D]
        dim_means = gate_mat.mean(axis=0)
        dim_sorted = np.sort(dim_means)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(dim_sorted, "o", markersize=3, alpha=0.7)
        axes[0].axhline(0.5, ls="--", color="gray", lw=1.0)
        axes[0].set_xlabel("Feature dimension (sorted by gate mean)")
        axes[0].set_ylabel("Mean gate value")
        axes[0].set_title("Per-Dimension Gate Values (sorted)")

        axes[1].hist(dim_means, bins=30, color="#4C72B0", edgecolor="white")
        axes[1].axvline(0.5, ls="--", color="gray", lw=1.0)
        axes[1].set_xlabel("Mean gate value per feature dimension")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Distribution of Dim-wise Gate Means")

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig7_per_dim_gate.png"), dpi=150)
        plt.close(fig)
        print("  Saved fig7_per_dim_gate.png")

    # ── Summary stats text ────────────────────────────────────────────────────
    print("\n=== Gate Analysis Summary ===")
    print(f"  Total samples:    {len(records)}")
    print(f"  Gate mean (all):  {gates.mean():.4f}  ± {gates.std():.4f}")
    print(f"  Gate > 0.5 (MACE-dominant): {(gates > 0.5).mean():.1%}")
    print(f"  Gate < 0.5 (ALIGNN-dominant): {(gates < 0.5).mean():.1%}")
    print()
    for cs_name in CRYSTAL_SYSTEM_ORDER:
        mask = cs == cs_name
        if mask.sum() > 0:
            print(f"  {cs_name:<14}: N={mask.sum():4d}  g={gates[mask].mean():.3f} ± {gates[mask].std():.3f}")
    print()
    for cls_idx, cls_name in GT_LABELS.items():
        mask = gt_true == cls_idx
        if mask.sum() > 0:
            print(f"  GT {cls_name:<10}: N={mask.sum():4d}  g={gates[mask].mean():.3f} ± {gates[mask].std():.3f}")
    print()
    for cls_idx, cls_name in EH_LABELS.items():
        mask = eh_true == cls_idx
        if mask.sum() > 0:
            print(f"  EH {cls_name:<10}: N={mask.sum():4d}  g={gates[mask].mean():.3f} ± {gates[mask].std():.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gate analysis for MACE+ALIGNN model")
    parser.add_argument("--data-dir",
        default=os.path.join(PROJECT, "Data/multitask"))
    parser.add_argument("--mace-cache-dir", default=None)
    parser.add_argument("--crystal-cache-dir", default=None)
    parser.add_argument("--ckpt-dir",
        default=os.path.join(PROJECT, "checkpoints/dual_backbone"))
    parser.add_argument("--alignn-ckpt-dir",
        default=os.path.join(PROJECT, "checkpoints/alignn_mt"))
    parser.add_argument("--cif-dir",
        default=os.path.join(PROJECT, "Data/multitask"))
    parser.add_argument("--out-dir",
        default=os.path.join(PROJECT, "results/gate_analysis"))
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Device: {device}")

    mace_cache = args.mace_cache_dir or os.path.join(args.data_dir, "mace_cached_graphs")
    crystal_cache = args.crystal_cache_dir or os.path.join(args.data_dir, "crystal_cached_graphs")

    # ── Load full dataset ─────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = DualBackboneDataset(
        root_dir=args.data_dir,
        mace_cache_dir=mace_cache,
        crystal_cache_dir=crystal_cache,
        merge_metal_indirect=True,
        log_bg=True,
    )
    print(f"  {len(dataset)} samples loaded")

    # ── Crystal systems ───────────────────────────────────────────────────────
    print("Extracting crystal systems from CIF files...")
    all_mp_ids = [s["mp_id"] for s in dataset.samples]
    cs_map = get_crystal_systems(all_mp_ids, args.cif_dir)
    print(f"  {sum(1 for v in cs_map.values() if v != 'unknown')} / {len(cs_map)} resolved")

    # ── Process each fold ─────────────────────────────────────────────────────
    all_records = []

    for fold in range(args.k_folds):
        ckpt_path = os.path.join(args.ckpt_dir, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold}: checkpoint not found, skipping")
            continue

        print(f"\nFold {fold}: loading checkpoint from {ckpt_path}")
        model, norm_mean, norm_std = load_model(ckpt_path, device)

        # Patch fusion to expose gate
        fusion_hook = patch_fusion(model)
        print(f"  Patched GatedFusion → GatedFusionWithHook")

        # Get test indices for this fold
        _, _, test_idx = stratified_kfold_split(
            dataset, k_folds=args.k_folds, fold=fold,
            val_ratio=args.val_ratio, seed=args.seed,
        )
        print(f"  Test set: {len(test_idx)} samples")
        test_data = [dataset[i] for i in test_idx]
        loader = DataLoader(test_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers)

        # Extract gates
        records = extract_gates(
            model, loader, fusion_hook, norm_mean, norm_std, device, log_bg=True
        )

        # Attach crystal system and mp_id by re-indexing
        for i, (rec, idx) in enumerate(zip(records, test_idx)):
            mp_id = dataset.samples[idx]["mp_id"]
            rec["mp_id"] = mp_id
            rec["crystal_system"] = cs_map.get(mp_id, "unknown")
            rec["fold"] = fold

        all_records.extend(records)
        print(f"  Extracted {len(records)} gate values")

        del model
        torch.cuda.empty_cache()

    if not all_records:
        print("ERROR: No records collected — check checkpoint paths")
        return

    # ── Save raw CSV ──────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "gate_analysis.csv")
    fieldnames = [
        "fold", "mp_id", "crystal_system",
        "gate_mean", "gate_mace", "gate_alignn",
        "bg_true", "bg_pred", "gt_true", "gt_pred", "gt_correct",
        "eh_true", "eh_pred", "eh_correct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\nRaw data saved → {csv_path}  ({len(all_records)} rows)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_all(all_records, args.out_dir)
    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
