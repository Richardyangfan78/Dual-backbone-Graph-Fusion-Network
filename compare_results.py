"""
Aggregate and compare results from all models on chalcohalide tasks.

Usage: python compare_results.py [--base-dir <checkpoints_dir>] [--fig-out <path>]

Reads fold_*_results.csv from each model's checkpoint directory.
Generates benchmark_results.csv and a multi-panel bar chart figure.
"""

import os
import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MODELS = {
    "CGCNN":       None,              # read from hardcoded values below
    "MACE":        "mace_finetune_v2",
    "ALIGNN":      "alignn_mt",
    "M3GNet":      "m3gnet_mt",
    "MACE+M3GNet": "mace_m3gnet",
    "MACE+ALIGNN": "dual_backbone",
}

CGCNN_RESULTS = {
    # Extracted from v10 log files (best composite model per fold)
    0: {"bg_mae": 0.6303, "gt_acc": 0.6943, "gt_f1": 0.7995, "eh_acc": 0.7751, "eh_f1": None},
    1: {"bg_mae": 0.5503, "gt_acc": 0.7160, "gt_f1": 0.8182, "eh_acc": 0.8264, "eh_f1": None},
    2: {"bg_mae": 0.6837, "gt_acc": 0.6950, "gt_f1": 0.7990, "eh_acc": 0.7941, "eh_f1": None},
    3: {"bg_mae": 0.6081, "gt_acc": 0.6554, "gt_f1": 0.7661, "eh_acc": 0.8158, "eh_f1": None},
    4: {"bg_mae": 0.6116, "gt_acc": 0.6891, "gt_f1": 0.7898, "eh_acc": 0.7980, "eh_f1": None},
}


def read_model_results(ckpt_dir, n_folds=5):
    """Read fold_*_results.csv files; return dict of metric → list[fold values]."""
    metrics = {}
    for fold in range(n_folds):
        path = os.path.join(ckpt_dir, f"fold{fold}_results.csv")
        if not os.path.exists(path):
            print(f"  WARNING: missing {path}")
            continue
        with open(path) as f:
            for row in csv.reader(f):
                if row[0] == "metric":
                    continue
                k, v = row[0], float(row[1])
                metrics.setdefault(k, []).append(v)
    return metrics


def summarise(values):
    """Return (mean, std) tuple."""
    if not values:
        return None, None
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def summarise_str(values):
    """Return mean ± std as string."""
    mean, std = summarise(values)
    if mean is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}"


def print_table(results_by_model):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print(f"{'Model':<14}  {'BG MAE (eV)':^22}  {'GT Acc':^22}  {'GT F1':^22}  {'EH Acc':^22}")
    print("=" * 100)
    for model, res in results_by_model.items():
        bg  = summarise_str(res.get("bg_mae", []))
        gt  = summarise_str(res.get("gt_acc", []))
        gtf = summarise_str(res.get("gt_f1",  []))
        eh  = summarise_str(res.get("eh_acc", []))
        print(f"{model:<14}  {bg:^22}  {gt:^22}  {gtf:^22}  {eh:^22}")
    print("=" * 100)

    print()
    print(f"{'Model':<14}  {'EH F1':^22}")
    print("-" * 40)
    for model, res in results_by_model.items():
        eh_f1 = summarise_str(res.get("eh_f1", []))
        print(f"{model:<14}  {eh_f1:^22}")
    print("-" * 40)

    # Per-fold details
    for metric, label in [("bg_mae","BG MAE"), ("gt_acc","GT Acc"), ("eh_acc","EH Acc")]:
        print(f"\n── Per-fold {label} ──")
        for model, res in results_by_model.items():
            vals = res.get(metric, [])
            fold_str = "  ".join(f"{v:.4f}" for v in vals) if vals else "N/A"
            print(f"{model:<14}: {fold_str}")


def generate_figure(results_by_model, fig_out):
    """Generate a 3-panel bar chart comparing all models."""
    model_names = list(results_by_model.keys())
    n = len(model_names)
    x = np.arange(n)
    width = 0.65

    # Nature NPG color palette (ggsci npg)
    COLORS = {
        "CGCNN":       "#B09C85",  # warm gray-brown (weakest baseline)
        "MACE":        "#4DBBD5",  # NPG cyan
        "ALIGNN":      "#00A087",  # NPG teal
        "M3GNet":      "#E64B35",  # NPG red-orange
        "MACE+M3GNet": "#3C5488",  # NPG dark navy
        "MACE+ALIGNN": "#DC0000",  # NPG deep red (best model)
    }
    bar_colors = [COLORS.get(m, "#7E6148") for m in model_names]

    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          10,
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "figure.dpi":         300,
        "savefig.dpi":        300,
    })

    # Nature double-column width: ~7.08 in; height ~4.5 in
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.2))
    fig.suptitle("Chalcohalide GNN — Multi-Model Benchmark (5-fold CV)",
                 fontsize=10, fontweight="bold", y=1.03)

    def _bar(ax, metric, ylabel, title, lower_better=True):
        means, stds = [], []
        for m in model_names:
            mu, sd = summarise(results_by_model[m].get(metric, []))
            means.append(mu if mu is not None else 0.0)
            stds.append(sd if sd is not None else 0.0)
        bars = ax.bar(x, means, width, yerr=stds, capsize=3,
                      color=bar_colors, edgecolor="white", linewidth=0.6,
                      error_kw=dict(elinewidth=1.0, ecolor="#444444",
                                    capthick=1.0))
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
        ax.yaxis.grid(True, alpha=0.25, ls="--", lw=0.6, color="#555555")
        ax.set_axisbelow(True)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        # Value annotations
        for bar, mu in zip(bars, means):
            if mu > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(means) * 0.015,
                        f"{mu:.3f}", ha="center", va="bottom",
                        fontsize=6.5, color="#222222")
        if lower_better:
            ax.set_ylim(0, max(means) * 1.28)
        else:
            valid = [m for m in means if m > 0]
            ax.set_ylim(max(0.0, min(valid) - 0.12), 1.04)

    _bar(axes[0], "bg_mae",  "MAE (eV)",   "Bandgap MAE ↓",          lower_better=True)
    _bar(axes[1], "gt_acc",  "Accuracy",   "Gap Type Accuracy ↑",     lower_better=False)
    _bar(axes[2], "eh_acc",  "Accuracy",   "EH Stability Accuracy ↑", lower_better=False)

    # Compact legend below panels
    legend_patches = [Patch(facecolor=COLORS.get(m, "#7E6148"),
                            edgecolor="none", label=m)
                      for m in model_names]
    fig.legend(handles=legend_patches, loc="lower center", ncol=n,
               bbox_to_anchor=(0.5, -0.10), fontsize=7.5,
               frameon=False, handlelength=1.2, handletextpad=0.5,
               columnspacing=1.0)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(fig_out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nBenchmark figure saved → {fig_out}")


def save_csv(results_by_model, csv_out):
    """Save mean ± std summary to CSV."""
    metrics = ["bg_mae", "gt_acc", "gt_f1", "eh_acc", "eh_f1"]
    rows = []
    for model, res in results_by_model.items():
        row = {"model": model}
        for m in metrics:
            mu, sd = summarise(res.get(m, []))
            row[f"{m}_mean"] = f"{mu:.4f}" if mu is not None else "N/A"
            row[f"{m}_std"]  = f"{sd:.4f}" if sd is not None else "N/A"
        rows.append(row)
    with open(csv_out, "w", newline="") as f:
        fieldnames = ["model"] + [f"{m}_{s}" for m in metrics for s in ("mean","std")]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV saved → {csv_out}")


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",
        default=str(project / "checkpoints"))
    parser.add_argument("--fig-out",
        default=str(project / "benchmark_comparison.png"))
    parser.add_argument("--csv-out",
        default=str(project / "benchmark_results.csv"))
    args = parser.parse_args()

    results_by_model = {}

    # CGCNN: use hardcoded values
    cgcnn_res = {}
    for k in ["bg_mae", "gt_acc", "gt_f1", "eh_acc"]:
        cgcnn_res[k] = [CGCNN_RESULTS[f][k] for f in range(5)
                        if CGCNN_RESULTS[f].get(k) is not None]
    results_by_model["CGCNN"] = cgcnn_res

    # Other models: read CSV files
    for model_name, ckpt_subdir in MODELS.items():
        if ckpt_subdir is None:
            continue
        ckpt_dir = os.path.join(args.base_dir, ckpt_subdir)
        if not os.path.isdir(ckpt_dir):
            print(f"[{model_name}] checkpoint dir not found: {ckpt_dir}")
            results_by_model[model_name] = {}
            continue
        res = read_model_results(ckpt_dir)
        n_folds = len(res.get("bg_mae", []))
        if res:
            print(f"[{model_name}] {n_folds} folds found")
        results_by_model[model_name] = res

    print_table(results_by_model)
    save_csv(results_by_model, args.csv_out)
    generate_figure(results_by_model, args.fig_out)


if __name__ == "__main__":
    main()
