#!/usr/bin/env python3
"""Plot six-model benchmark comparison from benchmark_results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "CGCNN",
    "MACE",
    "ALIGNN",
    "M3GNet",
    "MACE+M3GNet",
    "MACE+ALIGNN",
]

MODEL_COLORS = {
    "CGCNN": "#B09C85",
    "MACE": "#4DBBD5",
    "ALIGNN": "#00A087",
    "M3GNet": "#E64B35",
    "MACE+M3GNet": "#3C5488",
    "MACE+ALIGNN": "#DC0000",
}

METRICS = [
    ("bg_mae_mean", "bg_mae_std", "Bandgap MAE (eV)", True),
    ("gt_acc_mean", "gt_acc_std", "Gap Type Accuracy", False),
    ("gt_f1_mean", "gt_f1_std", "Gap Type F1", False),
    ("eh_acc_mean", "eh_acc_std", "EH Stability Accuracy", False),
    ("eh_f1_mean", "eh_f1_std", "EH Stability F1", False),
]


def to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    if "model" not in df.columns:
        raise ValueError(f"Missing 'model' column in {csv_path}")

    df = df[df["model"].isin(MODEL_ORDER)].copy()
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("model").reset_index(drop=True)

    missing = [m for m in MODEL_ORDER if m not in set(df["model"].astype(str))]
    if missing:
        raise ValueError(f"Missing models in CSV: {missing}")

    return df


def plot_metrics(df: pd.DataFrame, out_path: Path, dpi: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    axes = axes.flatten()

    x = np.arange(len(MODEL_ORDER))
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    for ax_idx, (mean_col, std_col, title, lower_better) in enumerate(METRICS):
        ax = axes[ax_idx]
        means = df[mean_col].map(to_float).to_numpy()
        stds = df[std_col].map(to_float).to_numpy()

        plot_vals = np.nan_to_num(means, nan=0.0)
        plot_err = np.nan_to_num(stds, nan=0.0)
        bars = ax.bar(
            x,
            plot_vals,
            yerr=plot_err,
            capsize=3,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"elinewidth": 0.9, "ecolor": "#333333", "capthick": 0.9},
        )

        valid = ~np.isnan(means)
        if np.any(valid):
            valid_vals = means[valid]
            y_min = float(np.min(valid_vals))
            y_max = float(np.max(valid_vals))
            y_span = max(y_max - y_min, 1e-3)

            if lower_better:
                best_idx = int(np.nanargmin(means))
                ax.set_ylim(0.0, y_max + y_span * 0.9)
            else:
                best_idx = int(np.nanargmax(means))
                ax.set_ylim(max(0.0, y_min - y_span * 0.45), min(1.05, y_max + y_span * 0.35))

            best_height = bars[best_idx].get_height()
            ax.text(
                best_idx,
                best_height + y_span * 0.05,
                "best",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#111111",
                fontweight="bold",
            )

        y0, y1 = ax.get_ylim()
        y_text_offset = max((y1 - y0) * 0.015, 0.003)

        for i, (bar, mean_v) in enumerate(zip(bars, means)):
            if np.isnan(mean_v):
                bar.set_color("#D3D3D3")
                bar.set_hatch("//")
                ax.text(i, y0 + y_text_offset, "N/A", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_text_offset,
                    f"{mean_v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#1f1f1f",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=28, ha="right")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.90,
        "Six-model benchmark\n(5-fold mean ± std)",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    axes[-1].text(
        0.02,
        0.70,
        "Models:\n- CGCNN\n- MACE\n- ALIGNN\n- M3GNet\n- MACE+M3GNet\n- MACE+ALIGNN",
        fontsize=10,
        va="top",
    )

    fig.suptitle("Chalcohalide_GNN Model Performance Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot six-model benchmark comparison.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_results.csv",
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "benchmark_six_models.png",
        help="Output figure path",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    args = parser.parse_args()

    df = load_data(args.csv)
    plot_metrics(df, args.out, args.dpi)

    print(f"Input CSV  : {args.csv}")
    print(f"Output PNG : {args.out}")


if __name__ == "__main__":
    main()
