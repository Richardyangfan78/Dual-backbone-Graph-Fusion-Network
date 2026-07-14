#!/usr/bin/env python3
"""t-SNE: highlight ideal candidates in the final main6 predicted set."""
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("/path/to/Dual-backbone-Graph-Fusion-Network")
OUT = BASE / "classical_ml_baseline/final_main6_tsne"
PKL = OUT / "features_final_main6.pkl"
EMB = OUT / "tsne_embedding_final_main6.npy"

BG_LO, BG_HI = 0.5, 1.1

with PKL.open("rb") as f:
    d = pickle.load(f)
bg = np.asarray(d["bg"], dtype=float)
gt = np.asarray(d["gt"], dtype=int)
eh = np.asarray(d["eh"], dtype=float)
screen_pass = np.asarray(d.get("screen_pass", np.zeros_like(bg, dtype=bool)), dtype=bool)
X2d = np.load(EMB)

is_direct = gt == 0
is_stable = eh < 0.1
is_bg_range = (bg >= BG_LO) & (bg <= BG_HI)
is_ideal = is_direct & is_stable & is_bg_range
not_ideal = ~is_ideal

print(
    f"Total: {len(gt)} | Direct: {int(is_direct.sum())} | Stable: {int(is_stable.sum())} | "
    f"BG range: {int(is_bg_range.sum())} | IDEAL(0.5-1.1): {int(is_ideal.sum())} | "
    f"screen_pass: {int(screen_pass.sum())}",
    flush=True,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GREY_BG = "#D0D0D0"
IDEAL_C = "#C0392B"

fig, ax = plt.subplots(figsize=(6.2, 5.8))
xlim = (X2d[:, 0].min() - 4, X2d[:, 0].max() + 4)
ylim = (X2d[:, 1].min() - 4, X2d[:, 1].max() + 4)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

ax.scatter(X2d[not_ideal, 0], X2d[not_ideal, 1],
           c=GREY_BG, s=7, alpha=0.38, linewidths=0, zorder=1, rasterized=True)
ax.scatter(X2d[is_ideal, 0], X2d[is_ideal, 1],
           c=IDEAL_C, s=9, marker="o", linewidths=0, alpha=0.95, zorder=4, rasterized=True)

criteria_txt = (
    "Selection criteria\n"
    "  Direct bandgap\n"
    f"  $E_{{g}}$ = {BG_LO}-{BG_HI} eV\n"
    "  Stable"
)
ax.text(
    0.02,
    0.98,
    criteria_txt,
    transform=ax.transAxes,
    va="top",
    ha="left",
    fontsize=8.5,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#BBBBBB", lw=0.7, alpha=0.9),
    zorder=6,
)

handles = [
    mpatches.Patch(facecolor=GREY_BG, edgecolor="none", label=f"Other materials (n = {int(not_ideal.sum())})"),
    mpatches.Patch(facecolor=IDEAL_C, edgecolor="none", label=f"Ideal candidates (n = {int(is_ideal.sum())})"),
]
ax.legend(handles=handles, fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC", handlelength=1.2)

ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.set_title("Ideal Chalcohalides", fontsize=11, fontweight="bold", pad=8)
ax.grid(True, alpha=0.15, ls="--", lw=0.5, color="#AAAAAA")
ax.set_axisbelow(True)

fig.tight_layout()
png = OUT / "final_main6_tsne_ideal_materials.png"
pdf = OUT / "final_main6_tsne_ideal_materials.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(pdf, bbox_inches="tight", facecolor="white")
print(f"Saved -> {png}", flush=True)
print(f"Saved -> {pdf}", flush=True)
