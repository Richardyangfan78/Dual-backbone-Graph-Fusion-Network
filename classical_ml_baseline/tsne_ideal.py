#!/usr/bin/env python3
"""t-SNE: highlight ideal materials (Direct + 0.5–1.1 eV + Stable)."""
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
PKL  = f"{BASE}/classical_ml_baseline/features.pkl"
EMB  = f"{BASE}/classical_ml_baseline/tsne_embedding.npy"
OUT  = f"{BASE}/classical_ml_baseline/tsne_ideal_materials.png"

BG_LO, BG_HI = 0.5, 1.1

# ── Load ──────────────────────────────────────────────────────────────────────
with open(PKL, "rb") as f:
    d = pickle.load(f)
bg  = d["bg"]
gt  = d["gt"]   # 0=Direct 1=Indirect 2=Metal
eh  = d["eh"]
X2d = np.load(EMB)

# ── Masks ─────────────────────────────────────────────────────────────────────
is_direct   = gt == 0
is_stable   = eh < 0.1
is_bg_range = (bg >= BG_LO) & (bg <= BG_HI)
is_ideal    = is_direct & is_stable & is_bg_range
not_ideal   = ~is_ideal

n_ideal = is_ideal.sum()
print(f"Total: {len(gt)}  |  Direct: {is_direct.sum()}  |  "
      f"Stable: {is_stable.sum()}  |  BG range: {is_bg_range.sum()}  |  IDEAL: {n_ideal}")

# ── Style — Nature palette ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

GREY_BG   = "#D0D0D0"   # non-ideal background
IDEAL_C   = "#C0392B"   # Nature-style deep red for ideal candidates
IDEAL_EC  = "#7B241C"   # edge colour

fig, ax = plt.subplots(figsize=(6.2, 5.8))

xlim = (X2d[:, 0].min() - 4, X2d[:, 0].max() + 4)
ylim = (X2d[:, 1].min() - 4, X2d[:, 1].max() + 4)
ax.set_xlim(xlim); ax.set_ylim(ylim)

# Layer 1 — all non-ideal: neutral grey
ax.scatter(X2d[not_ideal, 0], X2d[not_ideal, 1],
           c=GREY_BG, s=8, alpha=0.40, linewidths=0,
           zorder=1, rasterized=True)

# Layer 2 — ideal candidates: same size as others, vivid red
ax.scatter(X2d[is_ideal, 0], X2d[is_ideal, 1],
           c=IDEAL_C, s=8, marker="o",
           linewidths=0,
           alpha=0.95, zorder=4)

# ── Criteria box (top-left) ───────────────────────────────────────────────────
criteria_txt = (
    "Selection criteria\n"
    f"  Direct bandgap\n"
    f"  $E_{{g}}$ = {BG_LO}–{BG_HI} eV\n"
    f"  $E_{{hull}}$ < 0.1 eV/atom"
)
ax.text(0.02, 0.98, criteria_txt,
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.45", fc="white",
                  ec="#BBBBBB", lw=0.7, alpha=0.9),
        zorder=6)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=GREY_BG, edgecolor="none",
                   label=f"Other materials (n = {not_ideal.sum()})"),
    mpatches.Patch(facecolor=IDEAL_C, edgecolor="none",
                   label=f"Ideal Candidates (n = {n_ideal})"),
]
ax.legend(handles=handles, fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#CCCCCC", handlelength=1.2)

ax.set_xlabel("t-SNE 1", fontsize=10)
ax.set_ylabel("t-SNE 2", fontsize=10)
ax.set_title(
    "Ideal Chalcohalides",
    fontsize=11, fontweight="bold", pad=8,
)
ax.grid(True, alpha=0.15, ls="--", lw=0.5, color="#AAAAAA")
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
