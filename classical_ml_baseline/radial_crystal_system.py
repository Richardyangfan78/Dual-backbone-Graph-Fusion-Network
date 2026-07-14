#!/usr/bin/env python3
"""Radial tree: crystal system → individual materials coloured by gap type."""
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
PKL  = f"{BASE}/classical_ml_baseline/features.pkl"
OUT  = f"{BASE}/classical_ml_baseline/radial_crystal_system.png"

# ── Load ──────────────────────────────────────────────────────────────────────
with open(PKL, "rb") as f:
    d = pickle.load(f)

X    = d["X"]
cols = d["cols"]
bg   = d["bg"]
gt   = d["gt"]    # 0=Direct 1=Indirect 2=Metal
eh   = d["eh"]

cs_int = X[:, cols.index("crystal_system_int")].astype(int)

CS_MAP = {
    1: "Triclinic",
    2: "Monoclinic",
    3: "Orthorhombic",
    4: "Tetragonal",
    5: "Trigonal",
    6: "Hexagonal",
    7: "Cubic",
}
CS_ORDER = [1, 2, 3, 4, 5, 6, 7]

# ── Colours ───────────────────────────────────────────────────────────────────
GAP_COLOR = {
    0: "#E64B35",   # Direct   – red
    1: "#3C5488",   # Indirect – deep blue
    2: "#BBBBBB",   # Metal    – grey
}
CS_BRANCH_COLOR = {
    1: "#B09C85",
    2: "#4DBBD5",
    3: "#00A087",
    4: "#E64B35",
    5: "#3C5488",
    6: "#F39B7F",
    7: "#DC0000",
}

# ── Group materials by crystal system ─────────────────────────────────────────
groups = defaultdict(list)
for i, cs in enumerate(cs_int):
    groups[cs].append(i)

# ── Layout parameters ─────────────────────────────────────────────────────────
R_INNER   = 0.18   # radius of center circle (fraction of figure)
R_BRANCH  = 0.38   # length of main branch line
R_FAN_MIN = 0.45   # start of fan nodes
R_FAN_MAX = 0.90   # end of fan nodes
DOT_SIZE  = 18
LINE_LW   = 0.35
LINE_ALPHA = 0.35

fig, ax = plt.subplots(figsize=(12, 12), facecolor="white")
ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)

# ── Sector angles: proportional to group size ─────────────────────────────────
counts   = [len(groups[cs]) for cs in CS_ORDER]
total    = sum(counts)
gap_deg  = 2.5                          # gap between sectors (degrees)
usable   = 360 - gap_deg * len(CS_ORDER)
sector_degs = [usable * c / total for c in counts]

# Starting angle (top, going clockwise ≡ counter-clockwise in math)
start = 90.0
sector_starts = []
for sd in sector_degs:
    sector_starts.append(start)
    start -= (sd + gap_deg)

# ── Draw center circle ────────────────────────────────────────────────────────
center_circle = plt.Circle((0, 0), R_INNER, color="#2C3E50", zorder=5)
ax.add_patch(center_circle)
ax.text(0, 0.025, "Chalcohalide", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=6)
ax.text(0, -0.04, "Dataset", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=6)
ax.text(0, -0.10, f"n = {total}", ha="center", va="center",
        fontsize=8, color="#AACCEE", zorder=6)

# ── Draw each crystal system sector ───────────────────────────────────────────
for idx, cs in enumerate(CS_ORDER):
    idxs  = groups[cs]
    n     = len(idxs)
    if n == 0:
        continue

    s_deg = sector_starts[idx]
    w_deg = sector_degs[idx]
    mid_deg = s_deg - w_deg / 2          # sector midpoint angle
    mid_rad = np.deg2rad(mid_deg)

    bc = CS_BRANCH_COLOR[cs]

    # Main branch line (center → branch tip)
    bx = R_BRANCH * np.cos(mid_rad)
    by = R_BRANCH * np.sin(mid_rad)
    ax.plot([0, bx], [0, by], color=bc, lw=1.4, alpha=0.7, zorder=2)

    # Crystal system label (just beyond branch tip)
    lx = (R_BRANCH + 0.07) * np.cos(mid_rad)
    ly = (R_BRANCH + 0.07) * np.sin(mid_rad)
    ha = "left" if np.cos(mid_rad) >= 0 else "right"
    ax.text(lx, ly, f"{CS_MAP[cs]}\n(n={n})",
            ha=ha, va="center", fontsize=8.5, fontweight="bold",
            color=bc, zorder=6)

    # Fan out individual materials
    if n == 1:
        angles = [mid_deg]
    else:
        fan_half = min(w_deg / 2 * 0.85, 35)
        angles = np.linspace(mid_deg + fan_half, mid_deg - fan_half, n)

    # Sort by bandgap for visual ordering
    order = np.argsort(bg[idxs])
    sorted_idxs = [idxs[o] for o in order]

    for j, (angle, mat_i) in enumerate(zip(angles, sorted_idxs)):
        # Stagger radius slightly for dense groups
        r = R_FAN_MIN + (R_FAN_MAX - R_FAN_MIN) * (j / max(n - 1, 1))
        ar = np.deg2rad(angle)
        mx = r * np.cos(ar)
        my = r * np.sin(ar)

        # Thin line from branch base to material
        ax.plot([bx, mx], [by, my],
                color=bc, lw=LINE_LW, alpha=LINE_ALPHA, zorder=1)

        # Material dot
        mc = GAP_COLOR[gt[mat_i]]
        stable = eh[mat_i] < 0.1
        marker = "o" if stable else "^"
        ax.scatter(mx, my, s=DOT_SIZE, c=mc, marker=marker,
                   alpha=0.80, linewidths=0, zorder=3)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=GAP_COLOR[0], edgecolor="none", label=f"Direct gap"),
    mpatches.Patch(facecolor=GAP_COLOR[1], edgecolor="none", label=f"Indirect gap"),
    mpatches.Patch(facecolor=GAP_COLOR[2], edgecolor="none", label=f"Metal"),
    plt.scatter([], [], marker="o",  color="#555", s=30, label="Stable ($E_{hull}$<0.1 eV/atom)"),
    plt.scatter([], [], marker="^",  color="#555", s=30, label="Unstable"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.02), ncol=5,
          fontsize=8.5, frameon=False)

ax.set_title("Structural diversity of the chalcohalide dataset\nby crystal system",
             fontsize=13, fontweight="bold", pad=10, y=1.01)

fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")

# ── Stats ─────────────────────────────────────────────────────────────────────
for cs in CS_ORDER:
    n = len(groups[cs])
    print(f"  {CS_MAP[cs]:<14}: {n:>4}  ({100*n/total:.1f}%)")
