#!/usr/bin/env python3
"""
Figure 2: Gate Weight Analysis — composite 2×3 figure
Reads:  results/gate_analysis/gate_analysis_v2.csv
Writes: results/gate_analysis/fig2_gate_composite_v2.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import pearsonr

BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
DATA = os.path.join(BASE, "results/gate_analysis/gate_analysis_v2.csv")
OUT  = os.path.join(BASE, "results/gate_analysis/fig2_gate_composite_v2.png")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.visible": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# Nature NPG palette
C = dict(
    red="#DC0000", blue="#4DBBD5", green="#00A087",
    navy="#3C5488", orange="#F39B7F", purple="#7E6148",
    tan="#B09C85", dark="#333333",
)
CS_ORDER  = ["Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"]
CS_COLORS = {
    "Triclinic":    C["tan"],
    "Monoclinic":   C["blue"],
    "Orthorhombic": C["green"],
    "Tetragonal":   C["red"],
    "Trigonal":     C["navy"],
    "Hexagonal":    C["orange"],
    "Cubic":        "#DC0000",
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
# Encoding: gt_true=1 → Direct, gt_true=0 → Indirect/Metal
#           eh_true=0 → Stable,  eh_true=1 → Unstable
df["gap_label"]  = df["gt_true"].map({0: "Direct", 1: "Indirect/\nMetal"})
df["stab_label"] = df["eh_true"].map({0: "Stable", 1: "Unstable"})
df["cs_cap"]     = df["crystal_system"].str.capitalize()

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 8.5))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                        left=0.07, right=0.97, top=0.91, bottom=0.09)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
LABELS = "abcdef"

def tag(ax, label):
    ax.text(-0.13, 1.06, f"({label})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")

# ── (a) Gate weight distribution ─────────────────────────────────────────────
ax = axes[0]; tag(ax, "a")
g  = df["gate_mean"].values
mu, sd = g.mean(), g.std()
n_bins = 40
ax.hist(g, bins=n_bins, color=C["navy"], alpha=0.80, edgecolor="white", linewidth=0.35)
ylim = ax.get_ylim()
ax.axvline(mu, color=C["dark"], lw=1.6, ls="--", zorder=5)
ax.set_xlim(0.44, 0.78)
ax.set_xlabel("Mean gate weight $\\bar{g}$ (MACE)", fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.set_title("Gate weight distribution", fontsize=9.5, fontweight="bold", pad=6)
ax.text(mu + 0.007, ax.get_ylim()[1] * 0.92,
        f"$\\bar{{g}}$ = {mu:.3f} ± {sd:.3f}\nn = {len(g):,}",
        fontsize=7.5, color=C["dark"], va="top")

# ── (b) By gap type ───────────────────────────────────────────────────────────
ax = axes[1]; tag(ax, "b")
gap_order  = ["Direct", "Indirect/\nMetal"]
gap_colors = [C["red"], C["blue"]]
gap_vals   = [df[df["gap_label"] == gl]["gate_mean"] for gl in gap_order]
means_b = [v.mean() for v in gap_vals]
stds_b  = [v.std()  for v in gap_vals]
ns_b    = [len(v)   for v in gap_vals]
x_b = np.arange(len(gap_order))
bars = ax.bar(x_b, means_b, yerr=stds_b, color=gap_colors, width=0.5,
              capsize=5, error_kw={"elinewidth": 1.1, "ecolor": C["dark"], "capthick": 1.1},
              edgecolor="white", linewidth=0.5)
for i, (m, s, n) in enumerate(zip(means_b, stds_b, ns_b)):
    ax.text(i, m + s + 0.004, f"{m:.3f}\n(n={n:,})",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x_b); ax.set_xticklabels(gap_order, fontsize=8.5)
ax.set_ylabel("Mean gate weight $\\bar{g}$", fontsize=9)
ax.set_title("By bandgap type", fontsize=9.5, fontweight="bold", pad=6)
lo = min(means_b) - max(stds_b) - 0.02
hi = max(means_b) + max(stds_b) + 0.025
ax.set_ylim(lo, hi)

# ── (c) By crystal system ─────────────────────────────────────────────────────
ax = axes[2]; tag(ax, "c")
cs_present = [cs for cs in CS_ORDER if cs in df["cs_cap"].values]
means_c = [df[df["cs_cap"] == cs]["gate_mean"].mean() for cs in cs_present]
stds_c  = [df[df["cs_cap"] == cs]["gate_mean"].std()  for cs in cs_present]
ns_c    = [len(df[df["cs_cap"] == cs])                for cs in cs_present]
colors_c = [CS_COLORS[cs] for cs in cs_present]
x_c = np.arange(len(cs_present))
ax.bar(x_c, means_c, yerr=stds_c, color=colors_c, width=0.6,
       capsize=3.5, error_kw={"elinewidth": 0.9, "ecolor": C["dark"], "capthick": 0.9},
       edgecolor="white", linewidth=0.5)
for i, (m, n) in enumerate(zip(means_c, ns_c)):
    ax.text(i, m + stds_c[i] + 0.003, f"n={n}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x_c)
ax.set_xticklabels([cs[:4] for cs in cs_present], rotation=38, ha="right", fontsize=8)
ax.set_ylabel("Mean gate weight $\\bar{g}$", fontsize=9)
ax.set_title("By crystal system", fontsize=9.5, fontweight="bold", pad=6)
lo_c = min(means_c) - max(stds_c) - 0.02
hi_c = max(means_c) + max(stds_c) + 0.022
ax.set_ylim(lo_c, hi_c)

# ── (d) Heatmap: crystal system × gap type ────────────────────────────────────
ax = axes[3]; tag(ax, "d")
gt_cols = ["Direct", "Indirect/\nMetal"]
heat = np.full((len(cs_present), len(gt_cols)), np.nan)
for i, cs in enumerate(cs_present):
    for j, gl in enumerate(gt_cols):
        sub = df[(df["cs_cap"] == cs) & (df["gap_label"] == gl)]
        if len(sub) >= 3:
            heat[i, j] = sub["gate_mean"].mean()
vmin, vmax = np.nanmin(heat), np.nanmax(heat)
im = ax.imshow(heat, aspect="auto", cmap="RdYlBu_r", vmin=vmin - 0.002, vmax=vmax + 0.002)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Direct", "Indirect/\nMetal"], fontsize=8.5)
ax.set_yticks(range(len(cs_present)))
ax.set_yticklabels(cs_present, fontsize=7.5)
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(0.5)
cbar = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
cbar.set_label("$\\bar{g}$", fontsize=8)
cbar.ax.tick_params(labelsize=7)
for i in range(len(cs_present)):
    for j in range(2):
        if not np.isnan(heat[i, j]):
            ax.text(j, i, f"{heat[i,j]:.3f}", ha="center", va="center",
                    fontsize=6.8, color="black", fontweight="bold")
ax.set_title("Heatmap: crystal system × gap type", fontsize=9.5, fontweight="bold", pad=6)

# ── (e) Scatter: bandgap vs gate weight ───────────────────────────────────────
ax = axes[4]; tag(ax, "e")
mask = df["bg_true"] > 0.05
bg_v = df.loc[mask, "bg_true"].values
gv   = df.loc[mask, "gate_mean"].values
r, p = pearsonr(bg_v, gv)
ax.scatter(bg_v, gv, alpha=0.12, s=3, color=C["navy"], rasterized=True)
m_fit, b_fit = np.polyfit(bg_v, gv, 1)
xline = np.linspace(bg_v.min(), bg_v.max(), 200)
ax.plot(xline, m_fit * xline + b_fit, color=C["red"], lw=1.8, zorder=5)
ax.set_xlabel("DFT bandgap $E_g$ (eV)", fontsize=9)
ax.set_ylabel("Mean gate weight $\\bar{g}$", fontsize=9)
ax.set_title("Gate weight vs bandgap magnitude", fontsize=9.5, fontweight="bold", pad=6)
p_str = f"{p:.1e}" if p >= 1e-99 else "< 10$^{{-99}}$"
ax.text(0.96, 0.96,
        f"$r$ = {r:.3f}\n$p$ = {p_str}\n$n$ = {mask.sum():,}",
        transform=ax.transAxes, fontsize=8, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", lw=0.7))

# ── (f) By stability ──────────────────────────────────────────────────────────
ax = axes[5]; tag(ax, "f")
stab_order  = ["Stable", "Unstable"]
stab_colors = [C["green"], C["red"]]
stab_vals   = [df[df["stab_label"] == sl]["gate_mean"] for sl in stab_order]
means_f = [v.mean() for v in stab_vals]
stds_f  = [v.std()  for v in stab_vals]
ns_f    = [len(v)   for v in stab_vals]
x_f = np.arange(len(stab_order))
ax.bar(x_f, means_f, yerr=stds_f, color=stab_colors, width=0.5,
       capsize=5, error_kw={"elinewidth": 1.1, "ecolor": C["dark"], "capthick": 1.1},
       edgecolor="white", linewidth=0.5)
for i, (m, s, n) in enumerate(zip(means_f, stds_f, ns_f)):
    ax.text(i, m + s + 0.004, f"{m:.3f}\n(n={n:,})",
            ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x_f)
ax.set_xticklabels(["Stable\n(≤0.1 eV/atom)", "Unstable"], fontsize=8.5)
ax.set_ylabel("Mean gate weight $\\bar{g}$", fontsize=9)
ax.set_title("By thermodynamic stability", fontsize=9.5, fontweight="bold", pad=6)
lo_f = min(means_f) - max(stds_f) - 0.02
hi_f = max(means_f) + max(stds_f) + 0.025
ax.set_ylim(lo_f, hi_f)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 2  |  Gate weight analysis of the DBGFN model across 1,768 chalcohalide compounds",
    fontsize=10.5, fontweight="bold", y=0.97
)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUT}")
