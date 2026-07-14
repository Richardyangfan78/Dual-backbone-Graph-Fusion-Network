#!/usr/bin/env python3
"""
Figure 3: Prospective Screening of Hypothetical Chalcohalide Compositions
(a) Screening funnel   (b) Bandgap distribution of 769   (c) Eg vs stability scatter
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from scipy.stats import gaussian_kde

BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
CSV  = os.path.join(BASE, "predictions_jacs_final_relaxed.csv")
OUT  = os.path.join(BASE, "results/fig3_screening.png")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

C = dict(
    red="#DC0000", blue="#4DBBD5", green="#00A087",
    navy="#3C5488", orange="#F39B7F", tan="#B09C85",
    gold="#E6B800", dark="#333333", light="#DDDDDD",
)

# ── Load & compute criteria ───────────────────────────────────────────────────
df = pd.read_csv(CSV)
bg_ok  = (df.bg_eV >= 0.5) & (df.bg_eV <= 1.0)
direct = df.bg_type == "Direct"
stable = df.ehull   == "Stable"
pass3  = bg_ok & direct & stable          # 219 candidates

# Funnel steps
funnel = [
    ("All hypothetical\nchalcohalides",       len(df),          "#AAAAAA"),
    ("Bandgap\n0.5–1.0 eV",                   bg_ok.sum(),      C["blue"]),
    ("+ Direct\nbandgap",                      (bg_ok & direct).sum(), C["navy"]),
    ("+ Thermodynamically\nstable",            pass3.sum(),      C["green"]),
]
funnel_pcts = [100.0] + [100.0 * f[1] / len(df) for f in funnel[1:]]

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.5, 5.2))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                        left=0.06, right=0.97, top=0.88, bottom=0.14,
                        wspace=0.38)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

PANEL_KW = dict(fontsize=11, fontweight="bold", va="top", ha="left")

# ═══════════════════════════════════════════════════════════════════════════════
# (a)  Funnel chart
# ═══════════════════════════════════════════════════════════════════════════════
ax_a.text(-0.14, 1.06, "(a)", transform=ax_a.transAxes, **PANEL_KW)

ax_a.set_xlim(-0.5, 1.5)
ax_a.set_ylim(-0.6, len(funnel) - 0.2)
ax_a.axis("off")

bar_h  = 0.58          # bar height
gap    = 0.42          # gap between bars
max_w  = 1.0           # max bar width (normalised)

for i, (label, n, color) in enumerate(reversed(funnel)):
    y     = i * (bar_h + gap)
    frac  = n / len(df)
    w     = max_w * frac
    xc    = 0.5        # centre
    rect  = Rectangle((xc - w/2, y), w, bar_h,
                       facecolor=color, edgecolor="white",
                       linewidth=0.8, zorder=3, alpha=0.88)
    ax_a.add_patch(rect)

    # count
    ax_a.text(xc, y + bar_h/2, f"{n:,}",
              ha="center", va="center", fontsize=9.5,
              fontweight="bold", color="white", zorder=5,
              path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    # label (right side)
    ax_a.text(xc + w/2 + 0.04, y + bar_h/2, label,
              ha="left", va="center", fontsize=8, color=C["dark"])

    # percentage badge (left)
    if i > 0:
        pct = 100.0 * n / len(df)
        ax_a.text(xc - w/2 - 0.04, y + bar_h/2,
                  f"{pct:.1f}%",
                  ha="right", va="center", fontsize=8,
                  color=color, fontweight="bold")

    # downward arrow between bars
    if i < len(funnel) - 1:
        ax_a.annotate("", xy=(xc, y + bar_h + gap * 0.1),
                      xytext=(xc, y + bar_h + gap * 0.9),
                      arrowprops=dict(arrowstyle="-|>", color=C["dark"],
                                      lw=1.2, mutation_scale=10), zorder=4)

ax_a.set_title("Screening funnel", fontsize=10, fontweight="bold", pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# (b)  Bandgap distribution of 219 candidates
# ═══════════════════════════════════════════════════════════════════════════════
ax_b.text(-0.14, 1.06, "(b)", transform=ax_b.transAxes, **PANEL_KW)

bg_pass = df.loc[pass3, "bg_eV"].values
bins    = np.arange(0.5, 1.05, 0.025)
ax_b.hist(bg_pass, bins=bins, color=C["green"], alpha=0.80,
          edgecolor="white", linewidth=0.5, label="All 219 candidates")


# KDE overlay
xs     = np.linspace(0.45, 1.05, 300)
kde    = gaussian_kde(bg_pass, bw_method=0.25)
ax_b2  = ax_b.twinx()
ax_b2.plot(xs, kde(xs), color=C["red"], lw=2.0)
ax_b2.set_ylabel("Density", fontsize=8, color=C["red"])
ax_b2.tick_params(axis="y", labelcolor=C["red"], labelsize=7)
ax_b2.spines["top"].set_visible(False)

# stats
mu_b = bg_pass.mean(); sd_b = bg_pass.std()
ax_b.axvline(mu_b, color=C["dark"], lw=1.5, ls="--")
ax_b.text(mu_b + 0.05, ax_b.get_ylim()[1] * 0.92 if ax_b.get_ylim()[1] > 0 else 5,
          f"mean = {mu_b:.2f} eV\nSD = {sd_b:.2f} eV",
          fontsize=7.5, color=C["dark"], va="top")

ax_b.set_xlabel("Predicted bandgap $E_g$ (eV)", fontsize=9)
ax_b.set_ylabel("Count", fontsize=9)
ax_b.set_xlim(0.45, 1.05)
ax_b.set_title(f"Bandgap distribution\n(219 multi-criteria candidates)",
               fontsize=10, fontweight="bold", pad=8)

# ═══════════════════════════════════════════════════════════════════════════════
# (c)  Eg vs stability scatter — all 10,687 with 769 highlighted
# ═══════════════════════════════════════════════════════════════════════════════
ax_c.text(-0.14, 1.06, "(c)", transform=ax_c.transAxes, **PANEL_KW)

rng = np.random.default_rng(42)

def jitter(arr, scale=0.08):
    return arr + rng.uniform(-scale, scale, len(arr))

# Assign y: Stable=1, Unstable=0
y_all = np.where(stable.values, 1.0, 0.0)

# Background: Unstable (grey)
mask_unstable = ~stable.values & ~pass3.values
ax_c.scatter(df.loc[mask_unstable, "bg_eV"],
             jitter(y_all[mask_unstable]),
             s=2, alpha=0.15, color=C["light"], rasterized=True, zorder=1)

# Background: Stable, not passing (blue)
mask_stable_no = stable.values & ~pass3.values
ax_c.scatter(df.loc[mask_stable_no, "bg_eV"],
             jitter(y_all[mask_stable_no]),
             s=2, alpha=0.20, color=C["blue"], rasterized=True, zorder=2)

# 219 candidates (red stars)
ax_c.scatter(df.loc[pass3, "bg_eV"],
             jitter(y_all[pass3.values], scale=0.06),
             s=14, alpha=0.75, color=C["red"],
             edgecolors="darkred", linewidths=0.3,
             marker="*", label="219 candidates", zorder=5)

# BG window lines
ax_c.axvline(0.5, color=C["dark"], lw=1.0, ls=":", alpha=0.6)
ax_c.axvline(1.0, color=C["dark"], lw=1.0, ls=":", alpha=0.6)
ax_c.axvspan(0.5, 1.0, alpha=0.04, color=C["green"], zorder=0)

ax_c.set_yticks([0, 1])
ax_c.set_yticklabels(["Unstable", "Stable"], fontsize=9)
ax_c.set_xlabel("Predicted bandgap $E_g$ (eV)", fontsize=9)
ax_c.set_ylabel("Predicted stability", fontsize=9)
ax_c.set_xlim(-0.1, 7.5)
ax_c.set_ylim(-0.35, 1.35)
ax_c.set_title("All 10,687 screened compositions\n(219 candidates highlighted)",
               fontsize=10, fontweight="bold", pad=8)

legend_elements = [
    Patch(facecolor=C["light"],  alpha=0.6, label="Unstable"),
    Patch(facecolor=C["blue"],   alpha=0.6, label="Stable (excl. BG/direct criteria)"),
    plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C["red"],
               markersize=9, markeredgecolor="darkred", label="219 multi-criteria candidates"),
]
ax_c.legend(handles=legend_elements, fontsize=7.5, loc="upper right",
            framealpha=0.85, edgecolor="#cccccc")

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "Figure 3  |  Prospective screening of 10,687 hypothetical chalcohalide compositions (PV-targeted: 0.5–1.0 eV)",
    fontsize=10.5, fontweight="bold", y=0.97,
)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {OUT}")
