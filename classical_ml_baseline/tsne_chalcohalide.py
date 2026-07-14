#!/usr/bin/env python3
"""t-SNE of Chalcohalide dataset — 3 separate figures.

  A) tsne_A_bandgap.png   — coloured by bandgap (eV), metals in grey
  B) tsne_B_gaptype.png   — Direct / Indirect / Metal
  C) tsne_C_stability.png — Stable (E_hull < 0.1) / Unstable
"""
import pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
ML   = f"{BASE}/classical_ml_baseline"
PKL  = f"{ML}/features.pkl"
EMB  = f"{ML}/tsne_embedding.npy"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading features …")
with open(PKL, "rb") as f:
    d = pickle.load(f)

X   = d["X"].astype(np.float32)
bg  = d["bg"]   # bandgap eV; gt=2 → bg=0 (metals)
gt  = d["gt"]   # 0=Direct  1=Indirect  2=Metal
eh  = d["eh"]   # energy above hull eV/atom

# ── t-SNE: reuse saved embedding or recompute ─────────────────────────────────
import os
if os.path.exists(EMB):
    X2d = np.load(EMB)
    print(f"Loaded embedding from {EMB}")
else:
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.)
    X_sc  = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=50, random_state=42).fit_transform(X_sc)
    print(f"PCA done → {X_pca.shape}")
    print("Running t-SNE …")
    X2d = TSNE(n_components=2, perplexity=40, max_iter=1000,
               learning_rate="auto", init="pca",
               random_state=42, n_jobs=8).fit_transform(X_pca)
    np.save(EMB, X2d)
    print(f"t-SNE done, saved → {EMB}")

# ── Shared settings ───────────────────────────────────────────────────────────
DIRECT_C   = "#E64B35"
INDIRECT_C = "#3C5488"
METAL_C    = "#AAAAAA"
STABLE_C   = "#00A087"
UNSTABLE_C = "#F39B7F"

S      = 12
ALPHA  = 0.70
FIG_SZ = (5.5, 5.0)   # single-panel size

xlim = (X2d[:, 0].min() - 4, X2d[:, 0].max() + 4)
ylim = (X2d[:, 1].min() - 4, X2d[:, 1].max() + 4)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

def base_ax(ax, panel_label=None):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.grid(True, alpha=0.18, ls="--", lw=0.5, color="#888")
    ax.set_axisbelow(True)

# ── Figure A: bandgap ─────────────────────────────────────────────────────────
is_metal = gt == 2
is_semi  = ~is_metal
bg_semi  = bg[is_semi]
bg_min, bg_max = bg_semi.min(), bg_semi.max()
cmap = plt.cm.plasma_r
norm = Normalize(vmin=bg_min, vmax=bg_max)

fig, ax = plt.subplots(figsize=FIG_SZ)
ax.scatter(X2d[is_metal, 0], X2d[is_metal, 1],
           c=METAL_C, s=S, alpha=0.40, linewidths=0, zorder=1, rasterized=True)
ax.scatter(X2d[is_semi, 0], X2d[is_semi, 1],
           c=bg_semi, cmap=cmap, norm=norm,
           s=S, alpha=ALPHA, linewidths=0, zorder=2, rasterized=True)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    fraction=0.046, pad=0.04, shrink=0.88)
cbar.set_label("Band gap (eV)", fontsize=9.5)
cbar.ax.tick_params(labelsize=9)
metal_patch = mpatches.Patch(facecolor=METAL_C, edgecolor="none",
                              label=f"Metal (n={is_metal.sum()})")
ax.legend(handles=[metal_patch], fontsize=9, loc="lower right",
          framealpha=0.75, edgecolor="none")
ax.set_title("Band gap (eV)", fontsize=11, fontweight="bold", pad=6)
base_ax(ax, "A")
fig.tight_layout()
out_a = f"{ML}/tsne_A_bandgap.png"
fig.savefig(out_a, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_a}")

# ── Figure B: gap type ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIG_SZ)
gt_info = [
    (2, "Metal",    METAL_C,    0.40),
    (1, "Indirect", INDIRECT_C, ALPHA),
    (0, "Direct",   DIRECT_C,   ALPHA),
]
handles_b = []
for code, label, color, alpha in gt_info:
    mask = gt == code
    ax.scatter(X2d[mask, 0], X2d[mask, 1],
               c=color, s=S, alpha=alpha, linewidths=0, zorder=2, rasterized=True)
    handles_b.append(mpatches.Patch(facecolor=color, edgecolor="none",
                                    label=f"{label} (n={mask.sum()})"))
ax.legend(handles=handles_b, fontsize=9, loc="lower right",
          framealpha=0.75, edgecolor="none")
ax.set_title("Gap type", fontsize=11, fontweight="bold", pad=6)
base_ax(ax, "B")
fig.tight_layout()
out_b = f"{ML}/tsne_B_gaptype.png"
fig.savefig(out_b, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_b}")

# ── Figure C: stability ───────────────────────────────────────────────────────
stable   = eh < 0.1
unstable = ~stable

fig, ax = plt.subplots(figsize=FIG_SZ)
ax.scatter(X2d[unstable, 0], X2d[unstable, 1],
           c=UNSTABLE_C, s=S, alpha=0.45, linewidths=0, zorder=1, rasterized=True)
ax.scatter(X2d[stable, 0], X2d[stable, 1],
           c=STABLE_C, s=S, alpha=ALPHA, linewidths=0, zorder=2, rasterized=True)
handles_c = [
    mpatches.Patch(facecolor=STABLE_C,   edgecolor="none",
                   label=f"Stable $E_{{hull}}$<0.1 (n={stable.sum()})"),
    mpatches.Patch(facecolor=UNSTABLE_C, edgecolor="none",
                   label=f"Unstable (n={unstable.sum()})"),
]
ax.legend(handles=handles_c, fontsize=9, loc="lower right",
          framealpha=0.75, edgecolor="none")
ax.set_title("Thermodynamic stability", fontsize=11, fontweight="bold", pad=6)
base_ax(ax, "C")
fig.tight_layout()
out_c = f"{ML}/tsne_C_stability.png"
fig.savefig(out_c, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved → {out_c}")

print("\nAll done.")
