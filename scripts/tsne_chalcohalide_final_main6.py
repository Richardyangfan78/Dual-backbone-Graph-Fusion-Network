#!/usr/bin/env python3
"""t-SNE of the final main6 predicted chalcohalide set."""
import os
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


BASE = Path("/path/to/Dual-backbone-Graph-Fusion-Network")
OUT = BASE / "classical_ml_baseline/final_main6_tsne"
PKL = OUT / "features_final_main6.pkl"
EMB = OUT / "tsne_embedding_final_main6.npy"

print("Loading final main6 features ...", flush=True)
with PKL.open("rb") as f:
    d = pickle.load(f)

X = d["X"].astype(np.float32)
bg = np.asarray(d["bg"], dtype=float)
gt = np.asarray(d["gt"], dtype=int)
eh = np.asarray(d["eh"], dtype=float)

if EMB.exists():
    X2d = np.load(EMB)
    if X2d.shape[0] != X.shape[0]:
        print(f"Ignoring stale embedding {X2d.shape}; expected {X.shape[0]} rows", flush=True)
        X2d = None
    else:
        print(f"Loaded embedding from {EMB}", flush=True)
else:
    X2d = None

if X2d is None:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_sc = StandardScaler().fit_transform(X)
    n_components = min(50, X_sc.shape[0] - 1, X_sc.shape[1])
    X_pca = PCA(n_components=n_components, random_state=42).fit_transform(X_sc)
    print(f"PCA done -> {X_pca.shape}", flush=True)
    print("Running t-SNE ...", flush=True)
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    X2d = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_jobs=n_jobs,
    ).fit_transform(X_pca)
    np.save(EMB, X2d)
    print(f"t-SNE saved -> {EMB}", flush=True)

DIRECT_C = "#E64B35"
INDIRECT_C = "#3C5488"
METAL_C = "#AAAAAA"
STABLE_C = "#00A087"
UNSTABLE_C = "#F39B7F"

S = 8
ALPHA = 0.68
FIG_SZ = (5.5, 5.0)

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
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def base_ax(ax):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.grid(True, alpha=0.18, ls="--", lw=0.5, color="#888")
    ax.set_axisbelow(True)


def save(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved -> {png}", flush=True)
    print(f"Saved -> {pdf}", flush=True)


is_metal = gt == 2
is_semi = ~is_metal
bg_semi = bg[is_semi]
bg_min, bg_max = float(np.nanmin(bg_semi)), float(np.nanmax(bg_semi))
cmap = plt.cm.plasma_r
norm = Normalize(vmin=bg_min, vmax=bg_max)

fig, ax = plt.subplots(figsize=FIG_SZ)
ax.scatter(X2d[is_metal, 0], X2d[is_metal, 1],
           c=METAL_C, s=S, alpha=0.38, linewidths=0, zorder=1, rasterized=True)
ax.scatter(X2d[is_semi, 0], X2d[is_semi, 1],
           c=bg_semi, cmap=cmap, norm=norm,
           s=S, alpha=ALPHA, linewidths=0, zorder=2, rasterized=True)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                    fraction=0.046, pad=0.04, shrink=0.88)
cbar.set_label("Predicted band gap (eV)", fontsize=9.5)
cbar.ax.tick_params(labelsize=9)
ax.legend(
    handles=[mpatches.Patch(facecolor=METAL_C, edgecolor="none", label=f"Metal / $E_g$ <= 0 (n={int(is_metal.sum())})")],
    fontsize=9,
    loc="lower right",
    framealpha=0.75,
    edgecolor="none",
)
ax.set_title("Band gap (eV)", fontsize=11, fontweight="bold", pad=6)
base_ax(ax)
fig.tight_layout()
save(fig, "final_main6_tsne_A_bandgap")
plt.close(fig)

fig, ax = plt.subplots(figsize=FIG_SZ)
gt_info = [
    (2, "Metal / $E_g$ <= 0", METAL_C, 0.38),
    (1, "Indirect", INDIRECT_C, ALPHA),
    (0, "Direct", DIRECT_C, ALPHA),
]
handles_b = []
for code, label, color, alpha in gt_info:
    mask = gt == code
    ax.scatter(X2d[mask, 0], X2d[mask, 1],
               c=color, s=S, alpha=alpha, linewidths=0, zorder=2, rasterized=True)
    handles_b.append(mpatches.Patch(facecolor=color, edgecolor="none", label=f"{label} (n={int(mask.sum())})"))
ax.legend(handles=handles_b, fontsize=9, loc="lower right", framealpha=0.75, edgecolor="none")
ax.set_title("Gap type", fontsize=11, fontweight="bold", pad=6)
base_ax(ax)
fig.tight_layout()
save(fig, "final_main6_tsne_B_gaptype")
plt.close(fig)

stable = eh < 0.1
unstable = ~stable
fig, ax = plt.subplots(figsize=FIG_SZ)
ax.scatter(X2d[unstable, 0], X2d[unstable, 1],
           c=UNSTABLE_C, s=S, alpha=0.45, linewidths=0, zorder=1, rasterized=True)
ax.scatter(X2d[stable, 0], X2d[stable, 1],
           c=STABLE_C, s=S, alpha=ALPHA, linewidths=0, zorder=2, rasterized=True)
handles_c = [
    mpatches.Patch(facecolor=STABLE_C, edgecolor="none", label=f"Stable (n={int(stable.sum())})"),
    mpatches.Patch(facecolor=UNSTABLE_C, edgecolor="none", label=f"Unstable (n={int(unstable.sum())})"),
]
ax.legend(handles=handles_c, fontsize=9, loc="lower right", framealpha=0.75, edgecolor="none")
ax.set_title("Thermodynamic stability", fontsize=11, fontweight="bold", pad=6)
base_ax(ax)
fig.tight_layout()
save(fig, "final_main6_tsne_C_stability")
plt.close(fig)

print("All done.", flush=True)
