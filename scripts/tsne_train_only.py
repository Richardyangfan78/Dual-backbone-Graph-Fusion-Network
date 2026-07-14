#!/usr/bin/env python3
"""
t-SNE: 仅训练集 (Data/multitask, 2529 structures)
单张图，按晶系配色，Nature NPG palette
"""
import os, csv, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure, Element

PROJECT       = "/path/to/Dual-backbone-Graph-Fusion-Network"
TRAIN_CIF_DIR = os.path.join(PROJECT, "Data/multitask")
ID_PROP_CSV   = os.path.join(PROJECT, "Data/multitask/id_prop.csv")
OUT_FIG       = os.path.join(PROJECT, "tsne_train_only.png")

def sg_to_cs(sg):
    if sg <= 2:   return "Triclinic"
    if sg <= 15:  return "Monoclinic"
    if sg <= 74:  return "Orthorhombic"
    if sg <= 142: return "Tetragonal"
    if sg <= 167: return "Trigonal"
    if sg <= 194: return "Hexagonal"
    return "Cubic"

CS_ORDER = ["Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"]
CS_COLORS = {
    "Triclinic":    "#B09C85",
    "Monoclinic":   "#4DBBD5",
    "Orthorhombic": "#00A087",
    "Tetragonal":   "#E64B35",
    "Trigonal":     "#3C5488",
    "Hexagonal":    "#F39B7F",
    "Cubic":        "#DC0000",
}

# Element property cache
_EP = {}
for sym in Element.__members__:
    try:
        e = Element(sym)
        _EP[e.Z] = (e.X or 2.0, float(e.atomic_radius or 1.4), e.block)
    except Exception:
        pass

def extract_features(struct):
    comp  = struct.composition
    elems = comp.elements
    fracs = [comp.get_atomic_fraction(e) for e in elems]
    ens   = [_EP.get(e.Z, (2., 1.4, 'p'))[0] for e in elems]
    rads  = [_EP.get(e.Z, (2., 1.4, 'p'))[1] for e in elems]
    blocks= [_EP.get(e.Z, (2., 1.4, 'p'))[2] for e in elems]
    mean_Z   = float(np.dot(fracs, [e.Z for e in elems]))
    mean_en  = float(np.dot(fracs, ens))
    mean_rad = float(np.dot(fracs, rads))
    en_diff  = max(ens) - min(ens)
    n_elem   = len(elems)
    bf = {"s":0.,"p":0.,"d":0.,"f":0.}
    for b, f in zip(blocks, fracs):
        if b in bf: bf[b] += f
    lat = struct.lattice
    a, b_, c = lat.a, lat.b, lat.c
    vol_pa  = lat.volume / max(len(struct), 1)
    density = struct.density
    c_a = c/a if a > 0 else 1.
    b_a = b_/a if a > 0 else 1.
    anion_frac = fracs[ens.index(max(ens))]
    return [mean_Z, mean_en, mean_rad, en_diff, n_elem,
            bf["s"], bf["p"], bf["d"], bf["f"],
            vol_pa, density, c_a, b_a, anion_frac]

# ── Load training ─────────────────────────────────────────────────────────────
with open(ID_PROP_CSV) as f:
    train_ids = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
print(f"Training IDs: {len(train_ids)}")

feats, sgs, cs_list, bgs, ehulls = [], [], [], [], []
with open(ID_PROP_CSV) as f:
    rows = {r[0].strip(): r for r in csv.reader(f) if r}

for i, cid in enumerate(train_ids):
    p = os.path.join(TRAIN_CIF_DIR, f"{cid}.cif")
    if not os.path.exists(p): continue
    try:
        s = Structure.from_file(p)
        _, sg = s.get_space_group_info()
        feat = extract_features(s)
        row = rows.get(cid, [])
        feats.append(feat)
        sgs.append(sg)
        cs_list.append(sg_to_cs(sg))
        bgs.append(float(row[1]) if len(row) > 1 else 0.)
        ehulls.append(float(row[3]) if len(row) > 3 else 0.)
    except Exception:
        pass
    if (i+1) % 500 == 0:
        print(f"  {i+1}/{len(train_ids)} loaded …")

n = len(feats)
print(f"Loaded: {n} structures")

# ── t-SNE ─────────────────────────────────────────────────────────────────────
X = StandardScaler().fit_transform(
    np.nan_to_num(np.array(feats, dtype=np.float32), nan=0., posinf=0., neginf=0.))
print(f"Running t-SNE on {n} structures …")
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
            learning_rate="auto", init="pca", random_state=42, n_jobs=4)
X2d = tsne.fit_transform(X)
print("t-SNE done.")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.linewidth": 0.9, "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(7, 6.5))
fig.patch.set_facecolor("white")

present = [cs for cs in CS_ORDER if cs in cs_list]
for cs in present:
    idx = [i for i, c in enumerate(cs_list) if c == cs]
    ax.scatter(X2d[idx, 0], X2d[idx, 1],
               c=CS_COLORS[cs], s=28, alpha=0.72,
               linewidths=0.2, edgecolors="white",
               zorder=2, rasterized=True, label=cs)

ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.set_title(f"Training Set  (n = {n})\nt-SNE coloured by crystal system",
             fontsize=12, fontweight="bold", pad=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5, color="#888")
ax.xaxis.grid(True, alpha=0.2, ls="--", lw=0.5, color="#888")
ax.set_axisbelow(True)

# Legend
handles = [mpatches.Patch(facecolor=CS_COLORS[cs], edgecolor="none", label=cs)
           for cs in present]
ax.legend(handles=handles, title="Crystal system", title_fontsize=10,
          fontsize=9.5, loc="best", frameon=True,
          framealpha=0.85, edgecolor="#cccccc")

# Crystal system count annotation
cnt = Counter(cs_list)
info = "  ".join(f"{cs[:4]}: {cnt[cs]}" for cs in present)
ax.annotate(info, xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=7.5, color="#666", va="bottom")

plt.tight_layout()
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT_FIG}")

cnt = Counter(cs_list)
print(f"\n{'Crystal System':<16} {'n':>6} {'%':>6}")
print("─" * 30)
for cs in CS_ORDER:
    if cs in cnt:
        print(f"{cs:<16} {cnt[cs]:>6} {100*cnt[cs]/n:>5.1f}%")
