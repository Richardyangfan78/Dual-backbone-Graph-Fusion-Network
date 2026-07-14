#!/usr/bin/env python3
"""
t-SNE: 训练集 (Data/multitask, 2529) vs 预测集 (Data/predict_jacs_relaxed, 10689)
快速版本：
  - 训练集: 使用 pymatgen Structure (需要精确特征)
  - 预测集: 用 regex 解析 CIF header (cell params + composition + SG)，速度提升100x
"""
import os, re, csv, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure, Composition, Element

PROJECT       = "/path/to/Dual-backbone-Graph-Fusion-Network"
TRAIN_CIF_DIR = os.path.join(PROJECT, "Data/multitask")
ID_PROP_CSV   = os.path.join(PROJECT, "Data/multitask/id_prop.csv")
PRED_CIF_DIR  = os.path.join(PROJECT, "Data/predict_jacs_relaxed")
OUT_FIG       = os.path.join(PROJECT, "tsne_spacegroup.png")

# ── Crystal system ────────────────────────────────────────────────────────────
def sg_to_cs(sg):
    if sg is None or sg <= 0: return "Orthorhombic"
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
GRAY = "#CCCCCC"

# ── Shared element property table ─────────────────────────────────────────────
_EP = {}  # Z -> (en, rad, block)
for sym in Element.__members__:
    try:
        e = Element(sym)
        _EP[e.Z] = (e.X or 2.0, float(e.atomic_radius or 1.4), e.block)
    except Exception:
        pass

def _comp_features(comp):
    elems = comp.elements
    fracs = [comp.get_atomic_fraction(e) for e in elems]
    Zs    = [e.Z for e in elems]
    ens   = [_EP.get(e.Z, (2.0, 1.4, 'p'))[0] for e in elems]
    rads  = [_EP.get(e.Z, (2.0, 1.4, 'p'))[1] for e in elems]
    blocks= [_EP.get(e.Z, (2.0, 1.4, 'p'))[2] for e in elems]
    mean_Z   = float(np.dot(fracs, Zs))
    mean_en  = float(np.dot(fracs, ens))
    mean_rad = float(np.dot(fracs, rads))
    en_diff  = max(ens) - min(ens)
    n_elem   = len(elems)
    bf = {"s":0.,"p":0.,"d":0.,"f":0.}
    for b, f in zip(blocks, fracs):
        if b in bf: bf[b] += f
    anion_frac = fracs[ens.index(max(ens))]
    return mean_Z, mean_en, mean_rad, en_diff, n_elem, bf["s"], bf["p"], bf["d"], bf["f"], anion_frac

# ── Training: use pymatgen (need exact lattice/density) ──────────────────────
def extract_features_pymatgen(struct):
    mZ, men, mrad, end, nel, bs, bp, bd, bf, af = _comp_features(struct.composition)
    lat = struct.lattice
    a, b, c = lat.a, lat.b, lat.c
    vol_pa  = lat.volume / max(len(struct), 1)
    density = struct.density
    c_a = c/a if a > 0 else 1.
    b_a = b/a if a > 0 else 1.
    return [mZ, men, mrad, end, nel, bs, bp, bd, bf, vol_pa, density, c_a, b_a, af]

print("Loading training data …")
with open(ID_PROP_CSV) as f:
    train_ids = [r[0].strip() for r in csv.reader(f) if r and r[0].strip()]
print(f"  {len(train_ids)} IDs")

tr_f, tr_sg = [], []
for i, cid in enumerate(train_ids):
    p = os.path.join(TRAIN_CIF_DIR, f"{cid}.cif")
    if not os.path.exists(p): continue
    try:
        s = Structure.from_file(p)
        _, sg = s.get_space_group_info()
        tr_f.append(extract_features_pymatgen(s))
        tr_sg.append(sg)
    except Exception:
        pass
    if (i+1) % 500 == 0:
        print(f"  [Train] {i+1}/{len(train_ids)}")
print(f"  [Train] done: {len(tr_f)} ok")

# ── Prediction: fast regex parsing ────────────────────────────────────────────
_SG_RE  = re.compile(r'_symmetry_Int_Tables_number\s+(\d+)|_space_group_IT_number\s+(\d+)')
_A_RE   = re.compile(r'_cell_length_a\s+([\d.]+)')
_B_RE   = re.compile(r'_cell_length_b\s+([\d.]+)')
_C_RE   = re.compile(r'_cell_length_c\s+([\d.]+)')
_V_RE   = re.compile(r'_cell_volume\s+([\d.]+)')
_Z_RE   = re.compile(r'_cell_formula_units_Z\s+(\d+)')
_FM_RE  = re.compile(r"_chemical_formula_structural\s+'?([^'\n]+)'?|_chemical_formula_sum\s+'?([^'\n]+)'?")
_MASS_RE= re.compile(r'_cell_measurement_temperature|_cell_measurement_reflns')

def parse_cif_fast(cif_path):
    """Extract (sg, a, b, c, volume, Z_formula, formula_str) from CIF header."""
    try:
        with open(cif_path, 'r', errors='ignore') as f:
            txt = f.read(8192)
        sg = None
        m = _SG_RE.search(txt)
        if m: sg = int(m.group(1) or m.group(2))
        a = b = c = vol = None
        ma = _A_RE.search(txt)
        mb = _B_RE.search(txt)
        mc = _C_RE.search(txt)
        mv = _V_RE.search(txt)
        mz = _Z_RE.search(txt)
        if ma: a = float(ma.group(1))
        if mb: b = float(mb.group(1))
        if mc: c = float(mc.group(1))
        if mv: vol = float(mv.group(1))
        elif a and b and c:
            # approximate volume for orthorhombic/cubic (ignore angles)
            vol = a * b * c
        Z_formula = int(mz.group(1)) if mz else 1
        fm = _FM_RE.search(txt)
        formula_str = (fm.group(1) or fm.group(2)).strip() if fm else None
        return sg, a, b, c, vol, Z_formula, formula_str
    except Exception:
        return None, None, None, None, None, 1, None

def extract_features_fast(cif_path, formula_fallback=None):
    sg, a, b, c, vol, Z_f, formula_str = parse_cif_fast(cif_path)
    if formula_str is None: formula_str = formula_fallback
    if formula_str is None: return None, sg
    try:
        comp = Composition(formula_str)
    except Exception:
        return None, sg
    mZ, men, mrad, end, nel, bs, bp, bd, bf, af = _comp_features(comp)
    # lattice features
    a = a or 5.; b = b or 5.; c = c or 5.; vol = vol or 125.
    # estimate n_atoms from Z_formula * num atoms in formula
    n_atoms_cell = sum(int(round(v)) for v in comp.values()) * Z_f
    n_atoms_cell = max(n_atoms_cell, 1)
    vol_pa  = vol / n_atoms_cell
    # estimate density: mass = sum(atomic_mass * count)
    mass_cell = sum(e.atomic_mass * comp[e] for e in comp.elements) * Z_f
    density   = float(mass_cell) / (vol * 1e-24 * 6.022e23) if vol > 0 else 3.0
    c_a = c/a if a > 0 else 1.
    b_a = b/a if a > 0 else 1.
    feat = [mZ, men, mrad, end, nel, bs, bp, bd, bf, vol_pa, density, c_a, b_a, af]
    return feat, sg

print("\nLoading prediction data (fast regex) …")
pred_files = sorted(f for f in os.listdir(PRED_CIF_DIR) if f.endswith(".cif"))
print(f"  {len(pred_files)} CIF files found")

# Load formula fallbacks from CSV
pred_formula_map = {}
pred_csv = os.path.join(PROJECT, "predictions_jacs_final_relaxed.csv")
if os.path.exists(pred_csv):
    with open(pred_csv) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            pred_formula_map[row['formula']] = row['formula']

pr_f, pr_sg = [], []
for i, fn in enumerate(pred_files):
    basename = fn[:-4]  # remove .cif
    feat, sg = extract_features_fast(
        os.path.join(PRED_CIF_DIR, fn),
        formula_fallback=pred_formula_map.get(basename, basename)
    )
    if feat is not None:
        pr_f.append(feat)
        pr_sg.append(sg)
    if (i+1) % 2000 == 0:
        print(f"  [Predict] {i+1}/{len(pred_files)}")
print(f"  [Predict] done: {len(pr_f)} ok")

# ── Joint t-SNE ───────────────────────────────────────────────────────────────
n_tr = len(tr_f);  n_pr = len(pr_f)
all_feats = np.nan_to_num(np.array(tr_f + pr_f, dtype=np.float32),
                           nan=0., posinf=0., neginf=0.)
all_sg    = [sg_to_cs(sg) for sg in (tr_sg + pr_sg)]

X = StandardScaler().fit_transform(all_feats)
print(f"\nt-SNE on {len(X)} structures (train={n_tr}, predict={n_pr}) …")
tsne = TSNE(n_components=2, perplexity=50, max_iter=1000,
            learning_rate="auto", init="pca",
            random_state=42, n_jobs=4)
X2d = tsne.fit_transform(X)
print("t-SNE done.")

X_tr = X2d[:n_tr];  X_pr = X2d[n_tr:]
cs_tr = all_sg[:n_tr];  cs_pr = all_sg[n_tr:]

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.linewidth":0.8,"axes.spines.top":False,
                     "axes.spines.right":False})

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Chemical Space Coverage: Training vs. Prediction (coloured by crystal system)",
             fontsize=11, fontweight="bold", y=1.02)

xlim = (X2d[:,0].min()-3, X2d[:,0].max()+3)
ylim = (X2d[:,1].min()-3, X2d[:,1].max()+3)

def draw_panel(ax, title, hi_X, hi_cs, lo_X, marker, n_lo, lo_label):
    ax.scatter(lo_X[:,0], lo_X[:,1], c=GRAY, s=4, alpha=0.25,
               linewidths=0, zorder=1, rasterized=True, label=lo_label)
    for cs in CS_ORDER:
        idx = [i for i,c in enumerate(hi_cs) if c==cs]
        if not idx: continue
        ax.scatter(hi_X[idx,0], hi_X[idx,1], c=CS_COLORS[cs],
                   marker=marker, s=18 if marker=="o" else 20,
                   alpha=0.65, linewidths=0, zorder=2, rasterized=True)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("t-SNE 1", fontsize=9); ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.yaxis.grid(True, alpha=0.2, ls="--", lw=0.5, color="#888")
    ax.xaxis.grid(True, alpha=0.2, ls="--", lw=0.5, color="#888")
    ax.set_axisbelow(True)

draw_panel(axes[0], f"Training set  (n = {n_tr})",
           X_tr, cs_tr, X_pr, "o", n_pr, f"Prediction (gray, n={n_pr})")
draw_panel(axes[1], f"Prediction set  (n = {n_pr})",
           X_pr, cs_pr, X_tr, "^", n_tr, f"Training (gray, n={n_tr})")

present_cs = [cs for cs in CS_ORDER if cs in cs_tr or cs in cs_pr]
handles = [mpatches.Patch(facecolor=CS_COLORS[cs], edgecolor="none", label=cs)
           for cs in present_cs]
handles.append(mpatches.Patch(facecolor=GRAY, edgecolor="none", label="Other set (background)"))
fig.legend(handles=handles, title="Crystal system", title_fontsize=8.5,
           fontsize=8, loc="lower center", ncol=len(handles),
           bbox_to_anchor=(0.5,-0.08), frameon=False)

plt.tight_layout(rect=[0,0.06,1,1])
fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved → {OUT_FIG}")

cnt_tr = Counter(cs_tr); cnt_pr = Counter(cs_pr)
print(f"\n{'Crystal System':<16} {'Train':>8} {'%':>6}  {'Predict':>9} {'%':>6}")
print("─"*50)
for cs in CS_ORDER:
    t=cnt_tr.get(cs,0); p=cnt_pr.get(cs,0)
    print(f"{cs:<16} {t:>8} {100*t/n_tr:>5.1f}%  {p:>9} "
          f"{100*p/n_pr if n_pr else 0:>5.1f}%")
