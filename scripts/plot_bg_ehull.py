"""
Bandgap (x) vs Ehull (y) — v2
Left : training set  – SpacegroupAnalyzer crystal system colours
Right: prediction set – stability colours (Stable/Unstable)
         y-axis = jittered categorical (above/below 0.01 eV/atom threshold)
"""
import os, csv, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pymatgen.core import Structure

BASE          = "/path/to/Dual-backbone-Graph-Fusion-Network"
TRAIN_ID_CSV  = os.path.join(BASE, "Data/multitask/id_prop.csv")
TRAIN_CIF_DIR = os.path.join(BASE, "Data/multitask")
PRED_CSV      = os.path.join(BASE, "predictions_jacs_final_relaxed.csv")
OUT           = os.path.join(BASE, "plot_bg_ehull.png")

# Nature NPG crystal system colours (for training)
CS_ORDER  = ["Triclinic","Monoclinic","Orthorhombic","Tetragonal","Trigonal","Hexagonal","Cubic"]
CS_COLORS = {
    "Triclinic":    "#B09C85",
    "Monoclinic":   "#4DBBD5",
    "Orthorhombic": "#00A087",
    "Tetragonal":   "#E64B35",
    "Trigonal":     "#3C5488",
    "Hexagonal":    "#F39B7F",
    "Cubic":        "#DC0000",
}
# Prediction stability colours
STAB_COLORS = {"Stable": "#00A087", "Unstable": "#E64B35"}

def sg_to_cs(sg):
    if sg <= 2:   return "Triclinic"
    if sg <= 15:  return "Monoclinic"
    if sg <= 74:  return "Orthorhombic"
    if sg <= 142: return "Tetragonal"
    if sg <= 167: return "Trigonal"
    if sg <= 194: return "Hexagonal"
    return "Cubic"

# ── Training ──────────────────────────────────────────────────────────────────
print("Loading training data …")
rows = {}
with open(TRAIN_ID_CSV) as f:
    for r in csv.reader(f):
        if r: rows[r[0].strip()] = r

tr_bg, tr_ehull, tr_cs = [], [], []
for i, (cid, r) in enumerate(rows.items()):
    cif = os.path.join(TRAIN_CIF_DIR, f"{cid}.cif")
    if not os.path.exists(cif): continue
    try:
        s = Structure.from_file(cif)
        _, sg = s.get_space_group_info()
        tr_bg.append(float(r[1]))
        tr_ehull.append(float(r[3]))
        tr_cs.append(sg_to_cs(sg))
    except Exception:
        pass
    if (i+1) % 500 == 0:
        print(f"  Train {i+1}/{len(rows)}")

print(f"  Training done: {len(tr_bg)} structures")

# ── Prediction ────────────────────────────────────────────────────────────────
print("Loading prediction data …")
pred_df = pd.read_csv(PRED_CSV)
# Filter outliers: keep bg_eV in [0, 10]
pred_df = pred_df[(pred_df["bg_eV"] >= 0) & (pred_df["bg_eV"] <= 10)].copy()
print(f"  Prediction after filter: {len(pred_df)} structures")

# Jitter y-axis encoding: Stable → [0, 0.008], Unstable → [0.014, 0.25]
rng = np.random.default_rng(42)
stable = pred_df["ehull"].str.lower() == "stable"
y_pred = np.where(
    stable,
    rng.uniform(0.000, 0.008, len(pred_df)),
    rng.uniform(0.014, 0.25,  len(pred_df)),
)
pred_df["ehull_y"] = y_pred

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("white")

# --- Left: training ---
ax = axes[0]
present_cs = [cs for cs in CS_ORDER if cs in tr_cs]
for cs in present_cs:
    idx = [i for i, c in enumerate(tr_cs) if c == cs]
    ax.scatter([tr_bg[i] for i in idx], [tr_ehull[i] for i in idx],
               c=CS_COLORS[cs], s=6, alpha=0.5, linewidths=0, label=cs, rasterized=True)
ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Bandgap (eV)", fontsize=12)
ax.set_ylabel("Energy above hull (eV/atom)", fontsize=12)
ax.set_title(f"Training Set  (n = {len(tr_bg):,})", fontsize=13, fontweight="bold", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(left=-0.05)

# Legend for training crystal systems
handles_cs = [Patch(facecolor=CS_COLORS[cs], label=cs) for cs in present_cs]
ax.legend(handles=handles_cs, title="Crystal System",
          fontsize=8.5, title_fontsize=9, loc="upper right",
          frameon=True, framealpha=0.85, edgecolor="#cccccc")

# --- Right: prediction ---
ax = axes[1]
for stab, color in STAB_COLORS.items():
    m = pred_df["ehull"].str.lower() == stab.lower()
    ax.scatter(pred_df.loc[m, "bg_eV"], pred_df.loc[m, "ehull_y"],
               c=color, s=4, alpha=0.4, linewidths=0,
               label=f"{stab} (n={m.sum():,})", rasterized=True)
ax.axhline(0.010, color="gray", lw=0.8, ls="--", alpha=0.6)
ax.set_yticks([0.004, 0.010, 0.13])
ax.set_yticklabels(["Stable\n(EH < 0.01)", "threshold\n0.01 eV/atom", "Unstable\n(EH ≥ 0.01)"],
                    fontsize=8.5)
ax.set_xlabel("Predicted Bandgap (eV)", fontsize=12)
ax.set_ylabel("Predicted Stability", fontsize=12)
ax.set_title(f"Prediction Set  (n = {len(pred_df):,})", fontsize=13, fontweight="bold", pad=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(left=-0.05)
ax.legend(fontsize=9, loc="upper right",
          frameon=True, framealpha=0.85, edgecolor="#cccccc")

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {OUT}")
