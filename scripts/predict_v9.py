"""
v9 Ensemble Inference on Relaxed Chalcohalide Structures
-------------------------------------------------------
Same interface as predict_v7; loads v9 checkpoints from checkpoints/multitask_v9.
Uses fold_X_seed_Y/best_composite.pth.tar naming.
"""
import os
import sys
import csv
import glob
import warnings
import json

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = "/path/to/Dual-backbone-Graph-Fusion-Network"
sys.path.insert(0, os.path.join(PROJECT, 'multitask'))
sys.path.insert(0, os.path.join(PROJECT, 'cgcnn'))

from pymatgen.core.structure import Structure
from cgcnn.data import GaussianDistance, AtomCustomJSONInitializer
from model_mt_v7 import CrystalGraphConvNetMTV7
from model_mt_v9 import CrystalGraphConvNetMTV9

# paths
PREDICT_DIR = os.path.join(PROJECT, "Data/predict")
ATOM_INIT = os.path.join(PROJECT, "Data/multitask/atom_init.json")
CKPT_DIR = os.path.join(PROJECT, "checkpoints/multitask_v9")
OUTPUT_CSV = os.path.join(PROJECT, "predictions_v9.csv")

# v9 checkpoint pattern: fold_0_seed_42, fold_1_seed_42, fold_2_seed_179, etc.
CKPT_PATTERN = os.path.join(CKPT_DIR, "fold_*_seed_*", "best_composite.pth.tar")
# Set to "v9" if checkpoints were trained with --model-variant v9
MODEL_VARIANT = "v7"

# model hyperparams (same as v7/v9)
MODEL_KWARGS = dict(
    atom_fea_len=128,
    n_conv=6,
    h_fea_len=256,
    n_h=1,
    n_attn_heads=8,
    dropout=0.3,
    n_gap_classes=2,
    n_eh_classes=2,
)
RADIUS = 8
MAX_NUM_NBR = 12
DMIN, STEP = 0, 0.2
BG_THRESHOLD = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# fallback atom_init if extended not present
if not os.path.exists(ATOM_INIT):
    ATOM_INIT = os.path.join(PROJECT, "Data/multitask/atom_init_original_92dim.json")


class Normalizer:
    def __init__(self):
        self.center = 0.0
        self.scale = 1.0
        self.log_transform = False
        self.robust = False

    def denorm(self, t):
        t = t * self.scale + self.center
        if self.log_transform:
            t = torch.expm1(t)
        return t

    def load_state_dict(self, d):
        self.center = d['center']
        self.scale = d['scale']
        self.log_transform = d.get('log_transform', False)
        self.robust = d.get('robust', False)


ari = AtomCustomJSONInitializer(ATOM_INIT)
gdf = GaussianDistance(dmin=DMIN, dmax=RADIUS, step=STEP)
NBR_FEA_LEN = gdf.filter.shape[0]


def build_graph(structure, mat_id='?'):
    atom_fea = np.vstack([
        ari.get_atom_fea(structure[i].specie.number)
        for i in range(len(structure))
    ])
    atom_fea = torch.Tensor(atom_fea)
    all_nbrs = structure.get_all_neighbors(RADIUS, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < MAX_NUM_NBR:
            warnings.warn(f'{mat_id}: not enough neighbors')
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                              [0] * (MAX_NUM_NBR - len(nbr)))
            nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                          [RADIUS + 1.] * (MAX_NUM_NBR - len(nbr)))
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:MAX_NUM_NBR])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:MAX_NUM_NBR])))
    nbr_fea = gdf.expand(np.array(nbr_fea))
    return (torch.Tensor(atom_fea),
            torch.Tensor(nbr_fea),
            torch.LongTensor(np.array(nbr_fea_idx)))


cif_files = sorted(glob.glob(os.path.join(PREDICT_DIR, "*.cif")))
cif_files = [f for f in cif_files if not f.endswith("summary.cif")]
print(f"Found {len(cif_files)} CIF files for prediction.")

structures = {}
failed_load = []
for cif_path in cif_files:
    mat_id = os.path.splitext(os.path.basename(cif_path))[0]
    try:
        structures[mat_id] = Structure.from_file(cif_path)
    except Exception as e:
        print(f"  WARNING: could not load {mat_id}: {e}")
        failed_load.append(mat_id)
print(f"Loaded {len(structures)} structures ({len(failed_load)} failed).\n")

print("Building crystal graphs...")
graphs = {}
failed_graph = []
for mat_id, struct in structures.items():
    try:
        graphs[mat_id] = build_graph(struct, mat_id)
    except Exception as e:
        print(f"  WARNING: graph build failed for {mat_id}: {e}")
        failed_graph.append(mat_id)
mat_ids = [m for m in sorted(graphs.keys())]
print(f"Graphs built: {len(mat_ids)}  failed: {len(failed_graph)}\n")

# collect v9 checkpoints: one per fold (prefer seed 42 for reproducibility)
ckpt_paths = sorted(glob.glob(CKPT_PATTERN))
fold_ckpts = {}
for p in ckpt_paths:
    parts = os.path.basename(os.path.dirname(p)).split("_")
    fold_num = parts[1]  # fold_X_seed_Y -> X
    seed_num = parts[3]  # seed_Y -> Y
    key = (fold_num, seed_num)
    if fold_num not in fold_ckpts:
        fold_ckpts[fold_num] = []
    fold_ckpts[fold_num].append((seed_num, p))
# take first seed per fold (typically 42)
ckpt_paths = []
for k in sorted(fold_ckpts.keys()):
    fold_list = sorted(fold_ckpts[k], key=lambda x: x[0])
    ckpt_paths.append(fold_list[0][1])
if not ckpt_paths:
    print(f"ERROR: No v9 checkpoints found at {CKPT_PATTERN}")
    sys.exit(1)
print(f"Using {len(ckpt_paths)} checkpoints: {[os.path.basename(os.path.dirname(p)) for p in ckpt_paths]}\n")

all_bg = {m: [] for m in mat_ids}
all_gt = {m: [] for m in mat_ids}
all_eh = {m: [] for m in mat_ids}

for ckpt_path in ckpt_paths:
    ckpt_name = os.path.basename(os.path.dirname(ckpt_path)) + "/best_composite.pth.tar"
    print(f"Loading {ckpt_name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    dummy_fea, dummy_nbr, _ = graphs[mat_ids[0]]
    orig_atom_fea_len = dummy_fea.shape[1]
    nbr_fea_len = dummy_nbr.shape[2]

    model_cls = CrystalGraphConvNetMTV9 if MODEL_VARIANT == "v9" else CrystalGraphConvNetMTV7
    model = model_cls(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        **MODEL_KWARGS,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    norm = Normalizer()
    norm.load_state_dict(ckpt['normalizer_bg'])

    with torch.no_grad():
        for mat_id in mat_ids:
            atom_fea, nbr_fea, nbr_fea_idx = graphs[mat_id]
            af = atom_fea.to(device)
            nf = nbr_fea.to(device)
            ni = nbr_fea_idx.to(device)
            cai = [torch.arange(af.size(0), device=device)]

            p_bg, p_gt, p_eh = model(af, nf, ni, cai)
            bg_val = norm.denorm(p_bg).item()
            gt_prob = F.softmax(p_gt, dim=-1).squeeze(0).cpu().numpy()
            eh_prob = F.softmax(p_eh, dim=-1).squeeze(0).cpu().numpy()

            all_bg[mat_id].append(bg_val)
            all_gt[mat_id].append(gt_prob)
            all_eh[mat_id].append(eh_prob)

    print(f"  Done: {ckpt_name}")

GT_LABELS = {0: "Direct", 1: "Indirect/Metal"}
EH_LABELS = {0: "Stable (EH<0.01)", 1: "Unstable"}

rows = []
print("\nEnsemble predictions:")
print(f"{'Material':<20} {'BG(eV)':>8} {'GT':>16} {'EH':>18}")
print("-" * 68)

for mat_id in mat_ids:
    bg_mean = float(np.mean(all_bg[mat_id]))
    bg_std = float(np.std(all_bg[mat_id]))

    gt_mean_prob = np.mean(all_gt[mat_id], axis=0)
    gt_pred = int(np.argmax(gt_mean_prob))
    gt_conf = float(gt_mean_prob[gt_pred])

    eh_mean_prob = np.mean(all_eh[mat_id], axis=0)
    eh_pred = int(np.argmax(eh_mean_prob))
    eh_conf = float(eh_mean_prob[eh_pred])

    if bg_mean < BG_THRESHOLD:
        gt_pred = 1

    rows.append({
        "material_id": mat_id,
        "bg_eV": round(bg_mean, 4),
        "bg_std_eV": round(bg_std, 4),
        "gap_type": GT_LABELS[gt_pred],
        "gt_confidence": round(gt_conf, 4),
        "eh_stability": EH_LABELS[eh_pred],
        "eh_confidence": round(eh_conf, 4),
    })
    print(f"{mat_id:<20} {bg_mean:>8.3f}+/-{bg_std:.3f}  "
          f"{GT_LABELS[gt_pred]:>14} ({gt_conf:.2f})  "
          f"{EH_LABELS[eh_pred]:>16} ({eh_conf:.2f})")

fieldnames = ["material_id", "bg_eV", "bg_std_eV", "gap_type", "gt_confidence",
              "eh_stability", "eh_confidence"]
with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\nPredictions saved to: {OUTPUT_CSV}")
print(f"Total: {len(rows)} materials predicted.")
if failed_load or failed_graph:
    print(f"Skipped (load failed): {failed_load}")
    print(f"Skipped (graph failed): {failed_graph}")
