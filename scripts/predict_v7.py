"""
v7 Ensemble Inference on Relaxed Chalcohalide Structures
---------------------------------------------------------
Loads 3 best_models_v7 checkpoints, runs ensemble prediction:
  - Band gap (eV): mean of 3 models (denormalized)
  - Gap type:      majority vote (0=Direct, 1=Indirect/Metal)
  - EH stability:  majority vote (0=stable EH<0.01, 1=unstable)
Post-processing: if BG_pred < 0.05 eV -> force GT = 1 (Indirect)
Saves results to predictions_v7.csv
"""
import os, sys, csv, glob, warnings, json
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

# ── paths ─────────────────────────────────────────────────────────────────────
PREDICT_DIR   = os.path.join(PROJECT, "Data/predict")
ATOM_INIT     = os.path.join(PROJECT, "Data/multitask/atom_init_original_92dim.json")
CKPT_DIR      = os.path.join(PROJECT, "checkpoints/best_models_v7")
OUTPUT_CSV    = os.path.join(PROJECT, "predictions_v7.csv")

CKPTS = [
    "fold0_seed42_best_composite.pth.tar",
    "fold1_seed42_best_composite.pth.tar",
    "fold2_seed179_best_composite.pth.tar",
]

# ── model hyperparams (from train_multitask_v7.pbs) ──────────────────────────
MODEL_KWARGS = dict(
    atom_fea_len  = 128,
    n_conv        = 6,
    h_fea_len     = 256,
    n_h           = 1,
    n_attn_heads  = 8,
    dropout       = 0.3,
    n_gap_classes = 2,   # merge_metal_indirect
    n_eh_classes  = 2,
)
RADIUS       = 8
MAX_NUM_NBR  = 12
DMIN, STEP   = 0, 0.2
BG_THRESHOLD = 0.05   # force GT=1 if BG_pred < threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Normalizer ────────────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self):
        self.center = 0.0; self.scale = 1.0
        self.log_transform = False; self.robust = False
    def denorm(self, t):
        t = t * self.scale + self.center
        if self.log_transform:
            t = torch.expm1(t)
        return t
    def load_state_dict(self, d):
        self.center = d['center']; self.scale = d['scale']
        self.log_transform = d.get('log_transform', False)
        self.robust = d.get('robust', False)

# ── graph builder ─────────────────────────────────────────────────────────────
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

def make_batch(atom_fea, nbr_fea, nbr_fea_idx):
    """Wrap single structure into batch format expected by the model."""
    crystal_atom_idx = [torch.arange(atom_fea.size(0))]
    return atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx

# ── load CIF files ────────────────────────────────────────────────────────────
cif_files = sorted(glob.glob(os.path.join(PREDICT_DIR, "*.cif")))
# exclude relaxation_summary if any
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

# ── build graphs ──────────────────────────────────────────────────────────────
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

# ── load models & run inference ───────────────────────────────────────────────
# Collect per-model predictions: bg (float), gt_prob (2), eh_prob (2)
all_bg   = {m: [] for m in mat_ids}
all_gt   = {m: [] for m in mat_ids}
all_eh   = {m: [] for m in mat_ids}

for ckpt_name in CKPTS:
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
    print(f"Loading checkpoint: {ckpt_name}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Determine orig_atom_fea_len and nbr_fea_len from first param
    # (build dummy to get nbr_fea_len)
    dummy_fea, dummy_nbr, _ = graphs[mat_ids[0]]
    orig_atom_fea_len = dummy_fea.shape[1]
    nbr_fea_len       = dummy_nbr.shape[2]

    model = CrystalGraphConvNetMTV7(
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
            af  = atom_fea.to(device)
            nf  = nbr_fea.to(device)
            ni  = nbr_fea_idx.to(device)
            cai = [torch.arange(af.size(0), device=device)]

            p_bg, p_gt, p_eh = model(af, nf, ni, cai)
            bg_val  = norm.denorm(p_bg).item()
            gt_prob = F.softmax(p_gt, dim=-1).squeeze(0).cpu().numpy()
            eh_prob = F.softmax(p_eh, dim=-1).squeeze(0).cpu().numpy()

            all_bg[mat_id].append(bg_val)
            all_gt[mat_id].append(gt_prob)
            all_eh[mat_id].append(eh_prob)

    print(f"  Done: {ckpt_name}")

# ── ensemble ──────────────────────────────────────────────────────────────────
GT_LABELS = {0: "Direct", 1: "Indirect/Metal"}
EH_LABELS = {0: "Stable (EH<0.01)", 1: "Unstable"}

rows = []
print("\nEnsemble predictions:")
print(f"{'Material':<20} {'BG(eV)':>8} {'GT':>16} {'EH':>18}")
print("-" * 68)

for mat_id in mat_ids:
    bg_mean  = float(np.mean(all_bg[mat_id]))
    bg_std   = float(np.std(all_bg[mat_id]))

    gt_mean_prob = np.mean(all_gt[mat_id], axis=0)
    gt_pred      = int(np.argmax(gt_mean_prob))
    gt_conf      = float(gt_mean_prob[gt_pred])

    eh_mean_prob = np.mean(all_eh[mat_id], axis=0)
    eh_pred      = int(np.argmax(eh_mean_prob))
    eh_conf      = float(eh_mean_prob[eh_pred])

    # Post-processing: near-zero BG → force Indirect
    if bg_mean < BG_THRESHOLD:
        gt_pred = 1

    rows.append({
        "material_id"    : mat_id,
        "bg_eV"          : round(bg_mean, 4),
        "bg_std_eV"      : round(bg_std, 4),
        "gap_type"       : GT_LABELS[gt_pred],
        "gt_confidence"  : round(gt_conf, 4),
        "eh_stability"   : EH_LABELS[eh_pred],
        "eh_confidence"  : round(eh_conf, 4),
    })
    print(f"{mat_id:<20} {bg_mean:>8.3f}±{bg_std:.3f}  "
          f"{GT_LABELS[gt_pred]:>14} ({gt_conf:.2f})  "
          f"{EH_LABELS[eh_pred]:>16} ({eh_conf:.2f})")

# ── save ──────────────────────────────────────────────────────────────────────
fieldnames = ["material_id","bg_eV","bg_std_eV","gap_type","gt_confidence",
              "eh_stability","eh_confidence"]
with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)

print(f"\nPredictions saved to: {OUTPUT_CSV}")
print(f"Total: {len(rows)} materials predicted.")
if failed_load or failed_graph:
    print(f"Skipped (load failed): {failed_load}")
    print(f"Skipped (graph failed): {failed_graph}")
