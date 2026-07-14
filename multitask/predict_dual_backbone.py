#!/usr/bin/env python3
"""
predict_dual_backbone.py — Ensemble inference on POSCAR/VASP/CIF structures
using the DualBackbone MACE+ALIGNN model (5-fold ensemble).

Usage:
    python predict_dual_backbone.py <input_dir> [output_csv]

Input dir : directory with .vasp / .cif / POSCAR files
Output CSV: predictions for BG, gap-type, EH stability (default: predictions_dual_backbone.csv)
"""

import os, sys, csv, glob, warnings, argparse
import numpy as np
import torch
import torch.nn.functional as F
import ase.io
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT    = "/path/to/Dual-backbone-Graph-Fusion-Network"
CKPT_DIR   = os.path.join(PROJECT, "checkpoints/dual_backbone")
ALIGNN_DIR = os.path.join(PROJECT, "checkpoints/alignn_mt")
sys.path.insert(0, os.path.join(PROJECT, "multitask"))

# ── Graph hyperparams (match training) ────────────────────────────────────────
R_MAX_MACE   = 6.0
RADIUS_ALIGNN = 8.0
MAX_NUM_NBR  = 12
DIST_BINS    = 80
ANGLE_BINS   = 40
EH_THRESHOLD = 0.01   # training label threshold (EH < 0.01 → stable)

# ── Model hyperparams (match training, from checkpoint args) ──────────────────
ALIGNN_KWARGS = dict(
    hidden_dim=256, n_alignn_layers=4, n_gcn_layers=4,
    dropout=0.3, edge_input_dim=DIST_BINS, angle_input_dim=ANGLE_BINS,
    n_gap_classes=2, n_eh_classes=2,
)
DUAL_KWARGS = dict(
    h_fea_len=256, n_attn_heads=8, dropout=0.3,
    n_gap_classes=2, n_eh_classes=2,
    use_cosine_classifier=True, cosine_temp=0.1,
)
N_FOLDS = 5

# ── Label maps ────────────────────────────────────────────────────────────────
# Model class 0 = Direct, class 1 = Indirect+Metal (merge_metal_indirect=True).
# Safeguard: if class-0 predicted but BG < threshold, model likely mis-classified Metal → force Indirect.
METAL_BG_THRESH = 0.05   # eV; below this treat as metallic
EH_LABELS = {0: "Stable(EH<0.01)", 1: "Unstable(EH>=0.01)"}


def resolve_gt_label(gt_pred_idx: int, bg_mean: float) -> str:
    """Map class-0 (Direct) / class-1 (Indirect+Metal) to label strings.
Safeguard: class-0 with BG < threshold is likely Metal, forced to Indirect."""
    if gt_pred_idx == 1:
        return "Indirect"
    # class 0 = Direct; safeguard for model mis-classifying Metal as Direct
    if bg_mean < METAL_BG_THRESH:
        return "Indirect"   # BG too low → likely Metal, treat as Indirect
    return "Direct"


def find_structures(input_dir):
    """Collect all structure files from the directory."""
    files = []
    for pattern in ("*.vasp", "*.cif", "POSCAR", "POSCAR*", "*.poscar"):
        files.extend(glob.glob(os.path.join(input_dir, pattern)))
    # Remove duplicates while preserving order
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return sorted(unique)


def file_to_structures(fpath):
    """Return (name, pmg_Structure, ase_Atoms) from a structure file."""
    name = os.path.splitext(os.path.basename(fpath))[0]
    try:
        struct = Structure.from_file(fpath)
        atoms  = AseAtomsAdaptor.get_atoms(struct)
        return name, struct, atoms
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(fpath)}: {e}")
        return name, None, None


def build_mace_graph(atoms, z_table, r_max, mace_data):
    """Convert ASE Atoms → MACE PyG Data (cached-format, no labels)."""
    from data_dual_backbone import DualGraphData
    keyspec    = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    config     = mace_data.config_from_atoms(atoms, key_specification=keyspec)
    atomic     = mace_data.AtomicData.from_config(
        config, z_table=z_table, cutoff=r_max, heads=["Default"])
    d = atomic.to_dict()
    _MACE_KEYS = ["positions", "edge_index", "shifts", "unit_shifts",
                  "cell", "pbc", "weight", "energy_weight", "forces_weight"]
    dual = DualGraphData()
    for k in _MACE_KEYS:
        if k in d: dual[k] = d[k]
    if "node_attrs" in d:
        dual["mace_node_attrs"] = d["node_attrs"]
    # head must be 1-D [B] (per-graph), not a 0-D scalar
    if "head" in d:
        h = d["head"]
        dual["head"] = h.unsqueeze(0) if h.dim() == 0 else h
    dual.num_nodes = d["positions"].shape[0]
    return dual


def build_alignn_graph(struct, rbf_dist, rbf_angle, dual):
    """Add ALIGNN crystal graph fields to an existing DualGraphData object."""
    from data_crystal_mt import build_crystal_data
    cd = build_crystal_data(struct, rbf_dist, rbf_angle,
                            radius=RADIUS_ALIGNN, max_num_nbr=MAX_NUM_NBR)
    dual.alignn_x              = cd.x
    dual.alignn_edge_index     = cd.edge_index
    dual.alignn_edge_attr      = cd.edge_attr
    dual.alignn_line_edge_index = cd.line_edge_index
    dual.alignn_line_attr      = cd.line_attr
    return dual


def collate_single(dual):
    """Create a batch of 1 from a DualGraphData (add batch + ptr vectors)."""
    n = dual.num_nodes
    dual.batch = torch.zeros(n, dtype=torch.long)
    # ptr: cumulative node counts [0, N] needed by PyG scatter ops
    dual.ptr = torch.tensor([0, n], dtype=torch.long)
    # ALIGNN line-graph batch/ptr
    n_alignn = dual.alignn_x.shape[0]
    dual.alignn_batch = torch.zeros(n_alignn, dtype=torch.long)
    dual.alignn_ptr   = torch.tensor([0, n_alignn], dtype=torch.long)
    return dual


def load_fold_model(fold, mace_model_base, device):
    """Reconstruct DualBackboneMultiTask and load checkpoint weights."""
    from model_dual_backbone   import DualBackboneMultiTask
    from model_alignn_pyg      import ALIGNNMultiTaskPyG

    # ── Load ALIGNN checkpoint and build model ──────────────────────
    alignn_ckpt = torch.load(
        os.path.join(ALIGNN_DIR, f"fold{fold}_best.pt"),
        weights_only=False, map_location=device)
    alignn_model = ALIGNNMultiTaskPyG(**ALIGNN_KWARGS)
    alignn_model.load_state_dict(alignn_ckpt["model_state_dict"])

    # ── Build DualBackbone ──────────────────────────────────────────
    model = DualBackboneMultiTask(mace_model_base, alignn_model, **DUAL_KWARGS)

    dual_ckpt = torch.load(
        os.path.join(CKPT_DIR, f"fold{fold}_best.pt"),
        weights_only=False, map_location=device)
    model.load_state_dict(dual_ckpt["model_state_dict"])
    model.to(device).eval()

    norm_mean = float(dual_ckpt["normalizer_mean"])
    norm_std  = float(dual_ckpt["normalizer_std"])
    return model, norm_mean, norm_std


@torch.no_grad()
def predict_one(model, dual, norm_mean, norm_std, device):
    """Run one forward pass; return (bg_eV, gt_probs[2], eh_probs[2])."""
    dual = dual.to(device)
    bg_out, gt_out, eh_out = model(dual)   # returns tuple (bg, gt, eh)

    # Bandgap: denormalize log-space → eV
    bg_log = bg_out.item() * norm_std + norm_mean
    bg_ev  = float(np.expm1(bg_log))

    # Gap type & EH: log-softmax → probabilities
    gt_probs = torch.exp(gt_out).squeeze(0).cpu().numpy()
    eh_probs = torch.exp(eh_out).squeeze(0).cpu().numpy()

    return bg_ev, gt_probs, eh_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with POSCAR/VASP/CIF files")
    parser.add_argument("output_csv", nargs="?",
                        default="predictions_dual_backbone.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Import model-code modules ─────────────────────────────────────────────
    from mace import data as mace_data
    from mace.calculators import mace_mp
    from data_crystal_mt import RBFExpansion

    # ── Load MACE-MP-0 small for z_table ──────────────────────────────────────
    print("Loading MACE-MP-0 small …")
    calc       = mace_mp(model="small", default_dtype="float32", device=str(device))
    mace_base  = calc.models[0]
    # Build proper ZTable (AtomicNumberTable) — not the raw atomic_numbers tensor
    from mace.tools import AtomicNumberTable
    z_table = AtomicNumberTable([int(z) for z in mace_base.atomic_numbers])

    # ── RBF kernels for ALIGNN ────────────────────────────────────────────────
    rbf_dist  = RBFExpansion(vmin=0.0, vmax=RADIUS_ALIGNN, bins=DIST_BINS)
    rbf_angle = RBFExpansion(vmin=-1.0, vmax=1.0, bins=ANGLE_BINS)

    # ── Load all 5 fold models ────────────────────────────────────────────────
    print(f"Loading {N_FOLDS} fold checkpoints …")
    models, norms = [], []
    for fold in range(N_FOLDS):
        m, nm, ns = load_fold_model(fold, mace_base, device)
        models.append(m)
        norms.append((nm, ns))
        print(f"  fold {fold} loaded (norm mean={nm:.4f}, std={ns:.4f})")

    # ── Find and process structures ───────────────────────────────────────────
    struct_files = find_structures(args.input_dir)
    print(f"\nFound {len(struct_files)} structure files in {args.input_dir}")

    rows = []
    for i, fpath in enumerate(struct_files, 1):
        name, struct, atoms = file_to_structures(fpath)
        if struct is None:
            continue

        try:
            # Build graphs
            dual = build_mace_graph(atoms, z_table, R_MAX_MACE, mace_data)
            dual = build_alignn_graph(struct, rbf_dist, rbf_angle, dual)
            dual = collate_single(dual)
        except Exception as e:
            print(f"  [GRAPH ERROR] {name}: {e}")
            continue

        # Ensemble over folds
        bg_preds, gt_preds, eh_preds = [], [], []
        for fold_idx, (model, (nm, ns)) in enumerate(zip(models, norms)):
            try:
                bg, gt_p, eh_p = predict_one(model, dual, nm, ns, device)
                bg_preds.append(bg)
                gt_preds.append(gt_p)
                eh_preds.append(eh_p)
            except Exception as e:
                print(f"  [PRED ERROR] {name} fold {fold_idx}: {e}")

        if not bg_preds:
            continue

        bg_arr = np.array(bg_preds)
        gt_arr = np.array(gt_preds)   # [n_folds, 2]
        eh_arr = np.array(eh_preds)   # [n_folds, 2]

        bg_mean = float(bg_arr.mean())
        bg_std  = float(bg_arr.std())
        gt_mean = gt_arr.mean(axis=0)  # [2]
        eh_mean = eh_arr.mean(axis=0)  # [2]

        gt_pred_idx = int(gt_mean.argmax())
        eh_pred_idx = int(eh_mean.argmax())

        gt_label = resolve_gt_label(gt_pred_idx, bg_mean)
        is_semi  = bg_mean >= METAL_BG_THRESH
        # screen_pass: semiconductor + stable + direct gap preferred
        screen   = is_semi and eh_pred_idx == 0

        row = {
            "material_id":       name,
            "bg_pred_eV":        round(bg_mean, 4),
            "bg_std_eV":         round(bg_std, 4),
            "gt_pred":           gt_label,
            "gt_prob_direct":    round(float(gt_mean[0]), 4),
            "gt_prob_indirect":  round(float(gt_mean[1]), 4),
            "eh_pred":           EH_LABELS[eh_pred_idx],
            "eh_prob_stable":    round(float(eh_mean[0]), 4),
            "eh_prob_unstable":  round(float(eh_mean[1]), 4),
            "is_semiconductor":  "Yes" if is_semi else "No",
            "screen_pass":       "Yes" if screen else "No",
        }
        rows.append(row)
        print(f"  [{i:3d}/{len(struct_files)}] {name:20s}  "
              f"BG={bg_mean:.3f}±{bg_std:.3f} eV  "
              f"GT={gt_label}  "
              f"EH={EH_LABELS[eh_pred_idx]}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if not rows:
        print("No valid predictions. Check input files.")
        return

    fieldnames = list(rows[0].keys())
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} predictions saved to: {args.output_csv}")

    # ── Summary ───────────────────────────────────────────────────────────────
    bg_all   = [r["bg_pred_eV"] for r in rows]
    n_semi     = sum(1 for r in rows if r["is_semiconductor"] == "Yes")
    n_direct   = sum(1 for r in rows if r["gt_pred"] == "Direct")
    n_indirect = sum(1 for r in rows if r["gt_pred"] == "Indirect")
    n_metal    = sum(1 for r in rows if r["gt_pred"] == "Metal")
    n_stable   = sum(1 for r in rows if r["eh_pred"] == "Stable(EH<0.01)")
    n_pass     = sum(1 for r in rows if r["screen_pass"] == "Yes")

    print(f"\n── Screening Summary ──────────────────────────────")
    print(f"  Total structures:       {len(rows)}")
    print(f"  Semiconductors (BG≥0.05): {n_semi}")
    print(f"    Direct gap:           {n_direct}")
    print(f"    Indirect gap:         {n_indirect}")
    print(f"    Metal (BG≈0):         {n_metal}")
    print(f"  Stable (EH<0.01 eV):   {n_stable}")
    print(f"  PASS (semi + stable):  {n_pass}")
    bg_semi = [r["bg_pred_eV"] for r in rows if r["is_semiconductor"] == "Yes"]
    if bg_semi:
        print(f"  BG range (semi): {min(bg_semi):.3f} – {max(bg_semi):.3f} eV "
              f"(mean {np.mean(bg_semi):.3f} eV)")


if __name__ == "__main__":
    main()
