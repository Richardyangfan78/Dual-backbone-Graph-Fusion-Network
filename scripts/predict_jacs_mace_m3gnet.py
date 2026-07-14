#!/usr/bin/env python3
"""
predict_jacs_mace_m3gnet.py — MACE+M3GNet DualBackbone inference (eh01 models)
for JACS submission screening.

Output CSV columns:
  formula, bg_type (Direct/Indirect), bg_eV, ehull (Stable/Unstable), screen_pass

screen_pass = Yes iff: Direct AND 0.5 <= bg_eV <= 1.0 AND Stable
"""

import os, sys, csv, glob, warnings, argparse
import numpy as np
import torch
import torch.nn.functional as F
import ase.io
from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT   = "/path/to/Dual-backbone-Graph-Fusion-Network"
CKPT_DIR  = os.path.join(PROJECT, "checkpoints/mace_m3gnet_eh01")
M3G_DIR   = os.path.join(PROJECT, "checkpoints/m3gnet_mt_eh01")
sys.path.insert(0, os.path.join(PROJECT, "multitask"))

# ── Graph hyperparams ─────────────────────────────────────────────────────────
R_MAX_MACE    = 6.0
RADIUS_CRYSTAL = 8.0
MAX_NUM_NBR   = 12
DIST_BINS     = 80
ANGLE_BINS    = 40

# ── Model hyperparams ─────────────────────────────────────────────────────────
M3GNET_KWARGS = dict(
    hidden_dim=128, n_blocks=3, dropout=0.3,
    edge_input_dim=DIST_BINS, angle_input_dim=ANGLE_BINS,
    n_gap_classes=2, n_eh_classes=2,
)
DUAL_KWARGS = dict(
    h_fea_len=256, n_attn_heads=8, dropout=0.3,
    n_gap_classes=2, n_eh_classes=2,
    use_cosine_classifier=True, cosine_temp=0.1,
)
N_FOLDS = 5

# ── Thresholds ────────────────────────────────────────────────────────────────
METAL_BG_THRESH = 0.05
BG_SCREEN_LO    = 0.5
BG_SCREEN_HI    = 1.0


def resolve_gt_label(gt_pred_idx: int, bg_mean: float) -> str:
    if gt_pred_idx == 1:
        return "Indirect"
    if bg_mean < METAL_BG_THRESH:
        return "Indirect"
    return "Direct"


def find_structures(input_dir):
    files = []
    for pattern in ("*.cif", "*.vasp", "POSCAR*", "*.poscar"):
        files.extend(glob.glob(os.path.join(input_dir, pattern)))
    seen, unique = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); unique.append(f)
    return sorted(unique)


def file_to_structures(fpath):
    name = os.path.splitext(os.path.basename(fpath))[0]
    try:
        struct = Structure.from_file(fpath)
        atoms  = AseAtomsAdaptor.get_atoms(struct)
        formula = Composition(struct.formula).reduced_formula
        return name, formula, struct, atoms
    except Exception as e:
        print(f"  [SKIP] {os.path.basename(fpath)}: {e}")
        return name, name, None, None


def build_mace_graph(atoms, z_table, r_max, mace_data):
    from data_dual_backbone import DualGraphData
    keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
    config  = mace_data.config_from_atoms(atoms, key_specification=keyspec)
    atomic  = mace_data.AtomicData.from_config(
        config, z_table=z_table, cutoff=r_max, heads=["Default"])
    d = atomic.to_dict()
    _MACE_KEYS = ["positions", "edge_index", "shifts", "unit_shifts",
                  "cell", "pbc", "weight", "energy_weight", "forces_weight"]
    dual = DualGraphData()
    for k in _MACE_KEYS:
        if k in d: dual[k] = d[k]
    if "node_attrs" in d:
        dual["mace_node_attrs"] = d["node_attrs"]
    if "head" in d:
        h = d["head"]
        dual["head"] = h.unsqueeze(0) if h.dim() == 0 else h
    dual.num_nodes = d["positions"].shape[0]
    return dual


def build_crystal_graph(struct, rbf_dist, rbf_angle, dual):
    from data_crystal_mt import build_crystal_data
    cd = build_crystal_data(struct, rbf_dist, rbf_angle,
                            radius=RADIUS_CRYSTAL, max_num_nbr=MAX_NUM_NBR)
    dual.alignn_x               = cd.x
    dual.alignn_edge_index      = cd.edge_index
    dual.alignn_edge_attr       = cd.edge_attr
    dual.alignn_line_edge_index = cd.line_edge_index
    dual.alignn_line_attr       = cd.line_attr
    return dual


def collate_single(dual):
    n = dual.num_nodes
    dual.batch = torch.zeros(n, dtype=torch.long)
    dual.ptr   = torch.tensor([0, n], dtype=torch.long)
    n_crystal = dual.alignn_x.shape[0]
    dual.alignn_batch = torch.zeros(n_crystal, dtype=torch.long)
    dual.alignn_ptr   = torch.tensor([0, n_crystal], dtype=torch.long)
    return dual


def load_fold_model(fold, mace_model_base, device):
    from model_mace_m3gnet import DualBackboneMACEM3GNet
    from model_m3gnet_pyg  import M3GNetMultiTaskPyG

    m3g_ckpt     = torch.load(
        os.path.join(M3G_DIR, f"fold{fold}_best.pt"),
        weights_only=False, map_location=device)
    m3gnet_model = M3GNetMultiTaskPyG(**M3GNET_KWARGS)
    m3gnet_model.load_state_dict(m3g_ckpt["model_state_dict"])

    model = DualBackboneMACEM3GNet(mace_model_base, m3gnet_model, **DUAL_KWARGS)
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
    dual    = dual.to(device)
    bg_out, gt_out, eh_out = model(dual)
    bg_log  = bg_out.item() * norm_std + norm_mean
    bg_ev   = float(np.expm1(bg_log))
    gt_probs = torch.exp(gt_out).squeeze(0).cpu().numpy()
    eh_probs = torch.exp(eh_out).squeeze(0).cpu().numpy()
    return bg_ev, gt_probs, eh_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_csv", nargs="?",
                        default="predictions_jacs_mace_m3gnet.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    from mace import data as mace_data
    from mace.calculators import mace_mp
    from data_crystal_mt import RBFExpansion

    print("Loading MACE-MP-0 small …")
    calc      = mace_mp(model="small", default_dtype="float32", device=str(device))
    mace_base = calc.models[0]
    from mace.tools import AtomicNumberTable
    z_table = AtomicNumberTable([int(z) for z in mace_base.atomic_numbers])

    rbf_dist  = RBFExpansion(vmin=0.0,  vmax=RADIUS_CRYSTAL, bins=DIST_BINS)
    rbf_angle = RBFExpansion(vmin=-1.0, vmax=1.0,            bins=ANGLE_BINS)

    print(f"Loading {N_FOLDS} fold checkpoints (mace_m3gnet_eh01) …")
    models, norms = [], []
    for fold in range(N_FOLDS):
        m, nm, ns = load_fold_model(fold, mace_base, device)
        models.append(m)
        norms.append((nm, ns))
        print(f"  fold {fold} loaded (norm mean={nm:.4f}, std={ns:.4f})")

    struct_files = find_structures(args.input_dir)
    print(f"\nFound {len(struct_files)} structure files in {args.input_dir}")

    rows = []
    for i, fpath in enumerate(struct_files, 1):
        name, formula, struct, atoms = file_to_structures(fpath)
        if struct is None:
            continue

        try:
            dual = build_mace_graph(atoms, z_table, R_MAX_MACE, mace_data)
            dual = build_crystal_graph(struct, rbf_dist, rbf_angle, dual)
            dual = collate_single(dual)
        except Exception as e:
            print(f"  [GRAPH ERROR] {name}: {e}")
            continue

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

        bg_mean     = float(np.array(bg_preds).mean())
        gt_mean     = np.array(gt_preds).mean(axis=0)
        eh_mean     = np.array(eh_preds).mean(axis=0)
        gt_pred_idx = int(gt_mean.argmax())
        eh_pred_idx = int(eh_mean.argmax())

        bg_type = resolve_gt_label(gt_pred_idx, bg_mean)
        ehull   = "Stable"   if eh_pred_idx == 0 else "Unstable"
        screen  = (bg_type == "Direct" and
                   BG_SCREEN_LO <= bg_mean <= BG_SCREEN_HI and
                   ehull == "Stable")

        rows.append({
            "formula":     formula,
            "bg_type":     bg_type,
            "bg_eV":       round(bg_mean, 4),
            "ehull":       ehull,
            "screen_pass": "Yes" if screen else "No",
        })
        if i % 100 == 0 or i <= 5:
            print(f"  [{i:5d}/{len(struct_files)}] {formula:25s}  "
                  f"BG={bg_mean:.3f} eV  {bg_type:8s}  {ehull}  "
                  f"{'PASS' if screen else ''}")

    if not rows:
        print("No valid predictions.")
        return

    fieldnames = ["formula", "bg_type", "bg_eV", "ehull", "screen_pass"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} predictions → {args.output_csv}")

    n_direct   = sum(1 for r in rows if r["bg_type"] == "Direct")
    n_indirect = sum(1 for r in rows if r["bg_type"] == "Indirect")
    n_stable   = sum(1 for r in rows if r["ehull"] == "Stable")
    n_pass     = sum(1 for r in rows if r["screen_pass"] == "Yes")
    bg_vals    = [r["bg_eV"] for r in rows]
    print(f"\n── Screening Summary ──────────────────")
    print(f"  Total predicted:   {len(rows)}")
    print(f"  Direct gap:        {n_direct}")
    print(f"  Indirect gap:      {n_indirect}")
    print(f"  Stable (EH<0.1):   {n_stable}")
    print(f"  PASS (Direct + 0.5-1.0 eV + Stable): {n_pass}")
    print(f"  BG range: {min(bg_vals):.3f} – {max(bg_vals):.3f} eV")


if __name__ == "__main__":
    main()
