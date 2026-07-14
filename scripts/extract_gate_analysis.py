#!/usr/bin/env python3
"""Extract per-crystal gate weights + full multitask OOF predictions for the
OFFICIAL MACE+ALIGNN dual-backbone checkpoint (dual_backbone_inorg_preAlignSplit_20260702_080248),
evaluated strictly on each fold's own held-out test set (proper OOF, not averaged across folds).

Reuses the proven graph-building / model-loading code from
scripts/predict_inorg_dual_backbone.py (only that script's model-loading path is
known-correct; the sibling multitask/extract_oof_probs.py errors out with a
TypeError on ALIGNNMultiTaskPyG(radius=...), so it is deliberately not used here).
"""
import os, sys, csv, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

PROJECT = "/path/to/Dual-backbone-Graph-Fusion-Network"
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
sys.path.insert(0, os.path.join(PROJECT, "multitask"))

import predict_inorg_dual_backbone as P  # noqa: E402

from sklearn.model_selection import StratifiedKFold
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Override to point at the OFFICIAL "latest reporting" checkpoint per MODEL_STATUS.md,
# NOT the plain checkpoints/dual_backbone_inorg default baked into predict_inorg_dual_backbone.py
P.CKPT_DIR = os.path.join(PROJECT, "checkpoints/dual_backbone_inorg_preAlignSplit_20260702_080248")


def load_fold_model_official(fold, mace_model_base, device):
    """Like predict_inorg_dual_backbone.load_fold_model, but does not depend on
    checkpoints/alignn_mt_inorg/foldN_best.pt (that directory no longer exists
    after the checkpoint reorg). The ALIGNN sub-module is built with fresh
    (random-init) weights purely for architecture scaffolding, then the full
    official dual checkpoint's state_dict (verified to include complete
    alignn_backbone.* + fusion.gate.* keys) is loaded with strict=True, which
    fully overwrites the scaffold and fails loudly on any real mismatch.
    """
    from model_dual_backbone import DualBackboneMultiTask
    from model_alignn_pyg import ALIGNNMultiTaskPyG

    alignn_model = ALIGNNMultiTaskPyG(**P.ALIGNN_KWARGS)
    model = DualBackboneMultiTask(mace_model_base, alignn_model, **P.DUAL_KWARGS)
    dual_ckpt = torch.load(
        os.path.join(P.CKPT_DIR, f"fold{fold}_best.pt"),
        weights_only=False, map_location=device)
    model.load_state_dict(dual_ckpt["model_state_dict"])  # strict=True: must match fully
    model.to(device).eval()

    norm_mean = float(dual_ckpt["normalizer_mean"])
    norm_std = float(dual_ckpt["normalizer_std"])
    return model, norm_mean, norm_std

DATA_DIR = os.path.join(PROJECT, "Data/Inorganic_datasets")
OUT_CSV  = os.path.join(PROJECT, "results/gate_analysis/gate_analysis_v2.csv")
N_FOLDS  = 5
SEED     = 42


def load_labels():
    rows = [r for r in csv.reader(open(os.path.join(DATA_DIR, "id_prop.csv"))) if r]
    ids = [r[0] for r in rows]
    bg = np.array([float(r[1]) for r in rows])
    gt = np.array([int(r[2]) for r in rows])
    eh = np.array([float(r[3]) for r in rows])
    return ids, bg, gt, eh


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print("CKPT_DIR:", P.CKPT_DIR, flush=True)

    ids, bg, gt, eh = load_labels()
    n = len(ids)
    print("Total dataset:", n, flush=True)
    gt_m = np.where(gt == 2, 1, gt).astype(int)   # merge metal(2) -> indirect(1)
    eh_c = (eh >= 0.1).astype(int)                 # 0=stable(<0.1), 1=unstable
    strat = gt_m * 2 + eh_c
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.arange(n), strat))
    print("Fold test sizes:", [len(te) for _, te in folds], flush=True)

    print("Loading MACE-MP-0 small ...", flush=True)
    from mace.calculators import mace_mp
    from mace.tools import AtomicNumberTable
    from mace import data as mace_data
    calc = mace_mp(model="small", default_dtype="float32", device=str(device))
    mace_base = calc.models[0]
    z_table = AtomicNumberTable([int(z) for z in mace_base.atomic_numbers])

    from data_crystal_mt import RBFExpansion
    rbf_dist = RBFExpansion(vmin=0.0, vmax=P.RADIUS_ALIGNN, bins=P.DIST_BINS)
    rbf_angle = RBFExpansion(vmin=-1.0, vmax=1.0, bins=P.ANGLE_BINS)

    all_rows = []
    for fold in range(N_FOLDS):
        print(f"\n--- Fold {fold} ---", flush=True)
        model, norm_mean, norm_std = load_fold_model_official(fold, mace_base, device)

        captured = {}

        def hook(module, inp, out, _c=captured):
            _c["gate"] = out.detach()

        handle = model.fusion.gate.register_forward_hook(hook)

        test_idx = folds[fold][1]
        print(f"  test set size: {len(test_idx)}", flush=True)

        n_ok = 0
        for k in test_idx:
            mp_id = ids[k]
            cif_path = os.path.join(DATA_DIR, f"{mp_id}.cif")
            try:
                struct = Structure.from_file(cif_path)
                atoms = AseAtomsAdaptor.get_atoms(struct)
            except Exception as e:
                print(f"  [SKIP-load] {mp_id}: {e}", flush=True)
                continue

            try:
                dual = P.build_mace_graph(atoms, z_table, P.R_MAX_MACE, mace_data)
                dual = P.build_alignn_graph(struct, rbf_dist, rbf_angle, dual)
                dual = P.collate_single(dual)
                dual = dual.to(device)
            except Exception as e:
                print(f"  [SKIP-graph] {mp_id}: {e}", flush=True)
                continue

            try:
                with torch.no_grad():
                    bg_out, gt_out, eh_out = model(dual)
            except Exception as e:
                print(f"  [SKIP-fwd] {mp_id}: {e}", flush=True)
                continue

            gate = captured.get("gate")
            g_mean = float(gate.mean().item()) if gate is not None else float("nan")

            bg_log_pred = bg_out.item() * norm_std + norm_mean
            bg_pred_ev = float(np.expm1(bg_log_pred))
            gt_probs = torch.exp(gt_out).squeeze(0).cpu().numpy()
            eh_probs = torch.exp(eh_out).squeeze(0).cpu().numpy()
            gt_pred = int(gt_probs.argmax())
            eh_pred = int(eh_probs.argmax())

            gt_true = int(gt_m[k])
            eh_true = int(eh_c[k])
            bg_true_ev = float(bg[k])

            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                crystal_system = sga.get_crystal_system()
            except Exception:
                crystal_system = "unknown"

            all_rows.append({
                "fold": fold, "mp_id": mp_id, "crystal_system": crystal_system,
                "gate_mean": g_mean, "gate_mace": g_mean, "gate_alignn": 1.0 - g_mean,
                "bg_true": round(bg_true_ev, 6), "bg_pred": round(bg_pred_ev, 6),
                "gt_true": gt_true, "gt_pred": gt_pred, "gt_correct": int(gt_true == gt_pred),
                "eh_true": eh_true, "eh_pred": eh_pred, "eh_correct": int(eh_true == eh_pred),
            })
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"  ... {n_ok}/{len(test_idx)} done", flush=True)

        handle.remove()
        print(f"  fold {fold} done: {n_ok}/{len(test_idx)} extracted", flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fieldnames = ["fold", "mp_id", "crystal_system", "gate_mean", "gate_mace", "gate_alignn",
                  "bg_true", "bg_pred", "gt_true", "gt_pred", "gt_correct",
                  "eh_true", "eh_pred", "eh_correct"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows -> {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
