#!/usr/bin/env python3
"""Build matminer features for the final main6 prediction set."""
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    BandCenter,
    ElementProperty,
    IonProperty,
    Stoichiometry,
    TMetalFraction,
    ValenceOrbital,
)
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    StructuralComplexity,
)
from pymatgen.core import Structure


BASE = Path("/path/to/Dual-backbone-Graph-Fusion-Network")
PRED = BASE / "predictions/predictions_final_main6_mace_alignn.csv"
MANIFEST = BASE / "Data/predict_final_main6_manifest.csv"
CIF_DIR = BASE / "Data/predict_final_main6"
OUT = BASE / "classical_ml_baseline/final_main6_tsne"


def _parse_ehull(label):
    text = str(label).strip().lower()
    if text == "stable":
        return 0.0
    if text == "unstable":
        return 0.2
    try:
        return float(label)
    except Exception:
        return np.nan


def _gap_type_code(bg_type, bg_ev):
    if np.isfinite(bg_ev) and bg_ev <= 0:
        return 2, "Metal"
    text = str(bg_type).strip().lower()
    if text == "direct":
        return 0, "Direct"
    if text == "indirect":
        return 1, "Indirect"
    if text == "metal":
        return 2, "Metal"
    return 1, str(bg_type)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = {}
    with MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            manifest[row["formula"]] = row

    pred_rows = []
    with PRED.open(newline="") as f:
        for row in csv.DictReader(f):
            pred_rows.append(row)

    structs = []
    keep_rows = []
    failures = []
    for row in pred_rows:
        formula = row["formula"]
        info = manifest.get(formula, {})
        dest_file = info.get("dest_file") or f"{formula}.cif"
        candidates = [
            CIF_DIR / dest_file,
            CIF_DIR / f"{formula}.cif",
            CIF_DIR / f"{formula}_relaxed.cif",
        ]
        cif_path = next((p for p in candidates if p.exists()), None)
        if cif_path is None:
            matches = sorted(CIF_DIR.glob(f"{formula}*.cif"))
            cif_path = matches[0] if matches else None
        if cif_path is None:
            failures.append((formula, "missing_cif"))
            continue
        try:
            structs.append(Structure.from_file(cif_path))
            out = dict(row)
            out["structure_type"] = info.get("type", "")
            out["source_dir"] = info.get("source_dir", "")
            out["source_file"] = info.get("source_file", "")
            out["dest_file"] = dest_file
            out["cif_path"] = str(cif_path)
            keep_rows.append(out)
        except Exception as exc:
            failures.append((formula, repr(exc)))

    if failures:
        fail_path = OUT / "feature_load_failures.csv"
        with fail_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["formula", "reason"])
            writer.writerows(failures)
        print(f"WARNING: {len(failures)} CIFs failed; details -> {fail_path}", flush=True)

    formulas = [r["formula"] for r in keep_rows]
    bg = np.array([max(float(r["bg_eV"]), 0.0) for r in keep_rows], dtype=float)
    bg_raw = np.array([float(r["bg_eV"]) for r in keep_rows], dtype=float)
    bg_std = np.array([float(r["bg_std_eV"]) for r in keep_rows], dtype=float)
    eh = np.array([_parse_ehull(r["ehull"]) for r in keep_rows], dtype=float)
    gap_info = [_gap_type_code(r["bg_type"], float(r["bg_eV"])) for r in keep_rows]
    gt = np.array([x[0] for x in gap_info], dtype=int)
    gt_label = [x[1] for x in gap_info]
    screen_pass = np.array(
        [str(r["screen_pass"]).strip().lower() == "yes" for r in keep_rows],
        dtype=bool,
    )
    structure_types = [r.get("structure_type", "") for r in keep_rows]

    print(f"loaded structures {len(structs)} / predictions {len(pred_rows)} in {time.time() - t0:.1f}s", flush=True)

    comps = [s.composition for s in structs]
    comp_f = MultipleFeaturizer([
        ElementProperty.from_preset("magpie"),
        Stoichiometry(),
        ValenceOrbital(),
        IonProperty(fast=True),
        TMetalFraction(),
        BandCenter(),
    ])
    struct_f = MultipleFeaturizer([
        DensityFeatures(),
        GlobalSymmetryFeatures(),
        StructuralComplexity(),
    ])
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    for featurizer in (comp_f, struct_f):
        featurizer.set_n_jobs(n_jobs)

    t1 = time.time()
    Xc = comp_f.featurize_many(comps, ignore_errors=True, pbar=False)
    print(f"composition features done {time.time() - t1:.1f}s", flush=True)
    t2 = time.time()
    Xs = struct_f.featurize_many(structs, ignore_errors=True, pbar=False)
    print(f"structure features done {time.time() - t2:.1f}s", flush=True)

    dfc = pd.DataFrame(Xc, columns=comp_f.feature_labels())
    dfs = pd.DataFrame(Xs, columns=struct_f.feature_labels())
    df = pd.concat([dfc, dfs], axis=1)
    df = df.apply(pd.to_numeric, errors="coerce")
    nuniq = df.nunique(dropna=True)
    df = df.loc[:, nuniq > 1]
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"feature matrix {df.shape} | NaN cells {int(df.isna().sum().sum())}", flush=True)

    payload = {
        "ids": formulas,
        "formulas": formulas,
        "X": df.values.astype("float64"),
        "cols": list(df.columns),
        "bg": bg,
        "bg_raw": bg_raw,
        "bg_std": bg_std,
        "gt": gt,
        "gt_label": gt_label,
        "eh": eh,
        "eh_label": [r["ehull"] for r in keep_rows],
        "screen_pass": screen_pass,
        "structure_type": structure_types,
        "metadata": keep_rows,
    }
    with (OUT / "features_final_main6.pkl").open("wb") as f:
        pickle.dump(payload, f)
    df.to_csv(OUT / "features_final_main6.csv", index=False)

    meta = pd.DataFrame(keep_rows)
    meta["bg_eV_raw"] = bg_raw
    meta["bg_eV_used"] = bg
    meta["gap_type_used"] = gt_label
    meta["ehull_numeric_used"] = eh
    meta["screen_pass_bool"] = screen_pass
    meta.to_csv(OUT / "metadata_final_main6.csv", index=False)
    print(f"SAVED -> {OUT}", flush=True)
    print(f"screen_pass_yes {int(screen_pass.sum())}", flush=True)
    print(f"ideal_0p5_1p1 {int(((gt == 0) & (eh < 0.1) & (bg >= 0.5) & (bg <= 1.1)).sum())}", flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
