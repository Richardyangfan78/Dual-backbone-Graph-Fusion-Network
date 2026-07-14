"""
MLIP Structure Relaxation using MACE-MP-0
"""
import os, sys, time, traceback, warnings, csv
import numpy as np
warnings.filterwarnings('ignore')

from ase.io import read
from ase.optimize import FIRE
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor

INPUT_DIR  = "/path/to/Dual-backbone-Graph-Fusion-Network/chalcohalide POSCAR"
OUTPUT_DIR = "/path/to/Dual-backbone-Graph-Fusion-Network/Data/predict"
FMAX       = 0.05
MAX_STEPS  = 500
MODEL      = "medium"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading MACE-MP-0 ({MODEL})...")
calc = mace_mp(model=MODEL, dispersion=False, default_dtype="float32", device="cuda")
print("Calculator ready.\n")

vasp_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".vasp"))
print(f"Found {len(vasp_files)} structures to relax.\n")

results, failed = [], []

for i, fname in enumerate(vasp_files):
    mat_id   = fname.replace(".vasp", "")
    in_path  = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, mat_id + ".cif")

    if os.path.exists(out_path):
        print(f"[{i+1}/{len(vasp_files)}] {mat_id}: already exists, skipping.")
        results.append((mat_id, "skipped", None, None))
        continue

    t0 = time.time()
    try:
        atoms = read(in_path)
        atoms.calc = calc
        e_before = atoms.get_potential_energy()

        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=FMAX, steps=MAX_STEPS)

        e_after    = atoms.get_potential_energy()
        n_steps    = opt.get_number_of_steps()
        forces     = atoms.get_forces()
        fmax_final = float(np.sqrt((forces**2).sum(axis=1)).max())
        converged  = fmax_final < FMAX

        pmg_struct = AseAtomsAdaptor().get_structure(atoms)
        pmg_struct.to(filename=out_path)

        dt     = time.time() - t0
        status = "converged" if converged else f"stopped@{n_steps}"
        print(f"[{i+1}/{len(vasp_files)}] {mat_id}: {status} | "
              f"ΔE={e_after-e_before:+.4f} eV | fmax={fmax_final:.4f} | {dt:.1f}s")
        results.append((mat_id, status, round(e_after - e_before, 5), n_steps))

    except Exception as e:
        dt = time.time() - t0
        print(f"[{i+1}/{len(vasp_files)}] {mat_id}: FAILED — {e} ({dt:.1f}s)")
        traceback.print_exc()
        failed.append((mat_id, str(e)))
        results.append((mat_id, "FAILED", None, None))

print(f"\n{'='*60}")
n_ok = sum(1 for r in results if r[1] not in ("FAILED", "skipped"))
print(f"Done: {n_ok} relaxed | {sum(1 for r in results if r[1]=='skipped')} skipped | {len(failed)} failed")

with open(os.path.join(OUTPUT_DIR, "relaxation_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["material_id","status","delta_E_eV","n_steps"]); w.writerows(results)
print(f"Summary: {OUTPUT_DIR}/relaxation_summary.csv")
