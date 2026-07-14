"""
relax_jacs.py — MLIP relaxation of CIF structures using MACE-MP-0 medium
For use as SLURM array job: each task handles a chunk of structures.

Usage:
    python relax_jacs.py --chunk <i> --total-chunks <N> [--fmax 0.05] [--max-steps 500]
"""
import os, sys, time, traceback, warnings, csv, argparse, glob
import numpy as np
warnings.filterwarnings('ignore')

from ase.io import read
from ase.optimize import FIRE
from mace.calculators import mace_mp
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

PROJECT   = "/path/to/Dual-backbone-Graph-Fusion-Network"
INPUT_DIR = os.path.join(PROJECT, "Data/predict_jacs")
OUT_DIR   = os.path.join(PROJECT, "Data/predict_jacs_relaxed")
os.makedirs(OUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--chunk",        type=int, default=0, help="0-based chunk index (SLURM_ARRAY_TASK_ID)")
parser.add_argument("--total-chunks", type=int, default=8, help="Total number of chunks")
parser.add_argument("--fmax",         type=float, default=0.05)
parser.add_argument("--max-steps",    type=int,   default=500)
parser.add_argument("--model",        default="medium")
parser.add_argument("--device",       default="cuda")
args = parser.parse_args()

all_cifs = sorted(glob.glob(os.path.join(INPUT_DIR, "*.cif")))
total    = len(all_cifs)
chunk_sz = (total + args.total_chunks - 1) // args.total_chunks
start    = args.chunk * chunk_sz
end      = min(start + chunk_sz, total)
my_cifs  = all_cifs[start:end]

print(f"Chunk {args.chunk}/{args.total_chunks-1}: structures [{start}:{end}] ({len(my_cifs)} files)")
print(f"Loading MACE-MP-0 ({args.model}) on {args.device}...")
calc = mace_mp(model=args.model, dispersion=False, default_dtype="float32", device=args.device)
print("Calculator ready.\n")

summary_csv = os.path.join(OUT_DIR, f"relax_summary_chunk{args.chunk}.csv")
results = []

for i, cif_path in enumerate(my_cifs, 1):
    name     = os.path.splitext(os.path.basename(cif_path))[0]
    out_path = os.path.join(OUT_DIR, name + ".cif")

    if os.path.exists(out_path):
        print(f"[{i}/{len(my_cifs)}] {name}: already exists, skipping.")
        results.append((name, "skipped", None, None))
        continue

    t0 = time.time()
    try:
        atoms = read(cif_path)
        atoms.calc = calc
        e_before = atoms.get_potential_energy()

        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=args.fmax, steps=args.max_steps)

        e_after   = atoms.get_potential_energy()
        n_steps   = opt.get_number_of_steps()
        forces    = atoms.get_forces()
        fmax_fin  = float(np.sqrt((forces**2).sum(axis=1)).max())
        converged = fmax_fin < args.fmax

        pmg = AseAtomsAdaptor().get_structure(atoms)
        pmg.to(filename=out_path)

        dt     = time.time() - t0
        status = "converged" if converged else f"stopped@{n_steps}"
        print(f"[{i}/{len(my_cifs)}] {name}: {status} | "
              f"ΔE={e_after-e_before:+.4f} eV | fmax={fmax_fin:.4f} | {dt:.1f}s")
        results.append((name, status, round(e_after - e_before, 5), n_steps))

    except Exception as e:
        dt = time.time() - t0
        print(f"[{i}/{len(my_cifs)}] {name}: FAILED — {e} ({dt:.1f}s)")
        traceback.print_exc()
        results.append((name, "FAILED", None, None))

n_ok     = sum(1 for r in results if r[1] not in ("FAILED", "skipped"))
n_skip   = sum(1 for r in results if r[1] == "skipped")
n_fail   = sum(1 for r in results if r[1] == "FAILED")
print(f"\n=== Chunk {args.chunk} done: {n_ok} relaxed | {n_skip} skipped | {n_fail} failed ===")

with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["material_id", "status", "delta_E_eV", "n_steps"])
    w.writerows(results)
print(f"Summary: {summary_csv}")
