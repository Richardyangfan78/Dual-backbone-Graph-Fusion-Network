"""
用 MACE-MP-0 弛豫生成的结构
"""
import os, sys, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np
from ase.io import read
from ase.optimize import FIRE
from mace.calculators import mace_mp
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("output_dir")
parser.add_argument("--device", default="cuda")
parser.add_argument("--fmax", type=float, default=0.05)
parser.add_argument("--max-steps", type=int, default=300)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

print("Loading MACE-MP-0 medium...")
calc = mace_mp(model="medium", dispersion=False, device=args.device, default_dtype="float32")
print("Calculator ready")

files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".cif"))
print("Structures to relax: " + str(len(files)))

success, failed = 0, 0
for i, fname in enumerate(files, 1):
    out_path = os.path.join(args.output_dir, fname)
    if os.path.exists(out_path):
        success += 1
        continue
    formula = fname.replace(".cif","")
    try:
        struct = Structure.from_file(os.path.join(args.input_dir, fname))
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.pbc = True
        atoms.calc = calc
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=args.fmax, steps=args.max_steps)
        relaxed = AseAtomsAdaptor.get_structure(atoms)
        CifWriter(relaxed).write_file(out_path)
        success += 1
        if i % 100 == 0 or i <= 5:
            print("[" + str(i) + "/" + str(len(files)) + "] " + formula + " done")
    except Exception as e:
        failed += 1
        if i <= 10: print("[" + str(i) + "] FAIL " + formula + ": " + str(e)[:60])

print("Success: " + str(success) + "  Failed: " + str(failed))
