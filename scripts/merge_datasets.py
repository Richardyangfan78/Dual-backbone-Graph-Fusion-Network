#!/usr/bin/env python3
"""
Merge two structure datasets:
  - 576 inorganic structures (VASP format, DFT-relaxed, higher quality)
  - 10,687 hypothetical JACS structures (CIF format)
Rules:
  - Combine both datasets into a single merged CIF directory
  - If same reduced formula appears in both, the inorganic (576) VASP file wins
  - Convert all VASP files to CIF
Output: Data/predict_merged/ with CIF files ready for predict_jacs_nometal.py
"""
import os, shutil, warnings
warnings.filterwarnings("ignore")

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

BASE      = "/path/to/Dual-backbone-Graph-Fusion-Network"
INORG_DIR = os.path.join(BASE, "Data/predict_dft_relaxed")    # 576 VASP files
JACS_DIR  = os.path.join(BASE, "Data/predict_jacs_relaxed")   # 10,687 CIF files
OUT_DIR   = os.path.join(BASE, "Data/predict_merged")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Load all inorganic VASP structures & build formula → (file, struct) map ──
print("Loading 576 inorganic structures (VASP)...")
inorg_map = {}  # reduced_formula -> (src_path, Structure)
inorg_files = sorted(f for f in os.listdir(INORG_DIR) if f.endswith(".vasp"))
failed_inorg = []
for fname in inorg_files:
    fpath = os.path.join(INORG_DIR, fname)
    try:
        struct = Structure.from_file(fpath)
        formula = struct.composition.reduced_formula
        inorg_map[formula] = (fpath, struct)
    except Exception as e:
        failed_inorg.append((fname, str(e)))

print(f"  Loaded: {len(inorg_map)} | Failed: {len(failed_inorg)}")

# ── Step 2: Build JACS formula map ──
print("Scanning 10,687 JACS hypothetical structures (CIF)...")
jacs_files = sorted(f for f in os.listdir(JACS_DIR) if f.endswith(".cif"))
jacs_map = {}  # reduced_formula -> src_path
failed_jacs = []
for fname in jacs_files:
    fpath = os.path.join(JACS_DIR, fname)
    try:
        struct = Structure.from_file(fpath)
        formula = struct.composition.reduced_formula
        jacs_map[formula] = fpath
    except Exception as e:
        failed_jacs.append((fname, str(e)))

print(f"  Loaded: {len(jacs_map)} | Failed: {len(failed_jacs)}")

# ── Step 3: Find overlap ──
overlap = set(inorg_map.keys()) & set(jacs_map.keys())
print(f"\nOverlap (formulas in both): {len(overlap)}")
if overlap:
    for f in sorted(overlap):
        print(f"  {f}")

# ── Step 4: Copy JACS CIFs to merged dir (skip overlapping) ──
print("\nCopying JACS CIFs to merged dir (excluding overlapping formulas)...")
copied_jacs = 0
skipped_jacs = 0
for formula, src in jacs_map.items():
    if formula in inorg_map:
        skipped_jacs += 1
        continue
    fname = os.path.basename(src)
    shutil.copy2(src, os.path.join(OUT_DIR, fname))
    copied_jacs += 1
print(f"  Copied: {copied_jacs} | Skipped (overridden by inorg): {skipped_jacs}")

# ── Step 5: Convert inorganic VASP → CIF and write to merged dir ──
print("Converting 576 inorganic VASP → CIF and writing...")
written_inorg = 0
failed_write = []
for formula, (src, struct) in inorg_map.items():
    fname_base = os.path.splitext(os.path.basename(src))[0]
    out_path = os.path.join(OUT_DIR, fname_base + ".cif")
    try:
        CifWriter(struct).write_file(out_path)
        written_inorg += 1
    except Exception as e:
        failed_write.append((fname_base, str(e)))

print(f"  Written: {written_inorg} | Failed: {len(failed_write)}")

# ── Summary ──
total = len(os.listdir(OUT_DIR))
print(f"\n=== Merge complete ===")
print(f"  JACS copied:         {copied_jacs}")
print(f"  Inorganic converted: {written_inorg}")
print(f"  Total in merged dir: {total}")
print(f"  Overlap replaced:    {len(overlap)}")
print(f"  Output: {OUT_DIR}")
