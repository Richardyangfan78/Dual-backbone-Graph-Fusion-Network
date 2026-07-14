"""
Pre-build MACE and crystal caches for JARVIS chalcohalide entries.
Run this before merged training to populate both cache directories.
"""
import os, sys, csv, warnings, time
import numpy as np
import torch

BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
sys.path.insert(0, os.path.join(BASE, "multitask"))
os.environ.setdefault("HF_HOME",   os.path.join(BASE, ".cache"))
os.environ.setdefault("MACE_CACHE", os.path.join(BASE, ".cache"))

JARVIS_DIR      = os.path.join(BASE, "Data", "jarvis_chalcohalide")
MACE_CACHE_DIR  = os.path.join(BASE, "Data", "multitask", "mace_cached_graphs")
CRYS_CACHE_DIR  = os.path.join(BASE, "Data", "multitask", "crystal_cached_graphs")
MACE_PRETRAINED = os.path.join(BASE, "checkpoints", "mace_pretrain", "pretrained_best.pt")

os.makedirs(MACE_CACHE_DIR, exist_ok=True)
os.makedirs(CRYS_CACHE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load MACE-MP-0 for z_table + r_max ─────────────────────────────
print("Loading MACE-MP-0...")
from mace.calculators import mace_mp
calc = mace_mp(model="small", default_dtype="float32", device=device)
z_table = calc.z_table
r_max   = calc.r_max
print(f"MACE-MP-0 ready, r_max={r_max}")

import ase.io
import mace.data as mace_data
from data_mace_mt import _mace_to_pyg, _MACE_KEYS
from data_crystal_mt import build_crystal_data, RBFExpansion

rbf_dist  = RBFExpansion(0.0, 8.0, 80)
rbf_angle = RBFExpansion(-1.0, 1.0, 40)

# ── Load JARVIS id_prop.csv ─────────────────────────────────────────
rows = []
with open(os.path.join(JARVIS_DIR, "id_prop.csv")) as f:
    for row in csv.reader(f):
        mid = row[0].strip()
        gt_raw = int(row[2])
        if gt_raw == 2:   # skip metals
            continue
        cif_path = os.path.join(JARVIS_DIR, f"{mid}.cif")
        if os.path.exists(cif_path):
            rows.append((mid, cif_path))

print(f"JARVIS non-metal entries to process: {len(rows)}")

# ── Process each entry ──────────────────────────────────────────────
mace_done = crys_done = skipped = 0
t0 = time.time()

for i, (mid, cif_path) in enumerate(rows):
    mace_cache_path = os.path.join(MACE_CACHE_DIR, f"{mid}.pt")
    crys_cache_path = os.path.join(CRYS_CACHE_DIR, f"{mid}.pt")

    need_mace = not os.path.exists(mace_cache_path)
    need_crys = not os.path.exists(crys_cache_path)

    if not need_mace and not need_crys:
        continue

    try:
        if need_mace:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                atoms = ase.io.read(cif_path)
            keyspec = mace_data.KeySpecification(info_keys={}, arrays_keys={})
            config  = mace_data.config_from_atoms(atoms, key_specification=keyspec)
            atomic_data = mace_data.AtomicData.from_config(
                config, z_table=z_table, cutoff=r_max, heads=["Default"]
            )
            pyg_data = _mace_to_pyg(atomic_data)
            tmp = mace_cache_path + f".tmp{os.getpid()}"
            torch.save(pyg_data, tmp)
            os.replace(tmp, mace_cache_path)
            mace_done += 1

        if need_crys:
            from pymatgen.core import Structure
            struct = Structure.from_file(cif_path)
            crys_data = build_crystal_data(struct, rbf_dist, rbf_angle, 8.0, 12, 94)
            tmp = crys_cache_path + f".tmp{os.getpid()}"
            torch.save(crys_data, tmp)
            os.replace(tmp, crys_cache_path)
            crys_done += 1

    except Exception as e:
        print(f"  [SKIP] {mid}: {e}")
        skipped += 1
        continue

    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta  = (len(rows) - i - 1) / max(rate, 1e-6)
        print(f"  [{i+1}/{len(rows)}] mace={mace_done} crys={crys_done} skip={skipped}  "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

print(f"\nDone! MACE caches built: {mace_done}, Crystal caches built: {crys_done}, Skipped: {skipped}")
