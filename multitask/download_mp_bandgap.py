"""
Download Materials Project bandgap dataset for MACE pre-training.

Downloads: material_id, band_gap, is_gap_direct, energy_above_hull, structure (CIF)
Target: ~130k materials with computed bandgap values.
"""
import os
import sys
import csv
import time
import json
import warnings
warnings.filterwarnings("ignore")

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("Usage: python download_mp_bandgap.py <API_KEY>")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/path/to/Dual-backbone-Graph-Fusion-Network/Data/mp_bandgap"
    cif_dir = os.path.join(out_dir, "cifs")
    os.makedirs(cif_dir, exist_ok=True)

    id_prop_file = os.path.join(out_dir, "id_prop.csv")

    # Check for existing progress
    existing_ids = set()
    if os.path.exists(id_prop_file):
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            for row in reader:
                existing_ids.add(row[0])
        print(f"Resuming: {len(existing_ids)} already downloaded")

    print(f"Connecting to Materials Project API...")
    t0 = time.time()

    with MPRester(api_key) as mpr:
        # Query all materials with bandgap data
        # Fields: material_id, band_gap, is_gap_direct, energy_above_hull, structure
        print("Querying MP summary (this may take a few minutes)...")

        docs = mpr.materials.summary.search(
            fields=["material_id", "band_gap", "is_gap_direct",
                     "energy_above_hull", "structure"],
            num_chunks=50,
        )
        print(f"Got {len(docs)} materials in {time.time()-t0:.1f}s")

    # Filter: must have bandgap and structure
    valid = []
    for doc in docs:
        mid = str(doc.material_id)
        bg = doc.band_gap
        is_direct = doc.is_gap_direct
        e_hull = doc.energy_above_hull
        struct = doc.structure

        if bg is None or struct is None or e_hull is None:
            continue
        if is_direct is None:
            is_direct = False  # default to indirect if unknown

        valid.append({
            "material_id": mid,
            "band_gap": bg,
            "gap_type": 1 if is_direct else 0,  # 1=Direct, 0=Indirect
            "energy_above_hull": e_hull,
            "structure": struct,
        })

    print(f"Valid materials: {len(valid)}")
    print(f"  BG=0 (metals): {sum(1 for v in valid if v['band_gap'] == 0)}")
    print(f"  BG>0: {sum(1 for v in valid if v['band_gap'] > 0)}")
    print(f"  Direct: {sum(1 for v in valid if v['gap_type'] == 1)}")
    print(f"  Indirect: {sum(1 for v in valid if v['gap_type'] == 0)}")
    print(f"  Stable (EH<0.01): {sum(1 for v in valid if v['energy_above_hull'] < 0.01)}")

    # Write id_prop.csv and CIF files
    n_new = 0
    n_skip = 0
    n_error = 0

    with open(id_prop_file, "w", newline="") as f:
        writer = csv.writer(f)
        for i, entry in enumerate(valid):
            mid = entry["material_id"]
            writer.writerow([
                mid,
                f"{entry['band_gap']:.6f}",
                entry["gap_type"],
                f"{entry['energy_above_hull']:.6f}",
            ])

            # Write CIF
            cif_path = os.path.join(cif_dir, f"{mid}.cif")
            if mid in existing_ids and os.path.exists(cif_path):
                n_skip += 1
            else:
                try:
                    cw = CifWriter(entry["structure"])
                    cw.write_file(cif_path)
                    n_new += 1
                except Exception as e:
                    n_error += 1
                    if n_error <= 5:
                        print(f"  Error writing {mid}: {e}")

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(valid)} ({n_new} new, {n_skip} skip, {n_error} err) [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"\nDone: {n_new} new CIFs, {n_skip} skipped, {n_error} errors [{elapsed:.0f}s]")
    print(f"id_prop.csv: {len(valid)} entries")
    print(f"Output: {out_dir}")

    # Also create symlinks in out_dir for compatibility
    # (our dataset expects CIFs in the same dir as id_prop.csv)
    print("Creating CIF symlinks...")
    n_link = 0
    for entry in valid:
        mid = entry["material_id"]
        src = os.path.join(cif_dir, f"{mid}.cif")
        dst = os.path.join(out_dir, f"{mid}.cif")
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            n_link += 1
    print(f"Created {n_link} symlinks")


if __name__ == "__main__":
    main()
