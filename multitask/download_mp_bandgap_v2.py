"""
Download targeted Materials Project bandgap dataset for MACE pre-training (v2).

Strategy:
  1. All non-metal materials (BG > 0) — ~60k, core training set for bandgap regression
  2. Chalcogenide & halide materials specifically — maximize chemical similarity
  3. Small sample of metals (BG=0) — boundary learning for GT classification
"""
import os
import sys
import csv
import time
import warnings
warnings.filterwarnings("ignore")

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


def main():
    api_key = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MP_API_KEY", "")
    if not api_key:
        print("Usage: python download_mp_bandgap_v2.py <API_KEY>")
        sys.exit(1)

    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/path/to/Dual-backbone-Graph-Fusion-Network/Data/mp_bandgap_v2"
    cif_dir = os.path.join(out_dir, "cifs")
    os.makedirs(cif_dir, exist_ok=True)

    all_entries = []
    t0 = time.time()

    with MPRester(api_key) as mpr:
        # ─── Query 1: All non-metals (BG > 0) ───
        print("Query 1: Non-metals (band_gap > 0)...")
        docs_nonmetal = mpr.materials.summary.search(
            band_gap=(0.01, None),  # BG > 0.01 eV (exclude metals)
            fields=["material_id", "band_gap", "is_gap_direct",
                     "energy_above_hull", "structure", "elements"],
        )
        print(f"  Got {len(docs_nonmetal)} non-metal materials [{time.time()-t0:.0f}s]")

        for doc in docs_nonmetal:
            mid = str(doc.material_id)
            bg = doc.band_gap
            is_direct = doc.is_gap_direct if doc.is_gap_direct is not None else False
            e_hull = doc.energy_above_hull
            struct = doc.structure
            elements = [str(e) for e in doc.elements] if doc.elements else []

            if bg is None or struct is None or e_hull is None:
                continue

            # Flag chalcogenide/halide related materials
            chalcogens = {"S", "Se", "Te"}
            halogens = {"F", "Cl", "Br", "I"}
            has_chalcogen = bool(chalcogens & set(elements))
            has_halogen = bool(halogens & set(elements))
            is_chalcohalide = has_chalcogen and has_halogen

            all_entries.append({
                "material_id": mid,
                "band_gap": bg,
                "gap_type": 1 if is_direct else 0,
                "energy_above_hull": e_hull,
                "structure": struct,
                "has_chalcogen": has_chalcogen,
                "has_halogen": has_halogen,
                "is_chalcohalide": is_chalcohalide,
            })

        # ─── Query 2: Metals with chalcogen/halogen elements ───
        # (for GT classification boundary learning)
        print("Query 2: Metals with chalcogen/halogen elements...")
        chalco_halo_elements = ["S", "Se", "Te", "F", "Cl", "Br", "I"]
        docs_metals = mpr.materials.summary.search(
            band_gap=(0, 0.01),  # metals
            elements=chalco_halo_elements,
            fields=["material_id", "band_gap", "is_gap_direct",
                     "energy_above_hull", "structure"],
        )
        print(f"  Got {len(docs_metals)} metal materials [{time.time()-t0:.0f}s]")

        existing_ids = {e["material_id"] for e in all_entries}
        n_metals_added = 0
        for doc in docs_metals:
            mid = str(doc.material_id)
            if mid in existing_ids:
                continue
            bg = doc.band_gap
            e_hull = doc.energy_above_hull
            struct = doc.structure
            if bg is None or struct is None or e_hull is None:
                continue

            all_entries.append({
                "material_id": mid,
                "band_gap": bg,
                "gap_type": 0,  # Metal → Indirect
                "energy_above_hull": e_hull,
                "structure": struct,
                "has_chalcogen": True,
                "has_halogen": False,
                "is_chalcohalide": False,
            })
            n_metals_added += 1

        print(f"  Added {n_metals_added} unique metals")

    # ─── Statistics ───
    total = len(all_entries)
    n_direct = sum(1 for e in all_entries if e["gap_type"] == 1)
    n_indirect = total - n_direct
    n_stable = sum(1 for e in all_entries if e["energy_above_hull"] < 0.01)
    n_chalco = sum(1 for e in all_entries if e["has_chalcogen"])
    n_halo = sum(1 for e in all_entries if e["has_halogen"])
    n_chalcohalide = sum(1 for e in all_entries if e["is_chalcohalide"])
    bg_values = [e["band_gap"] for e in all_entries]
    bg_nonzero = [b for b in bg_values if b > 0]

    print(f"\n{'='*60}")
    print(f"Total materials: {total}")
    print(f"  BG > 0 (non-metal): {len(bg_nonzero)}")
    print(f"  BG = 0 (metal): {total - len(bg_nonzero)}")
    print(f"  Direct gap: {n_direct} ({100*n_direct/total:.1f}%)")
    print(f"  Indirect gap: {n_indirect} ({100*n_indirect/total:.1f}%)")
    print(f"  Stable (EH<0.01): {n_stable} ({100*n_stable/total:.1f}%)")
    print(f"  Contains chalcogen: {n_chalco}")
    print(f"  Contains halogen: {n_halo}")
    print(f"  Chalcohalide: {n_chalcohalide}")
    if bg_nonzero:
        import numpy as np
        bga = np.array(bg_nonzero)
        print(f"  BG range (non-metal): {bga.min():.3f} - {bga.max():.3f} eV")
        print(f"  BG mean: {bga.mean():.3f} eV, median: {np.median(bga):.3f} eV")
    print(f"{'='*60}")

    # ─── Write id_prop.csv and CIF files ───
    print(f"\nWriting {total} CIF files...")
    n_new = 0
    n_error = 0

    with open(os.path.join(out_dir, "id_prop.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        for i, entry in enumerate(all_entries):
            mid = entry["material_id"]
            writer.writerow([
                mid,
                f"{entry['band_gap']:.6f}",
                entry["gap_type"],
                f"{entry['energy_above_hull']:.6f}",
            ])

            cif_path = os.path.join(cif_dir, f"{mid}.cif")
            if not os.path.exists(cif_path):
                try:
                    CifWriter(entry["structure"]).write_file(cif_path)
                    n_new += 1
                except Exception as e:
                    n_error += 1
                    if n_error <= 5:
                        print(f"  Error writing {mid}: {e}")

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{total} ({n_new} new, {n_error} err) [{elapsed:.0f}s]")

    # Create symlinks
    print("Creating CIF symlinks...")
    n_link = 0
    for entry in all_entries:
        mid = entry["material_id"]
        src = os.path.join(cif_dir, f"{mid}.cif")
        dst = os.path.join(out_dir, f"{mid}.cif")
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)
            n_link += 1

    elapsed = time.time() - t0
    print(f"\nDone: {n_new} CIFs, {n_error} errors, {n_link} symlinks [{elapsed:.0f}s]")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
