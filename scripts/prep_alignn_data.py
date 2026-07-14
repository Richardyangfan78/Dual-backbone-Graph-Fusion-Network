#!/usr/bin/env python3
"""Create ALIGNN-compatible data dirs: id_prop.csv with first column '{id}.cif' and symlinks to CIFs."""
import os
import csv
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prep ALIGNN data from CGCNN-style id_prop + CIFs")
    parser.add_argument("--data-root", default=None, help="Root containing Data/bandgap_regression etc.")
    parser.add_argument("--out-root", default=None, help="Output root for Data_alignn (default: data-root/../Data_alignn)")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    data_root = Path(args.data_root or project / "Data")
    out_root = Path(args.out_root or project / "Data_alignn")
    tasks = ["bandgap_regression", "gap_type_classification", "stability_regression"]
    for task in tasks:
        src_dir = data_root / task
        dst_dir = out_root / task
        id_prop_src = src_dir / "id_prop.csv"
        if not id_prop_src.exists():
            print(f"Skip {task}: no {id_prop_src}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        with open(id_prop_src, "r") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                sid = row[0].strip()
                vals = row[1:]
                cif_name = f"{sid}.cif" if not sid.endswith(".cif") else sid
                rows.append([cif_name] + vals)
        id_prop_dst = dst_dir / "id_prop.csv"
        with open(id_prop_dst, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        # Symlink CIFs (same names as in src: mp-xxx.cif)
        for row in rows:
            cif_name = row[0]
            src_cif = src_dir / cif_name
            dst_cif = dst_dir / cif_name
            if src_cif.exists() and not dst_cif.exists():
                dst_cif.symlink_to(os.path.relpath(src_cif, dst_dir))
        print(f"Prepared {task}: {len(rows)} rows -> {dst_dir}")


if __name__ == "__main__":
    main()
