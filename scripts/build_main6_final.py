#!/usr/bin/env python3
"""Build the final six-class main prediction set."""

from __future__ import annotations

import csv
import os
import shutil
from collections import Counter, defaultdict
from itertools import product

from pymatgen.core import Composition


BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
FINAL_DIR = os.path.join(BASE, "Data/predict_final_main6")
MANIFEST = os.path.join(BASE, "Data/predict_final_main6_manifest.csv")
MISSING_CSV = os.path.join(BASE, "Data/predict_final_main6_missing.csv")

A = ["Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Au"]
B = ["Mg", "Ca", "Sr", "Ba", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ge", "Sn", "Pd", "Pt"]
C = ["Al", "Ga", "In", "Sc", "Y", "La", "Lu", "Ti", "V", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Fe", "Ru", "Rh", "Bi", "Sb"]
CH = ["S", "Se", "Te"]
X = ["F", "Cl", "Br", "I"]

ROLE_ELEMENTS = {"A": A, "B": B, "C": C}
ROLE_CHARGE = {"A": 1, "B": 2, "C": 3}
SUPPLEMENT_PATTERNS = ["A1C2_Ch3_X1", "B1C1_Ch2_X1", "A1B1_Ch1_X1"]

SOURCE_DIRS = [
    os.path.join(BASE, "Data/predict_main6_supplement_relaxed"),
    os.path.join(BASE, "Data/predict_type3_relaxed"),
    os.path.join(BASE, "Data/predict_type3_relaxed_extra"),
    os.path.join(BASE, "Data/predict_new_relaxed"),
    os.path.join(BASE, "Data/predict_final"),
    os.path.join(BASE, "Data/predict_jacs_relaxed"),
]


def reduced_formula_from_counts(counts: dict[str, int]) -> str:
    return Composition(counts).reduced_formula


def parse_pattern(pattern: str):
    role_part, ch_part, x_part = pattern.split("_")
    slots = []
    i = 0
    while i < len(role_part):
        role = role_part[i]
        i += 1
        j = i
        while j < len(role_part) and role_part[j].isdigit():
            j += 1
        slots.append((role, int(role_part[i:j])))
        i = j
    return tuple(slots), int(ch_part[2:]), int(x_part[1:])


def enumerate_pattern(pattern: str):
    slots, ch_count, x_count = parse_pattern(pattern)
    choices = [ROLE_ELEMENTS[role] for role, _ in slots]
    for elements in product(*choices):
        if len(elements) != len(set(elements)):
            continue
        for ch in CH:
            for x in X:
                counts = {ch: ch_count, x: x_count}
                positive = 0
                for (role, count), element in zip(slots, elements):
                    counts[element] = counts.get(element, 0) + count
                    positive += ROLE_CHARGE[role] * count
                if positive == 2 * ch_count + x_count:
                    yield reduced_formula_from_counts(counts)


def expected_formulas() -> dict[str, str]:
    expected = {}

    # Type1: A C Ch X2
    for a in A:
        for c in C:
            if a == c:
                continue
            for ch in CH:
                for x in X:
                    expected[reduced_formula_from_counts({a: 1, c: 1, ch: 1, x: 2})] = "Type1_A_C_Ch_X2"

    # Type2: C Ch X
    for c in C:
        for ch in CH:
            for x in X:
                expected[reduced_formula_from_counts({c: 1, ch: 1, x: 1})] = "Type2_C_Ch_X"

    # Type3: B2 C Ch2 X3
    for b in B:
        for c in C:
            if b == c:
                continue
            for ch in CH:
                for x in X:
                    expected[reduced_formula_from_counts({b: 2, c: 1, ch: 2, x: 3})] = "Type3_B2_C_Ch2_X3"

    for pattern in SUPPLEMENT_PATTERNS:
        label = "TrainTemplate_" + pattern
        for formula in enumerate_pattern(pattern):
            expected[formula] = label

    return expected


def iter_sources():
    for source_dir in SOURCE_DIRS:
        if not os.path.isdir(source_dir):
            continue
        for name in sorted(os.listdir(source_dir)):
            if name.endswith(".cif"):
                yield source_dir, os.path.join(source_dir, name)


def reset_final_dir():
    data_dir = os.path.abspath(os.path.join(BASE, "Data"))
    final_abs = os.path.abspath(FINAL_DIR)
    if not final_abs.startswith(data_dir + os.sep):
        raise RuntimeError(f"Refusing unexpected final dir: {FINAL_DIR}")
    if os.path.basename(final_abs) != "predict_final_main6":
        raise RuntimeError(f"Refusing unexpected final dir: {FINAL_DIR}")
    if os.path.isdir(final_abs):
        shutil.rmtree(final_abs)
    os.makedirs(final_abs, exist_ok=True)


def main() -> int:
    expected = expected_formulas()
    reset_final_dir()
    rows = []
    seen = set()
    skipped = Counter()

    for source_dir, path in iter_sources():
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            formula = Composition(stem).reduced_formula
        except Exception:
            skipped["unparseable"] += 1
            continue
        if formula not in expected:
            skipped["outside_main6"] += 1
            continue
        if formula in seen:
            skipped["duplicate"] += 1
            continue
        dest = os.path.join(FINAL_DIR, formula + ".cif")
        shutil.copy2(path, dest)
        seen.add(formula)
        rows.append({
            "formula": formula,
            "type": expected[formula],
            "source_dir": source_dir,
            "source_file": path,
            "dest_file": dest,
        })

    missing = sorted(set(expected) - seen)
    with open(MANIFEST, "w", newline="") as f:
        fieldnames = ["formula", "type", "source_dir", "source_file", "dest_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(MISSING_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["formula", "type"])
        for formula in missing:
            writer.writerow([formula, expected[formula]])

    expected_by_type = Counter(expected.values())
    copied_by_type = Counter(row["type"] for row in rows)
    print("Main6 final build complete")
    print("Final dir:", FINAL_DIR)
    print("Manifest:", MANIFEST)
    print("Expected total:", len(expected))
    print("Copied total:", len(rows))
    print("Expected by type:", dict(expected_by_type))
    print("Copied by type:", dict(copied_by_type))
    print("Skipped:", dict(skipped))
    print("Missing expected:", len(missing))
    if missing:
        print("Missing examples:", missing[:40])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
