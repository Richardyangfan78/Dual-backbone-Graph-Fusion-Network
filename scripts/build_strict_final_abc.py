#!/usr/bin/env python3
"""Build a strict ABC chalcohalide prediction set.

The final set includes only:
  Type1: A(+1) C(+3) Ch X2
  Type2: C(+3) Ch X
  Type3: B(+2)2 C(+3) Ch2 X3

All formulas are checked against the user-specified role element whitelist and
the corresponding charge-balance rule before being copied.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from pymatgen.core import Composition


BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
FINAL_DIR = os.path.join(BASE, "Data/predict_final_strict_abc")
MANIFEST = os.path.join(BASE, "Data/predict_final_strict_abc_manifest.csv")
MISSING_CSV = os.path.join(BASE, "Data/predict_final_strict_abc_missing.csv")

SOURCE_DIRS = [
    os.path.join(BASE, "Data/predict_new_relaxed"),
    os.path.join(BASE, "Data/predict_type3_relaxed"),
    os.path.join(BASE, "Data/predict_type3_relaxed_extra"),
    os.path.join(BASE, "Data/predict_final"),
    os.path.join(BASE, "Data/predict_jacs_relaxed"),
]

A1 = ["Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Au"]
B2 = ["Mg", "Ca", "Sr", "Ba", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ge", "Sn", "Pd", "Pt"]
C3 = ["Al", "Ga", "In", "Sc", "Y", "La", "Lu", "Ti", "V", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Fe", "Ru", "Rh", "Bi", "Sb"]
CH = ["S", "Se", "Te"]
X = ["F", "Cl", "Br", "I"]

A1_SET = set(A1)
B2_SET = set(B2)
C3_SET = set(C3)
CH_SET = set(CH)
X_SET = set(X)


@dataclass(frozen=True)
class FormulaClass:
    kind: str
    reduced_formula: str
    charge_rule: str


def reduced_formula(formula: str) -> str:
    return Composition(formula).reduced_formula


def composition_counts(formula: str) -> dict[str, int] | None:
    try:
        comp = Composition(formula)
        counts: dict[str, int] = {}
        for el, amt in comp.get_el_amt_dict().items():
            rounded = round(amt)
            if abs(amt - rounded) > 1e-6:
                return None
            counts[str(el)] = int(rounded)
        return counts
    except Exception:
        return None


def classify_formula(formula: str) -> FormulaClass | None:
    counts = composition_counts(formula)
    if not counts:
        return None

    metals = [el for el in counts if el not in CH_SET and el not in X_SET]
    ch = [el for el in counts if el in CH_SET]
    hal = [el for el in counts if el in X_SET]
    reduced = reduced_formula(formula)

    if len(metals) == 2 and len(ch) == 1 and len(hal) == 1:
        ch_el, x_el = ch[0], hal[0]

        # Type1: A(+1) C(+3) Ch(-2) X(-1)2
        if counts[ch_el] == 1 and counts[x_el] == 2:
            a_candidates = [m for m in metals if counts[m] == 1 and m in A1_SET]
            c_candidates = [m for m in metals if counts[m] == 1 and m in C3_SET]
            if a_candidates and c_candidates and a_candidates[0] != c_candidates[0]:
                return FormulaClass(
                    kind="Type1_A1_C3_Ch_X2",
                    reduced_formula=reduced,
                    charge_rule="A(+1)+C(+3)+Ch(-2)+2*X(-1)=0",
                )

        # Type3: B(+2)2 C(+3) Ch(-2)2 X(-1)3
        if counts[ch_el] == 2 and counts[x_el] == 3:
            b_candidates = [m for m in metals if counts[m] == 2 and m in B2_SET]
            c_candidates = [m for m in metals if counts[m] == 1 and m in C3_SET]
            if b_candidates and c_candidates and b_candidates[0] != c_candidates[0]:
                return FormulaClass(
                    kind="Type3_B2_C3_Ch2_X3",
                    reduced_formula=reduced,
                    charge_rule="2*B(+2)+C(+3)+2*Ch(-2)+3*X(-1)=0",
                )

    # Type2: C(+3) Ch(-2) X(-1)
    if len(metals) == 1 and len(ch) == 1 and len(hal) == 1:
        m, ch_el, x_el = metals[0], ch[0], hal[0]
        if counts[m] == 1 and counts[ch_el] == 1 and counts[x_el] == 1 and m in C3_SET:
            return FormulaClass(
                kind="Type2_C3_Ch_X",
                reduced_formula=reduced,
                charge_rule="C(+3)+Ch(-2)+X(-1)=0",
            )

    return None


def expected_formulas() -> dict[str, tuple[str, str]]:
    expected: dict[str, tuple[str, str]] = {}
    for a in A1:
        for c in C3:
            for ch in CH:
                for x in X:
                    formula = f"{a}{c}{ch}{x}2"
                    expected[reduced_formula(formula)] = ("Type1_A1_C3_Ch_X2", formula)
    for c in C3:
        for ch in CH:
            for x in X:
                formula = f"{c}{ch}{x}"
                expected[reduced_formula(formula)] = ("Type2_C3_Ch_X", formula)
    for b in B2:
        for c in C3:
            if b == c:
                continue
            for ch in CH:
                for x in X:
                    formula = f"{b}2{c}{ch}2{x}3"
                    expected[reduced_formula(formula)] = ("Type3_B2_C3_Ch2_X3", formula)
    return expected


def iter_source_files() -> Iterable[tuple[str, str]]:
    for source_dir in SOURCE_DIRS:
        if not os.path.isdir(source_dir):
            continue
        for name in sorted(os.listdir(source_dir)):
            if name.endswith(".cif"):
                yield source_dir, os.path.join(source_dir, name)


def safe_reset_final_dir() -> None:
    data_dir = os.path.join(BASE, "Data")
    final_abs = os.path.abspath(FINAL_DIR)
    if not final_abs.startswith(os.path.abspath(data_dir) + os.sep):
        raise RuntimeError(f"Refusing to reset unexpected final dir: {FINAL_DIR}")
    if os.path.basename(final_abs) != "predict_final_strict_abc":
        raise RuntimeError(f"Refusing to reset unexpected final dir: {FINAL_DIR}")
    if os.path.isdir(final_abs):
        shutil.rmtree(final_abs)
    os.makedirs(final_abs, exist_ok=True)


def main() -> int:
    expected = expected_formulas()
    safe_reset_final_dir()

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    skipped = Counter()
    copied_by_kind = Counter()

    for source_dir, path in iter_source_files():
        stem = os.path.splitext(os.path.basename(path))[0]
        cls = classify_formula(stem)
        if cls is None:
            skipped["not_strict_abc"] += 1
            continue
        if cls.reduced_formula not in expected:
            skipped["outside_expected_grid"] += 1
            continue
        if cls.reduced_formula in seen:
            skipped["duplicate"] += 1
            continue

        dest_name = cls.reduced_formula + ".cif"
        dest_path = os.path.join(FINAL_DIR, dest_name)
        shutil.copy2(path, dest_path)
        seen.add(cls.reduced_formula)
        copied_by_kind[cls.kind] += 1
        rows.append({
            "formula": stem,
            "reduced_formula": cls.reduced_formula,
            "type": cls.kind,
            "source_dir": source_dir,
            "source_file": path,
            "dest_file": dest_path,
            "charge_rule": cls.charge_rule,
        })

    missing = sorted(set(expected) - seen)
    with open(MANIFEST, "w", newline="") as f:
        fieldnames = ["formula", "reduced_formula", "type", "source_dir", "source_file", "dest_file", "charge_rule"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(MISSING_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reduced_formula", "expected_type", "canonical_formula"])
        for rf in missing:
            kind, canonical = expected[rf]
            writer.writerow([rf, kind, canonical])

    print("Strict ABC final set build complete")
    print("Final dir:", FINAL_DIR)
    print("Manifest:", MANIFEST)
    print("Expected total:", len(expected))
    print("Copied total:", len(rows))
    print("Copied by type:", dict(copied_by_kind))
    print("Skipped:", dict(skipped))
    print("Missing expected:", len(missing))

    if missing:
        print("Missing examples:", missing[:30])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
