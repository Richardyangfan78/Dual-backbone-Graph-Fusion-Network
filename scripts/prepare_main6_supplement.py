#!/usr/bin/env python3
"""Prepare relax chunks for the six-class main prediction set supplement."""

from __future__ import annotations

import csv
import os
import shutil
from collections import Counter, defaultdict
from itertools import product

from pymatgen.core import Composition


BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
AUDIT = os.path.join(BASE, "Data/predict_train_template_generated_audit.csv")
GENERATED_DIR = os.path.join(BASE, "Data/predict_train_template_generated")
RELAXED_DIR = os.path.join(BASE, "Data/predict_main6_supplement_relaxed")
CHUNK_DIR = os.path.join(BASE, "Data/predict_main6_supplement_chunks")
MISSING_CSV = os.path.join(BASE, "Data/predict_main6_supplement_missing.csv")
MANIFEST = os.path.join(BASE, "Data/predict_main6_supplement_manifest.csv")

PATTERNS = {
    "A1C2_Ch3_X1",
    "B1C1_Ch2_X1",
    "A1B1_Ch1_X1",
}

A = ["Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Au"]
B = ["Mg", "Ca", "Sr", "Ba", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ge", "Sn", "Pd", "Pt"]
C = ["Al", "Ga", "In", "Sc", "Y", "La", "Lu", "Ti", "V", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Fe", "Ru", "Rh", "Bi", "Sb"]
CH = ["S", "Se", "Te"]
X = ["F", "Cl", "Br", "I"]
ROLE_ELEMENTS = {"A": A, "B": B, "C": C}
ROLE_CHARGE = {"A": 1, "B": 2, "C": 3}
CHUNK_SIZE = 1200

COPY_SOURCES = [
    os.path.join(BASE, "Data/predict_train_template_relaxed"),
    os.path.join(BASE, "Data/predict_final"),
    os.path.join(BASE, "Data/predict_jacs_relaxed"),
    os.path.join(BASE, "Data/predict_new_relaxed"),
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
        # Keep roles chemically legible: no same element across role slots.
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
                    yield reduced_formula_from_counts(counts), pattern


def reset_chunk_dir():
    data_dir = os.path.abspath(os.path.join(BASE, "Data"))
    for path in [CHUNK_DIR, RELAXED_DIR]:
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(data_dir + os.sep):
            raise RuntimeError(f"Refusing unexpected path: {path}")
        os.makedirs(abs_path, exist_ok=True)
    if os.path.isdir(CHUNK_DIR):
        shutil.rmtree(CHUNK_DIR)
    os.makedirs(CHUNK_DIR, exist_ok=True)


def index_sources():
    index: dict[str, str] = {}
    for source_dir in COPY_SOURCES:
        if not os.path.isdir(source_dir):
            continue
        for name in os.listdir(source_dir):
            if not name.endswith(".cif"):
                continue
            stem = os.path.splitext(name)[0]
            try:
                formula = Composition(stem).reduced_formula
            except Exception:
                continue
            index.setdefault(formula, os.path.join(source_dir, name))
    return index


def generated_index():
    rows = {}
    if not os.path.exists(AUDIT):
        return rows
    with open(AUDIT, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pattern") not in PATTERNS:
                continue
            formula = Composition(row["formula"]).reduced_formula
            path = os.path.join(GENERATED_DIR, formula + ".cif")
            if os.path.exists(path):
                rows[formula] = path
    return rows


def link_or_copy(src: str, dest: str):
    if os.path.exists(dest):
        return
    try:
        os.symlink(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def main() -> int:
    reset_chunk_dir()
    source_index = index_sources()
    generated = generated_index()
    expected = {}
    for pattern in sorted(PATTERNS):
        for formula, pat in enumerate_pattern(pattern):
            expected[formula] = pat

    rows = []
    missing = []
    to_relax = []
    stats = Counter()

    for formula, pattern in sorted(expected.items()):
        relaxed_path = os.path.join(RELAXED_DIR, formula + ".cif")
        if os.path.exists(relaxed_path):
            stats["already_in_main6_relaxed"] += 1
            rows.append({"formula": formula, "pattern": pattern, "status": "already_relaxed", "source": relaxed_path})
            continue
        if formula in source_index:
            shutil.copy2(source_index[formula], relaxed_path)
            stats["copied_existing_relaxed"] += 1
            rows.append({"formula": formula, "pattern": pattern, "status": "copied_relaxed", "source": source_index[formula]})
            continue
        if formula in generated:
            to_relax.append((formula, pattern, generated[formula]))
            rows.append({"formula": formula, "pattern": pattern, "status": "needs_relax", "source": generated[formula]})
            continue
        missing.append((formula, pattern))
        rows.append({"formula": formula, "pattern": pattern, "status": "missing_generated", "source": ""})

    for i, (formula, _pattern, src) in enumerate(to_relax):
        chunk_id = i // CHUNK_SIZE
        chunk = os.path.join(CHUNK_DIR, f"chunk_{chunk_id:03d}")
        os.makedirs(chunk, exist_ok=True)
        link_or_copy(src, os.path.join(chunk, formula + ".cif"))

    with open(MANIFEST, "w", newline="") as f:
        fieldnames = ["formula", "pattern", "status", "source"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(MISSING_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["formula", "pattern"])
        writer.writerows(missing)

    chunk_count = len([d for d in os.listdir(CHUNK_DIR) if d.startswith("chunk_")])
    by_pattern = Counter(pattern for _formula, pattern, _src in to_relax)
    expected_by_pattern = Counter(expected.values())

    print("Main6 supplement preparation complete")
    print("Expected supplement:", len(expected))
    print("Expected by pattern:", dict(expected_by_pattern))
    print("Stats:", dict(stats))
    print("Need relax:", len(to_relax))
    print("Need relax by pattern:", dict(by_pattern))
    print("Missing:", len(missing))
    print("Chunk count:", chunk_count)
    print("Chunk size:", CHUNK_SIZE)
    print("Relaxed dir:", RELAXED_DIR)
    print("Chunk dir:", CHUNK_DIR)
    print("Manifest:", MANIFEST)
    print("Missing CSV:", MISSING_CSV)
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
