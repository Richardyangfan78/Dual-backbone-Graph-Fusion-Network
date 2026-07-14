#!/usr/bin/env python3
"""Generate training-template ABC chalcohalide supplement structures.

This script mines role/stoichiometry patterns from the training CIFs and expands
selected simple, well-supported patterns by role-preserving element substitution.
Only user-approved A(+1), B(+2), C(+3), Ch(-2), and X(-1) elements are used.
"""

from __future__ import annotations

import csv
import os
import shutil
import warnings
from collections import Counter, defaultdict
from itertools import product

from pymatgen.core import Composition, Element, Structure
from pymatgen.io.cif import CifWriter

warnings.filterwarnings("ignore")


BASE = "/path/to/Dual-backbone-Graph-Fusion-Network"
TRAIN_DIR = os.path.join(BASE, "Data/Inorganic_datasets")
OUT_DIR = os.path.join(BASE, "Data/predict_train_template_generated")
CHUNK_DIR = os.path.join(BASE, "Data/predict_train_template_chunks")
AUDIT_CSV = os.path.join(BASE, "Data/predict_train_template_generated_audit.csv")
TYPE_REPORT = os.path.join(BASE, "Data/training_structure_type_report.csv")
CHUNK_SIZE = 1800

A = ["Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Au"]
B = ["Mg", "Ca", "Sr", "Ba", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ge", "Sn", "Pd", "Pt"]
C = ["Al", "Ga", "In", "Sc", "Y", "La", "Lu", "Ti", "V", "Nb", "Mo", "Zr", "Hf", "Ta", "W", "Fe", "Ru", "Rh", "Bi", "Sb"]
CH = ["S", "Se", "Te"]
X = ["F", "Cl", "Br", "I"]

ROLE_ELEMENTS = {"A": A, "B": B, "C": C}
ROLE_SETS = {role: set(elements) for role, elements in ROLE_ELEMENTS.items()}
ROLE_CHARGE = {"A": 1, "B": 2, "C": 3}
CH_SET = set(CH)
X_SET = set(X)

# Selected from reduced training-set role patterns:
# support >= 4, simple one-Ch/one-X patterns, excluding already-covered Type1/2/3.
SELECTED_PATTERNS = {
    "A1C2_Ch3_X1",
    "A3_Ch1_X1",
    "B1C1_Ch2_X1",
    "A1B1_Ch1_X1",
    "C3_Ch1_X7",
    "B1B2_Ch2_X2",
    "C1C2_Ch4_X1",
    "B3C1_Ch4_X1",
    "B4_Ch1_X6",
    "A1B2_Ch2_X1",
}

RADII = {
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Cu": 0.77, "Ag": 1.15, "Au": 1.37,
    "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35, "Mn": 0.83, "Fe": 0.78, "Co": 0.75,
    "Ni": 0.69, "Zn": 0.74, "Ge": 0.73, "Sn": 0.69, "Pd": 0.86, "Pt": 0.80,
    "Al": 0.535, "Ga": 0.62, "In": 0.80, "Sc": 0.745, "Y": 0.90, "La": 1.032, "Lu": 0.861,
    "Ti": 0.670, "V": 0.640, "Nb": 0.640, "Mo": 0.650, "Zr": 0.720, "Hf": 0.710,
    "Ta": 0.640, "W": 0.600, "Ru": 0.680, "Rh": 0.665, "Bi": 1.03, "Sb": 0.76,
    "S": 1.84, "Se": 1.98, "Te": 2.21, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20,
}


def radius(el: str) -> float:
    if el in RADII:
        return RADII[el]
    try:
        r = Element(el).average_ionic_radius
        return float(r) if r is not None else 1.0
    except Exception:
        return 1.0


def reduced_counts(struct: Structure) -> dict[str, int] | None:
    amounts = struct.composition.reduced_composition.get_el_amt_dict()
    if any(abs(v - round(v)) > 1e-6 for v in amounts.values()):
        return None
    return {str(el): int(round(v)) for el, v in amounts.items()}


def reduced_formula_from_counts(counts: dict[str, int]) -> str:
    return Composition(counts).reduced_formula


def role_assignments(metals: list[str], counts: dict[str, int], ch_count: int, x_count: int):
    needed_positive = 2 * ch_count + x_count
    options = []
    for metal in metals:
        roles = [role for role, elements in ROLE_SETS.items() if metal in elements]
        if not roles:
            return []
        options.append(roles)

    assignments = []
    for roles in product(*options):
        positive = sum(ROLE_CHARGE[role] * counts[metal] for metal, role in zip(metals, roles))
        if positive == needed_positive:
            slots = tuple(sorted((role, counts[metal], metal) for metal, role in zip(metals, roles)))
            assignments.append(slots)
    return sorted(set(assignments))


def pattern_key(slots: tuple[tuple[str, int, str], ...], ch_count: int, x_count: int) -> str:
    role_part = "".join(f"{role}{count}" for role, count, _ in slots)
    return f"{role_part}_Ch{ch_count}_X{x_count}"


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


def discover_templates():
    templates = defaultdict(list)
    type_counts = Counter()
    type_examples = defaultdict(list)

    for filename in sorted(os.listdir(TRAIN_DIR)):
        if not filename.endswith(".cif"):
            continue
        path = os.path.join(TRAIN_DIR, filename)
        try:
            struct = Structure.from_file(path)
        except Exception:
            continue
        counts = reduced_counts(struct)
        if not counts:
            continue
        ch = [el for el in counts if el in CH_SET]
        hal = [el for el in counts if el in X_SET]
        metals = [el for el in counts if el not in CH_SET and el not in X_SET]
        if len(ch) != 1 or len(hal) != 1:
            continue
        if not all(any(m in role_set for role_set in ROLE_SETS.values()) for m in metals):
            continue

        assignments = role_assignments(metals, counts, counts[ch[0]], counts[hal[0]])
        if not assignments:
            continue
        slots = assignments[0]
        pattern = pattern_key(slots, counts[ch[0]], counts[hal[0]])
        type_counts[pattern] += 1
        if len(type_examples[pattern]) < 8:
            type_examples[pattern].append(struct.composition.reduced_formula)
        if pattern not in SELECTED_PATTERNS:
            continue

        templates[pattern].append({
            "path": path,
            "filename": filename,
            "structure": struct,
            "slots": slots,
            "ch": ch[0],
            "x": hal[0],
            "ch_count": counts[ch[0]],
            "x_count": counts[hal[0]],
            "formula": struct.composition.reduced_formula,
        })

    with open(TYPE_REPORT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pattern", "training_support", "selected", "examples"])
        for pattern, support in type_counts.most_common():
            writer.writerow([pattern, support, "Yes" if pattern in SELECTED_PATTERNS else "No", ";".join(type_examples[pattern])])

    missing_patterns = sorted(pattern for pattern in SELECTED_PATTERNS if pattern not in templates)
    if missing_patterns:
        raise RuntimeError(f"No templates found for selected patterns: {missing_patterns}")
    return templates, type_counts


def existing_formulas() -> set[str]:
    formulas = set()
    source_dirs = [
        os.path.join(BASE, "Data/predict_final"),
        os.path.join(BASE, "Data/predict_new_relaxed"),
        os.path.join(BASE, "Data/predict_type3_generated"),
        os.path.join(BASE, "Data/predict_type3_generated_extra"),
        os.path.join(BASE, "Data/predict_type3_relaxed"),
        os.path.join(BASE, "Data/predict_type3_relaxed_extra"),
        OUT_DIR,
    ]
    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            continue
        for filename in os.listdir(source_dir):
            if filename.endswith(".cif"):
                stem = os.path.splitext(filename)[0]
                try:
                    formulas.add(Composition(stem).reduced_formula)
                except Exception:
                    formulas.add(stem)
    return formulas


def choose_template(candidates, target_slots, ch: str, x: str):
    def score(template):
        by_role_count = {(role, count): old_el for role, count, old_el in template["slots"]}
        value = 0.0
        for role, count, new_el in target_slots:
            old_el = by_role_count[(role, count)]
            if old_el != new_el:
                value += 1.5 + abs(radius(old_el) - radius(new_el))
        if template["ch"] != ch:
            value += 0.8 + abs(radius(template["ch"]) - radius(ch))
        if template["x"] != x:
            value += 0.8 + abs(radius(template["x"]) - radius(x))
        return value

    return min(candidates, key=score)


def substitute_and_scale(template, target_slots, ch: str, x: str):
    struct = template["structure"].copy()
    mapping = {}
    by_role_count = {(role, count): old_el for role, count, old_el in template["slots"]}
    weighted_ratios = []

    for role, count, new_el in target_slots:
        old_el = by_role_count[(role, count)]
        if old_el != new_el:
            mapping[old_el] = new_el
        weighted_ratios.extend([radius(new_el) / radius(old_el)] * count)
    if template["ch"] != ch:
        mapping[template["ch"]] = ch
    weighted_ratios.extend([radius(ch) / radius(template["ch"])] * template["ch_count"])
    if template["x"] != x:
        mapping[template["x"]] = x
    weighted_ratios.extend([radius(x) / radius(template["x"])] * template["x_count"])

    if mapping:
        struct.replace_species(mapping)

    scale = 1.0
    for ratio in weighted_ratios:
        scale *= ratio
    scale = scale ** (1.0 / max(len(weighted_ratios), 1))
    struct.scale_lattice(struct.volume * scale ** 3)
    return struct


def enumerate_targets(pattern: str):
    slots, ch_count, x_count = parse_pattern(pattern)
    role_choices = [ROLE_ELEMENTS[role] for role, _ in slots]
    for elements in product(*role_choices):
        repeated_role_values = defaultdict(list)
        for (role, _), element in zip(slots, elements):
            repeated_role_values[role].append(element)
        if any(len(values) != len(set(values)) for values in repeated_role_values.values()):
            continue
        for ch in CH:
            for x in X:
                counts = {ch: ch_count, x: x_count}
                target_slots = []
                for (role, count), element in zip(slots, elements):
                    counts[element] = counts.get(element, 0) + count
                    target_slots.append((role, count, element))
                positive = sum(ROLE_CHARGE[role] * count for role, count, _ in target_slots)
                if positive != 2 * ch_count + x_count:
                    continue
                yield tuple(target_slots), ch, x, reduced_formula_from_counts(counts)


def reset_output_dirs():
    os.makedirs(os.path.join(BASE, "Data"), exist_ok=True)
    for path in [OUT_DIR, CHUNK_DIR]:
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(os.path.abspath(os.path.join(BASE, "Data")) + os.sep):
            raise RuntimeError(f"Refusing to reset unexpected path: {path}")
        if os.path.exists(abs_path):
            shutil.rmtree(abs_path)
        os.makedirs(abs_path, exist_ok=True)


def make_chunks(files: list[str]):
    chunk_count = 0
    for index, file_path in enumerate(files):
        chunk_id = index // CHUNK_SIZE
        chunk_dir = os.path.join(CHUNK_DIR, f"chunk_{chunk_id:03d}")
        os.makedirs(chunk_dir, exist_ok=True)
        dest = os.path.join(chunk_dir, os.path.basename(file_path))
        try:
            os.symlink(file_path, dest)
        except OSError:
            shutil.copy2(file_path, dest)
        chunk_count = max(chunk_count, chunk_id + 1)
    return chunk_count


def main() -> int:
    reset_output_dirs()
    templates, type_counts = discover_templates()
    seen = existing_formulas()
    rows = []
    stats = Counter()

    for pattern in sorted(SELECTED_PATTERNS):
        candidates = templates[pattern]
        for target_slots, ch, x, formula in enumerate_targets(pattern):
            if formula in seen:
                stats["skip_existing"] += 1
                continue
            template = choose_template(candidates, target_slots, ch, x)
            out_path = os.path.join(OUT_DIR, formula + ".cif")
            if os.path.exists(out_path):
                stats["skip_duplicate_output"] += 1
                continue
            try:
                struct = substitute_and_scale(template, target_slots, ch, x)
                if struct.composition.reduced_formula != formula:
                    stats["skip_formula_mismatch"] += 1
                    continue
                CifWriter(struct, symprec=None).write_file(out_path)
                rows.append({
                    "formula": formula,
                    "pattern": pattern,
                    "template_formula": template["formula"],
                    "template_file": template["filename"],
                    "roles": ";".join(f"{role}{count}:{el}" for role, count, el in target_slots),
                    "chalcogen": ch,
                    "halide": x,
                    "charge_check": "sum(role_charge*stoich)=2*Ch+X",
                })
                seen.add(formula)
                stats["generated"] += 1
            except Exception as exc:
                stats["failed"] += 1
                if stats["failed"] <= 10:
                    print(f"FAILED {formula}: {exc}")

    with open(AUDIT_CSV, "w", newline="") as f:
        fieldnames = ["formula", "pattern", "template_formula", "template_file", "roles", "chalcogen", "halide", "charge_check"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    files = sorted(os.path.join(OUT_DIR, name) for name in os.listdir(OUT_DIR) if name.endswith(".cif"))
    chunk_count = make_chunks(files)

    by_pattern = Counter(row["pattern"] for row in rows)
    print("Training-template supplement generation complete")
    print("Selected patterns:", ",".join(sorted(SELECTED_PATTERNS)))
    print("Training support:", {pattern: type_counts.get(pattern, 0) for pattern in sorted(SELECTED_PATTERNS)})
    print("Stats:", dict(stats))
    print("Generated files:", len(files))
    print("Generated by pattern:", dict(by_pattern))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk count:", chunk_count)
    print("Output dir:", OUT_DIR)
    print("Chunk dir:", CHUNK_DIR)
    print("Audit CSV:", AUDIT_CSV)
    print("Type report:", TYPE_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
