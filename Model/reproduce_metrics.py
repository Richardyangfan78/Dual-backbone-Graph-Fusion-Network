#!/usr/bin/env python3
"""Recompute the reported five-fold DBGFN metrics from archived OOF predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("bg_mae", "gt_acc", "gt_f1", "eh_acc", "eh_f1")


def accuracy(targets: list[int], predictions: list[int]) -> float:
    return sum(a == b for a, b in zip(targets, predictions)) / len(targets)


def macro_f1(targets: list[int], predictions: list[int]) -> float:
    scores = []
    for label in sorted(set(targets) | set(predictions)):
        tp = sum(a == label and b == label for a, b in zip(targets, predictions))
        fp = sum(a != label and b == label for a, b in zip(targets, predictions))
        fn = sum(a == label and b != label for a, b in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores)


def compute_fold(rows: list[dict[str, str]]) -> dict[str, float]:
    bg_true = [float(row["bg_true"]) for row in rows]
    bg_pred = [float(row["bg_pred"]) for row in rows]
    gt_true = [int(row["gt_true"]) for row in rows]
    gt_pred = [int(row["gt_pred"]) for row in rows]
    eh_true = [int(row["eh_true"]) for row in rows]
    eh_pred = [int(row["eh_pred"]) for row in rows]
    return {
        "bg_mae": sum(abs(a - b) for a, b in zip(bg_true, bg_pred)) / len(rows),
        "gt_acc": accuracy(gt_true, gt_pred),
        "gt_f1": macro_f1(gt_true, gt_pred),
        "eh_acc": accuracy(eh_true, eh_pred),
        "eh_f1": macro_f1(eh_true, eh_pred),
    }


def read_reported(checkpoint_dir: Path, fold: int) -> dict[str, float]:
    path = checkpoint_dir / f"fold{fold}_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=ROOT / "Results" / "Performance"
        / "six_model_oof_predictions_best_bg_mace_alignn.csv",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=ROOT / "Checkpoint"
    )
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    by_fold: dict[int, list[dict[str, str]]] = {fold: [] for fold in range(5)}
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == "MACE+ALIGNN":
                by_fold[int(row["fold"])].append(row)

    if any(not rows for rows in by_fold.values()):
        raise RuntimeError("The OOF archive does not contain all five DBGFN folds.")

    computed = {fold: compute_fold(rows) for fold, rows in by_fold.items()}
    failures = []
    print("fold  n    BG_MAE    GT_Acc    GT_F1     TS_Acc    TS_F1")
    for fold in range(5):
        values = computed[fold]
        reported = read_reported(args.checkpoint_dir, fold)
        for metric in METRICS:
            delta = abs(values[metric] - reported[metric])
            if delta > args.tolerance:
                failures.append((fold, metric, delta))
        print(
            f"{fold:>4}  {len(by_fold[fold]):>3}  "
            f"{values['bg_mae']:.6f}  {values['gt_acc']:.6f}  "
            f"{values['gt_f1']:.6f}  {values['eh_acc']:.6f}  "
            f"{values['eh_f1']:.6f}"
        )

    mean = {
        metric: sum(computed[fold][metric] for fold in range(5)) / 5
        for metric in METRICS
    }
    print(
        "mean       "
        f"{mean['bg_mae']:.6f}  {mean['gt_acc']:.6f}  "
        f"{mean['gt_f1']:.6f}  {mean['eh_acc']:.6f}  "
        f"{mean['eh_f1']:.6f}"
    )

    if failures:
        for fold, metric, delta in failures:
            print(
                f"Mismatch: fold {fold} {metric} differs by {delta:.3e}",
                file=sys.stderr,
            )
        return 1
    print("PASS: archived OOF predictions reproduce all reported fold metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
