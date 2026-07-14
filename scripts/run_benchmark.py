#!/usr/bin/env python3
"""
Aggregate benchmark results for CGCNN, ALIGNN, MODNet on:
  - Bandgap (regression, MAE)
  - Gap type (classification, Accuracy / F1)
  - Stability / thermodynamic (regression, MAE)

Reads from: checkpoints + logs (CGCNN), alignn_output/* (ALIGNN), modnet_output/* (MODNet).
Outputs: benchmark_results.csv and printed table.
"""
import os
import re
import csv
import json
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_cgcnn_metrics():
    """Parse CGCNN logs for final test MAE (regression) or best val (classification)."""
    out = {}
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.exists():
        return out
    # Latest log per task
    for task, prefix in [
        ("bandgap", "bandgap_"),
        ("gaptype", "gaptype_"),
        ("stability", "stability_"),
    ]:
        logs = sorted(logs_dir.glob(f"{prefix}*.log"), key=os.path.getmtime, reverse=True)
        if not logs:
            continue
        text = logs[0].read_text(errors="ignore")
        if task in ("bandgap", "stability"):
            m = re.search(r"\*\*\s*MAE\s+([\d.]+)", text)
            if m:
                out[task] = {"mae": float(m.group(1))}
        else:
            m = re.search(r"Test (?:ROCAUC|Accuracy|accuracy)[:\s]+([\d.]+)", text, re.I)
            if m:
                out[task] = {"accuracy": float(m.group(1))}
            m2 = re.search(r"F1[:\s]+([\d.]+)", text, re.I)
            if m2 and task in out:
                out[task]["f1_weighted"] = float(m2.group(1))
    return out


def get_alignn_metrics():
    """Read ALIGNN prediction_results_test_set.csv and compute metrics."""
    out = {}
    base = PROJECT_ROOT / "alignn_output"
    if not base.exists():
        return out
    for task, subdir in [
        ("bandgap", "bandgap_regression"),
        ("gaptype", "gaptype_classification"),
        ("stability", "stability_regression"),
    ]:
        csv_path = base / subdir / "prediction_results_test_set.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "prediction" not in df.columns or "target" not in df.columns:
            continue
        y_true = df["target"].values
        y_pred = df["prediction"].values
        if task == "gaptype":
            from sklearn.metrics import accuracy_score, f1_score
            out[task] = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            }
        else:
            from sklearn.metrics import mean_absolute_error
            out[task] = {"mae": mean_absolute_error(y_true, y_pred)}
    return out


def get_modnet_metrics():
    """Read MODNet metrics.json per task."""
    out = {}
    base = PROJECT_ROOT / "modnet_output"
    if not base.exists():
        return out
    for task in ("bandgap", "gaptype", "stability"):
        p = base / task / "metrics.json"
        if not p.exists():
            continue
        with open(p) as f:
            out[task] = json.load(f)
    return out


def main():
    cgcnn = get_cgcnn_metrics()
    alignn = get_alignn_metrics()
    modnet = get_modnet_metrics()

    tasks = [
        ("bandgap", "Bandgap (MAE eV)", "mae", "regression"),
        ("gaptype", "Gap type (Acc / F1)", "classification", "classification"),
        ("stability", "Stability (MAE)", "mae", "regression"),
    ]
    rows = []
    for task_key, task_label, metric_key, task_type in tasks:
        row = {"task": task_label}
        for name, data in [("CGCNN", cgcnn), ("ALIGNN", alignn), ("MODNet", modnet)]:
            if task_key not in data:
                row[name] = ""
                continue
            d = data[task_key]
            if task_type == "regression":
                mae = d.get("mae")
                row[name] = f"{mae:.4f}" if mae is not None else ""
            else:
                acc = d.get("accuracy")
                f1 = d.get("f1_weighted")
                row[name] = f"Acc={acc:.3f}" if acc is not None else ""
                if f1 is not None:
                    row[name] += f" F1={f1:.3f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "benchmark_results.csv"
    df.to_csv(out_csv, index=False)
    print("Benchmark summary (same tasks: bandgap value, gap type, thermodynamic/stability):")
    print(df.to_string(index=False))
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
