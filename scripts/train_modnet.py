#!/usr/bin/env python3
"""
Train MODNet on Chalcohalide data: bandgap (regression), gaptype (3-class), stability (regression).
Uses same 80/10/10 split with seed 123 as CGCNN/ALIGNN for fair benchmark.
"""
import os
import sys
import csv
import json
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from pymatgen.core import Structure

# Add modnet to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "modnet"))

from modnet.preprocessing import MODData
from modnet.models import MODNetModel


def load_id_prop_and_structures(data_dir: Path):
    """Load id_prop.csv and CIFs; return list of (structure_id, structure, target)."""
    id_prop_path = data_dir / "id_prop.csv"
    if not id_prop_path.exists():
        raise FileNotFoundError(id_prop_path)
    rows = []
    with open(id_prop_path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            sid, val = row[0].strip(), row[1].strip()
            cif_path = data_dir / f"{sid}.cif"
            if not cif_path.exists():
                continue
            struct = Structure.from_file(str(cif_path))
            target = float(val) if "." in val or val.lstrip("-").isdigit() else int(val)
            rows.append((sid, struct, target))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["bandgap", "gaptype", "stability"], required=True)
    parser.add_argument("--data-root", default=None, help="Data root (default: PROJECT_ROOT/Data)")
    parser.add_argument("--out-dir", default=None, help="Output dir (default: modnet_output/<task>)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-feat", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data_root = Path(args.data_root or PROJECT_ROOT / "Data")
    task_to_folder = {
        "bandgap": "bandgap_regression",
        "gaptype": "gap_type_classification",
        "stability": "stability_regression",
    }
    data_dir = data_root / task_to_folder[args.task]
    out_dir = Path(args.out_dir or PROJECT_ROOT / "modnet_output" / args.task)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_id_prop_and_structures(data_dir)
    if not rows:
        raise RuntimeError(f"No valid (id, CIF, target) rows in {data_dir}")
    random.seed(args.seed)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    if n_test < 1 and n >= 2:
        n_test = 1
        n_train = n - n_val - n_test
    train_rows = rows[:n_train]
    val_rows = rows[n_train : n_train + n_val]
    test_rows = rows[n_train + n_val :]

    structures = [r[1] for r in rows]
    is_classification = args.task == "gaptype"
    raw_targets = [r[2] for r in rows]
    targets = (
        np.array(raw_targets, dtype=np.int64).reshape(-1, 1)
        if is_classification
        else np.array(raw_targets, dtype=np.float64).reshape(-1, 1)
    )
    ids = [r[0] for r in rows]

    num_classes = {"target": 3} if is_classification else {"target": 0}
    target_names = ["target"]

    # Matminer2023 与 matminer>=0.9 兼容；DeBreuck2020 要求 matminer==0.6.2 易冲突
    data = MODData(
        materials=structures,
        targets=targets,
        target_names=target_names,
        structure_ids=ids,
        num_classes=num_classes,
        featurizer="Matminer2023",
    )
    data.featurize()
    n_cols = len(data.df_featurized.columns)
    if n_cols == 0:
        raise RuntimeError("Featurization produced no features. Check featurizer and data.")
    data.feature_selection(n=min(500, n_cols), random_state=args.seed)
    n_opt = len(data.get_optimal_descriptors())
    n_feat = min(args.n_feat, n_opt) if n_opt else 1

    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n))
    train_data = data.from_indices(train_idx)
    val_data = data.from_indices(val_idx)
    test_data = data.from_indices(test_idx)

    model = MODNetModel(
        targets=[["target"]],
        weights={"target": 1.0},
        num_classes=num_classes,
        n_feat=n_feat,
        num_neurons=[[64], [32], [16], [16]],
    )
    model.fit(
        train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=64,
        lr=0.001,
        verbose=1,
    )

    # 分类时 return_prob=False 得到 pred["target"] 为类别索引；回归时 pred["target"] 为连续值
    pred = model.predict(test_data, return_prob=False)
    y_true = test_data.df_targets["target"].values

    if is_classification:
        from sklearn.metrics import accuracy_score, f1_score
        y_pred = np.asarray(pred["target"].values).ravel().astype(int)
        if y_true.dtype.kind in ("f", "c"):
            y_true = y_true.astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        metrics = {"accuracy": acc, "f1_weighted": f1}
        print(f"Test Accuracy: {acc:.4f}  F1 (weighted): {f1:.4f}")
    else:
        from sklearn.metrics import mean_absolute_error
        y_pred = np.asarray(pred["target"].values).ravel().astype(float)
        mae = mean_absolute_error(y_true, y_pred)
        metrics = {"mae": mae}
        print(f"Test MAE: {mae:.4f}")

    model.save(str(out_dir / "model.pkl"))
    ids = list(test_data.structure_ids)
    pd.DataFrame({"id": ids, "y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "test_predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model and predictions to {out_dir}")


if __name__ == "__main__":
    main()
