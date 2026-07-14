#!/usr/bin/env python3
"""Generate six-model OOF bandgap scatter-fitting and recall plots.

Outputs:
  - benchmark_bandgap_scatter_fits.png (2x3 subplots)
  - benchmark_recall_gaptype_stability.png (macro recall bars)
  - figures/model_bandgap_scatter/*_bandgap_scatter_fit.png (per-model)
  - six_model_oof_predictions.csv
  - six_model_recall_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.loader import DataLoader as PyGDataLoader


PROJECT = Path(__file__).resolve().parent
MULTITASK_DIR = PROJECT / "multitask"
if str(MULTITASK_DIR) not in sys.path:
    sys.path.insert(0, str(MULTITASK_DIR))

from data_mt_v4 import stratified_kfold_v4  # noqa: E402
from model_mt_v10 import CrystalGraphConvNetMTV10  # noqa: E402
from train_mt_v10 import CachedGraphDatasetV7, Normalizer as CGCNNNormalizer, apply_rules, collate_pool_mt_v7  # noqa: E402

from data_mace_mt import MACECrystalDataset, stratified_kfold_split as mace_split  # noqa: E402
from model_mace_mt import MACEMultiTask  # noqa: E402

from data_crystal_mt import CrystalMultiTaskDataset, stratified_kfold_split as crystal_split  # noqa: E402
from model_alignn_pyg import ALIGNNMultiTaskPyG  # noqa: E402
from model_m3gnet_pyg import M3GNetMultiTaskPyG  # noqa: E402

from data_dual_backbone import DualBackboneDataset, stratified_kfold_split as dual_split  # noqa: E402
from model_dual_backbone import DualBackboneMultiTask  # noqa: E402
from model_mace_m3gnet import DualBackboneMACEM3GNet  # noqa: E402


MODEL_ORDER = [
    "CGCNN",
    "MACE",
    "ALIGNN",
    "M3GNet",
    "MACE+M3GNet",
    "MACE+ALIGNN",
]

MODEL_COLORS = {
    "CGCNN": "#B09C85",
    "MACE": "#4DBBD5",
    "ALIGNN": "#00A087",
    "M3GNet": "#E64B35",
    "MACE+M3GNet": "#3C5488",
    "MACE+ALIGNN": "#DC0000",
}


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_bandgap_ev(x: torch.Tensor, log_bg: bool) -> torch.Tensor:
    if log_bg:
        return torch.expm1(x)
    return x


def append_rows(
    rows: List[Dict[str, object]],
    model_name: str,
    fold: int,
    bg_true: np.ndarray,
    bg_pred: np.ndarray,
    gt_true: np.ndarray,
    gt_pred: np.ndarray,
    eh_true: np.ndarray,
    eh_pred: np.ndarray,
) -> None:
    for i in range(len(bg_true)):
        rows.append(
            {
                "model": model_name,
                "fold": int(fold),
                "bg_true": float(bg_true[i]),
                "bg_pred": float(bg_pred[i]),
                "gt_true": int(gt_true[i]),
                "gt_pred": int(gt_pred[i]),
                "eh_true": int(eh_true[i]),
                "eh_pred": int(eh_pred[i]),
            }
        )


def load_mace_calculator(device: torch.device, model_name: str = "small"):
    from mace.calculators import mace_mp

    calc = mace_mp(model=model_name, default_dtype="float32", device=str(device))
    return calc


def evaluate_cgcnn(device: torch.device) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ckpt_root = PROJECT / "checkpoints" / "multitask_v10"
    first_ckpt = torch.load(
        ckpt_root / "fold_0_seed_42" / "best_composite.pth.tar",
        map_location="cpu",
        weights_only=False,
    )
    args = first_ckpt["args"]

    id_prop_path = Path(args["data_dir"]) / "id_prop.csv"
    with open(id_prop_path) as f:
        id_prop_data = list(csv.reader(f))
    random.seed(123)
    random.shuffle(id_prop_data)

    dataset = CachedGraphDatasetV7(
        args["cache_dir"],
        id_prop_data,
        merge_metal_indirect=bool(args.get("merge_metal_indirect", True)),
        augment_noise=0.0,
        data_dir=None,
        augment_perturb=0.0,
        augment_scale=0.0,
        augment_structure_prob=0.0,
    )
    dataset.augment = False

    folds = stratified_kfold_v4(
        dataset.id_prop_data,
        int(args.get("k_folds", 5)),
        random_seed=123,
        merge_metal_indirect=bool(args.get("merge_metal_indirect", True)),
    )

    (s_af, s_nf, _), _, _ = dataset[0]
    for fold in range(int(args.get("k_folds", 5))):
        print(f"[CGCNN] fold {fold}")
        ckpt_path = ckpt_root / f"fold_{fold}_seed_42" / "best_composite.pth.tar"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        model = CrystalGraphConvNetMTV10(
            s_af.shape[-1],
            s_nf.shape[-1],
            atom_fea_len=int(args["atom_fea_len"]),
            n_conv=int(args["n_conv"]),
            h_fea_len=int(args["h_fea_len"]),
            n_h=int(args["n_h"]),
            n_gap_classes=dataset.n_gap_classes,
            n_eh_classes=2,
            dropout=float(args["dropout"]),
            n_attn_heads=int(args["n_attn_heads"]),
            use_cosine_classifier=bool(args.get("use_cosine_classifier", False)),
            cosine_temp=float(args.get("cosine_temp", 0.1)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()

        normalizer_bg = CGCNNNormalizer(
            None,
            log_transform=bool(args.get("log_bg", True)),
            robust=bool(args.get("robust_norm", False)),
        )
        normalizer_bg.load_state_dict(ckpt["normalizer_bg"])

        test_idx = folds[fold]
        loader = TorchDataLoader(
            dataset,
            sampler=SubsetRandomSampler(test_idx),
            batch_size=int(args.get("batch_size", 48)),
            collate_fn=collate_pool_mt_v7,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        all_bg_true, all_bg_pred = [], []
        all_gt_true, all_gt_pred = [], []
        all_eh_true, all_eh_pred = [], []

        with torch.no_grad():
            for inputs, targets, _ in loader:
                atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
                if device.type == "cuda":
                    atom_fea = atom_fea.cuda()
                    nbr_fea = nbr_fea.cuda()
                    nbr_fea_idx = nbr_fea_idx.cuda()

                p_bg, p_gt, p_eh = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

                bg_pred = normalizer_bg.denorm(p_bg.detach().cpu()).view(-1)
                bg_true = targets["bandgap"].detach().cpu().view(-1)

                gt_pred = apply_rules(
                    bg_pred,
                    p_gt.detach().cpu(),
                    merge_metal_indirect=bool(args.get("merge_metal_indirect", True)),
                ).view(-1)
                gt_true = targets["gaptype"].detach().cpu().view(-1)

                eh_pred = p_eh.detach().cpu().argmax(dim=1).view(-1)
                eh_true = targets["eh_label"].detach().cpu().view(-1)

                all_bg_true.append(bg_true.numpy())
                all_bg_pred.append(bg_pred.numpy())
                all_gt_true.append(gt_true.numpy())
                all_gt_pred.append(gt_pred.numpy())
                all_eh_true.append(eh_true.numpy())
                all_eh_pred.append(eh_pred.numpy())

        append_rows(
            rows,
            "CGCNN",
            fold,
            np.concatenate(all_bg_true),
            np.concatenate(all_bg_pred),
            np.concatenate(all_gt_true),
            np.concatenate(all_gt_pred),
            np.concatenate(all_eh_true),
            np.concatenate(all_eh_pred),
        )

    return pd.DataFrame(rows)


def evaluate_mace(device: torch.device) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ckpt_dir = PROJECT / "checkpoints" / "mace_finetune_v2"
    ckpt0 = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args = ckpt0["args"]

    calc = load_mace_calculator(device, model_name=args.get("mace_model", "small"))
    dataset = MACECrystalDataset(
        root_dir=args["data_dir"],
        z_table=calc.z_table,
        r_max=calc.r_max,
        cache_dir=args["cache_dir"],
        merge_metal_indirect=bool(args.get("merge_metal_indirect", True)),
        log_bg=bool(args.get("log_bg", True)),
    )

    for fold in range(int(args.get("k_folds", 5))):
        print(f"[MACE] fold {fold}")
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)

        _, _, test_idx = mace_split(
            dataset,
            k_folds=int(args.get("k_folds", 5)),
            fold=fold,
            val_ratio=float(args.get("val_ratio", 0.1)),
            seed=int(args.get("seed", 42)),
        )
        loader = PyGDataLoader(
            [dataset[i] for i in test_idx],
            batch_size=int(args.get("batch_size", 32)),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        model = MACEMultiTask(
            calc.models[0],
            h_fea_len=int(args["h_fea_len"]),
            n_gap_classes=2,
            n_eh_classes=2,
            dropout=float(args["dropout"]),
            n_attn_heads=int(args["n_attn_heads"]),
            use_cosine_classifier=bool(args.get("use_cosine_classifier", True)),
            cosine_temp=float(args.get("cosine_temp", 0.1)),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if hasattr(model, "set_cond_weight"):
            model.set_cond_weight(1.0)
        model.eval()

        mean = float(ckpt["normalizer_mean"])
        std = float(ckpt["normalizer_std"])
        log_bg = bool(args.get("log_bg", True))

        all_bg_true, all_bg_pred = [], []
        all_gt_true, all_gt_pred = [], []
        all_eh_true, all_eh_pred = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                p_bg, p_gt, p_eh = model(batch.to_dict())

                bg_pred = (p_bg.view(-1).detach().cpu() * (std + 1e-8) + mean)
                bg_true = batch.bg.view(-1).detach().cpu()
                bg_pred = tensor_to_bandgap_ev(bg_pred, log_bg)
                bg_true = tensor_to_bandgap_ev(bg_true, log_bg)

                gt_pred = p_gt.detach().cpu().argmax(dim=1).view(-1)
                gt_true = batch.gt.view(-1).detach().cpu()
                eh_pred = p_eh.detach().cpu().argmax(dim=1).view(-1)
                eh_true = batch.eh.view(-1).detach().cpu()

                all_bg_true.append(bg_true.numpy())
                all_bg_pred.append(bg_pred.numpy())
                all_gt_true.append(gt_true.numpy())
                all_gt_pred.append(gt_pred.numpy())
                all_eh_true.append(eh_true.numpy())
                all_eh_pred.append(eh_pred.numpy())

        append_rows(
            rows,
            "MACE",
            fold,
            np.concatenate(all_bg_true),
            np.concatenate(all_bg_pred),
            np.concatenate(all_gt_true),
            np.concatenate(all_gt_pred),
            np.concatenate(all_eh_true),
            np.concatenate(all_eh_pred),
        )

    return pd.DataFrame(rows)


def evaluate_crystal(device: torch.device, model_type: str) -> pd.DataFrame:
    model_name = "ALIGNN" if model_type == "alignn" else "M3GNet"
    ckpt_dir = PROJECT / "checkpoints" / ("alignn_mt" if model_type == "alignn" else "m3gnet_mt")
    ckpt0 = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args = ckpt0["args"]

    rows: List[Dict[str, object]] = []
    dataset = CrystalMultiTaskDataset(
        root_dir=args["data_dir"],
        cache_dir=args["cache_dir"],
        radius=float(args.get("radius", 8.0)),
        max_num_nbr=int(args.get("max_num_nbr", 12)),
        dist_bins=int(args.get("dist_bins", 80)),
        angle_bins=int(args.get("angle_bins", 40)),
        log_bg=bool(args.get("log_bg", True)),
        merge_metal_indirect=bool(args.get("merge_metal_indirect", True)),
        eh_threshold=float(args.get("eh_threshold", 0.01)),
    )

    for fold in range(int(args.get("k_folds", 5))):
        print(f"[{model_name}] fold {fold}")
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)

        _, _, test_idx = crystal_split(
            dataset,
            k_folds=int(args.get("k_folds", 5)),
            fold=fold,
            val_ratio=float(args.get("val_ratio", 0.1)),
            seed=int(args.get("seed", 42)),
        )
        loader = PyGDataLoader(
            [dataset[i] for i in test_idx],
            batch_size=int(args.get("batch_size", 32)),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            follow_batch=["x"],
        )

        if model_type == "alignn":
            model = ALIGNNMultiTaskPyG(
                atom_input_dim=94,
                edge_input_dim=int(args.get("dist_bins", 80)),
                angle_input_dim=int(args.get("angle_bins", 40)),
                hidden_dim=int(args.get("hidden_dim", 256)),
                n_alignn_layers=int(args.get("n_alignn_layers", 4)),
                n_gcn_layers=int(args.get("n_gcn_layers", 4)),
                dropout=float(args.get("dropout", 0.3)),
            ).to(device)
        else:
            model = M3GNetMultiTaskPyG(
                atom_input_dim=94,
                edge_input_dim=int(args.get("dist_bins", 80)),
                angle_input_dim=int(args.get("angle_bins", 40)),
                hidden_dim=int(args.get("hidden_dim", 128)),
                n_blocks=int(args.get("n_blocks", 3)),
                dropout=float(args.get("dropout", 0.3)),
            ).to(device)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        mean = float(ckpt["normalizer_mean"])
        std = float(ckpt["normalizer_std"])
        log_bg = bool(args.get("log_bg", True))

        all_bg_true, all_bg_pred = [], []
        all_gt_true, all_gt_pred = [], []
        all_eh_true, all_eh_pred = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                p_bg, p_gt, p_eh = model(batch)

                bg_pred = (p_bg.view(-1).detach().cpu() * (std + 1e-8) + mean)
                bg_true = batch.bg.view(-1).detach().cpu()
                bg_pred = tensor_to_bandgap_ev(bg_pred, log_bg)
                bg_true = tensor_to_bandgap_ev(bg_true, log_bg)

                gt_pred = p_gt.detach().cpu().argmax(dim=1).view(-1)
                gt_true = batch.gt.view(-1).detach().cpu()
                eh_pred = p_eh.detach().cpu().argmax(dim=1).view(-1)
                eh_true = batch.eh.view(-1).detach().cpu()

                all_bg_true.append(bg_true.numpy())
                all_bg_pred.append(bg_pred.numpy())
                all_gt_true.append(gt_true.numpy())
                all_gt_pred.append(gt_pred.numpy())
                all_eh_true.append(eh_true.numpy())
                all_eh_pred.append(eh_pred.numpy())

        append_rows(
            rows,
            model_name,
            fold,
            np.concatenate(all_bg_true),
            np.concatenate(all_bg_pred),
            np.concatenate(all_gt_true),
            np.concatenate(all_gt_pred),
            np.concatenate(all_eh_true),
            np.concatenate(all_eh_pred),
        )

    return pd.DataFrame(rows)


def evaluate_dual(device: torch.device, dual_type: str) -> pd.DataFrame:
    model_name = "MACE+ALIGNN" if dual_type == "alignn" else "MACE+M3GNet"
    ckpt_dir = PROJECT / "checkpoints" / ("dual_backbone" if dual_type == "alignn" else "mace_m3gnet")
    ckpt0 = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args = ckpt0["args"]

    rows: List[Dict[str, object]] = []
    calc = load_mace_calculator(device, model_name=args.get("mace_model", "small"))

    dataset = DualBackboneDataset(
        root_dir=args["data_dir"],
        mace_cache_dir=args["mace_cache_dir"],
        crystal_cache_dir=args["crystal_cache_dir"],
        merge_metal_indirect=bool(args.get("merge_metal_indirect", True)),
        log_bg=bool(args.get("log_bg", True)),
    )

    for fold in range(int(args.get("k_folds", 5))):
        print(f"[{model_name}] fold {fold}")
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)

        _, _, test_idx = dual_split(
            dataset,
            k_folds=int(args.get("k_folds", 5)),
            fold=fold,
            val_ratio=float(args.get("val_ratio", 0.15)),
            seed=int(args.get("seed", 42)),
        )
        loader = PyGDataLoader(
            [dataset[i] for i in test_idx],
            batch_size=int(args.get("batch_size", 32)),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        if dual_type == "alignn":
            alignn_model = ALIGNNMultiTaskPyG(
                atom_input_dim=94,
                edge_input_dim=80,
                angle_input_dim=40,
                hidden_dim=256,
                n_alignn_layers=4,
                n_gcn_layers=4,
                n_gap_classes=2,
                n_eh_classes=2,
                dropout=float(args.get("dropout", 0.3)),
            )
            model = DualBackboneMultiTask(
                calc.models[0],
                alignn_model,
                h_fea_len=int(args["h_fea_len"]),
                n_attn_heads=int(args["n_attn_heads"]),
                dropout=float(args.get("dropout", 0.3)),
                use_cosine_classifier=bool(args.get("use_cosine_classifier", True)),
                cosine_temp=float(args.get("cosine_temp", 0.1)),
            ).to(device)
        else:
            m3gnet_model = M3GNetMultiTaskPyG(
                atom_input_dim=94,
                edge_input_dim=80,
                angle_input_dim=40,
                hidden_dim=128,
                n_blocks=3,
                dropout=float(args.get("dropout", 0.3)),
            )
            model = DualBackboneMACEM3GNet(
                calc.models[0],
                m3gnet_model,
                h_fea_len=int(args["h_fea_len"]),
                n_attn_heads=int(args["n_attn_heads"]),
                dropout=float(args.get("dropout", 0.3)),
                use_cosine_classifier=bool(args.get("use_cosine_classifier", True)),
                cosine_temp=float(args.get("cosine_temp", 0.1)),
            ).to(device)

        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if hasattr(model, "set_cond_weight"):
            model.set_cond_weight(1.0)
        model.eval()

        mean = float(ckpt["normalizer_mean"])
        std = float(ckpt["normalizer_std"])
        log_bg = bool(args.get("log_bg", True))

        all_bg_true, all_bg_pred = [], []
        all_gt_true, all_gt_pred = [], []
        all_eh_true, all_eh_pred = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                p_bg, p_gt, p_eh = model(batch)

                bg_pred = (p_bg.view(-1).detach().cpu() * (std + 1e-8) + mean)
                bg_true = batch.bg.view(-1).detach().cpu()
                bg_pred = tensor_to_bandgap_ev(bg_pred, log_bg)
                bg_true = tensor_to_bandgap_ev(bg_true, log_bg)

                gt_pred = p_gt.detach().cpu().argmax(dim=1).view(-1)
                gt_true = batch.gt.view(-1).detach().cpu()
                eh_pred = p_eh.detach().cpu().argmax(dim=1).view(-1)
                eh_true = batch.eh.view(-1).detach().cpu()

                all_bg_true.append(bg_true.numpy())
                all_bg_pred.append(bg_pred.numpy())
                all_gt_true.append(gt_true.numpy())
                all_gt_pred.append(gt_pred.numpy())
                all_eh_true.append(eh_true.numpy())
                all_eh_pred.append(eh_pred.numpy())

        append_rows(
            rows,
            model_name,
            fold,
            np.concatenate(all_bg_true),
            np.concatenate(all_bg_pred),
            np.concatenate(all_gt_true),
            np.concatenate(all_gt_pred),
            np.concatenate(all_eh_true),
            np.concatenate(all_eh_pred),
        )

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        y_true = sub["bg_true"].to_numpy()
        y_pred = sub["bg_pred"].to_numpy()

        mae = float(np.mean(np.abs(y_pred - y_true)))
        denom = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y_pred - y_true) ** 2)) / max(denom, 1e-12)

        slope, intercept = np.polyfit(y_true, y_pred, deg=1)

        gap_macro = recall_score(
            sub["gt_true"], sub["gt_pred"], labels=[0, 1], average="macro", zero_division=0
        )
        gap_per_class = recall_score(
            sub["gt_true"], sub["gt_pred"], labels=[0, 1], average=None, zero_division=0
        )

        eh_macro = recall_score(
            sub["eh_true"], sub["eh_pred"], labels=[0, 1], average="macro", zero_division=0
        )
        eh_per_class = recall_score(
            sub["eh_true"], sub["eh_pred"], labels=[0, 1], average=None, zero_division=0
        )

        summary_rows.append(
            {
                "model": model,
                "n_samples": int(len(sub)),
                "bg_mae": mae,
                "bg_r2": r2,
                "bg_fit_slope": float(slope),
                "bg_fit_intercept": float(intercept),
                "gap_recall_macro": float(gap_macro),
                "gap_recall_cls0": float(gap_per_class[0]),
                "gap_recall_cls1": float(gap_per_class[1]),
                "stability_recall_macro": float(eh_macro),
                "stability_recall_stable_cls0": float(eh_per_class[0]),
                "stability_recall_unstable_cls1": float(eh_per_class[1]),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values("model").reset_index(drop=True)


def plot_one_scatter(sub: pd.DataFrame, model: str, out_path: Path) -> None:
    x = sub["bg_true"].to_numpy()
    y = sub["bg_pred"].to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    mae = float(np.mean(np.abs(y - x)))
    r2 = 1.0 - float(np.sum((y - x) ** 2)) / max(float(np.sum((x - x.mean()) ** 2)), 1e-12)

    xy_min = min(float(np.min(x)), float(np.min(y)))
    xy_max = max(float(np.max(x)), float(np.max(y)))
    pad = 0.05 * max(xy_max - xy_min, 1e-6)

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.scatter(x, y, s=10, alpha=0.45, color=MODEL_COLORS[model], edgecolors="none")
    ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], "k--", lw=1.0, label="y=x")
    ax.plot(
        [xy_min - pad, xy_max + pad],
        [slope * (xy_min - pad) + intercept, slope * (xy_max + pad) + intercept],
        color="#222222",
        lw=1.2,
        label="linear fit",
    )

    ax.set_xlim(xy_min - pad, xy_max + pad)
    ax.set_ylim(xy_min - pad, xy_max + pad)
    ax.set_xlabel("True Bandgap (eV)")
    ax.set_ylabel("Predicted Bandgap (eV)")
    ax.set_title(f"{model}: Bandgap Scatter + Fit", fontweight="bold")
    ax.grid(alpha=0.25, ls="--")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(
        0.03,
        0.97,
        f"MAE = {mae:.3f} eV\\nR2 = {r2:.3f}\\ny = {slope:.2f}x + {intercept:.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_grid(df: pd.DataFrame, out_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        sub = df[df["model"] == model]
        x = sub["bg_true"].to_numpy()
        y = sub["bg_pred"].to_numpy()

        slope, intercept = np.polyfit(x, y, deg=1)
        mae = float(np.mean(np.abs(y - x)))
        r2 = 1.0 - float(np.sum((y - x) ** 2)) / max(float(np.sum((x - x.mean()) ** 2)), 1e-12)

        xy_min = min(float(np.min(x)), float(np.min(y)))
        xy_max = max(float(np.max(x)), float(np.max(y)))
        pad = 0.05 * max(xy_max - xy_min, 1e-6)

        ax.scatter(x, y, s=8, alpha=0.4, color=MODEL_COLORS[model], edgecolors="none")
        ax.plot([xy_min - pad, xy_max + pad], [xy_min - pad, xy_max + pad], "k--", lw=0.9)
        ax.plot(
            [xy_min - pad, xy_max + pad],
            [slope * (xy_min - pad) + intercept, slope * (xy_max + pad) + intercept],
            color="#222222",
            lw=1.1,
        )

        ax.set_xlim(xy_min - pad, xy_max + pad)
        ax.set_ylim(xy_min - pad, xy_max + pad)
        ax.set_title(model, fontweight="bold")
        ax.set_xlabel("True BG (eV)")
        ax.set_ylabel("Pred BG (eV)")
        ax.grid(alpha=0.23, ls="--")
        ax.text(
            0.03,
            0.97,
            f"MAE={mae:.3f}\\nR2={r2:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.82},
        )

    fig.suptitle("Bandgap Prediction Scatter + Linear Fitting (OOF, 5-fold)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_recall(summary: pd.DataFrame, out_path: Path) -> None:
    x = np.arange(len(MODEL_ORDER))
    gap_vals = summary["gap_recall_macro"].to_numpy()
    eh_vals = summary["stability_recall_macro"].to_numpy()
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)

    for ax, vals, title, ylabel in [
        (axes[0], gap_vals, "Gap Type Recall (Macro)", "Macro Recall"),
        (axes[1], eh_vals, "Stability Recall (Macro)", "Macro Recall"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.25, ls="--")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Gap Type / Stability Recall Comparison", fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot bandgap scatter-fitting and recall for six models.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT,
        help="Output directory for csv and figures",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir = out_dir / "figures" / "model_bandgap_scatter"
    scatter_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.force_cpu)
    print(f"Using device: {device}")

    dfs = []
    dfs.append(evaluate_cgcnn(device))
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dfs.append(evaluate_mace(device))
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dfs.append(evaluate_crystal(device, model_type="alignn"))
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dfs.append(evaluate_crystal(device, model_type="m3gnet"))
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dfs.append(evaluate_dual(device, dual_type="m3gnet"))
    torch.cuda.empty_cache() if device.type == "cuda" else None

    dfs.append(evaluate_dual(device, dual_type="alignn"))
    torch.cuda.empty_cache() if device.type == "cuda" else None

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["model"] = pd.Categorical(all_df["model"], MODEL_ORDER, ordered=True)
    all_df = all_df.sort_values(["model", "fold"]).reset_index(drop=True)

    summary = compute_summary(all_df)

    # Save tables
    pred_csv = out_dir / "six_model_oof_predictions.csv"
    summary_csv = out_dir / "six_model_recall_summary.csv"
    all_df.to_csv(pred_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    # Plots
    scatter_grid_png = out_dir / "benchmark_bandgap_scatter_fits.png"
    recall_png = out_dir / "benchmark_recall_gaptype_stability.png"

    plot_scatter_grid(all_df, scatter_grid_png)
    plot_recall(summary, recall_png)

    for model in MODEL_ORDER:
        sub = all_df[all_df["model"] == model]
        out_png = scatter_dir / f"{model.replace('+', '_plus_')}_bandgap_scatter_fit.png"
        plot_one_scatter(sub, model, out_png)

    print("\nDone.")
    print(f"OOF predictions: {pred_csv}")
    print(f"Recall summary : {summary_csv}")
    print(f"Scatter grid   : {scatter_grid_png}")
    print(f"Recall figure  : {recall_png}")
    print(f"Per-model plots: {scatter_dir}")


if __name__ == "__main__":
    main()
