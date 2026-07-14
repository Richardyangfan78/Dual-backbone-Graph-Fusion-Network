#!/usr/bin/env python3
"""Export six-model OOF predictions and paper-style comparison figures.

This script is intended to run from the repository root or with PROJECT_ROOT
pointing to that root. It uses the aligned split files and the best-BG
MACE+ALIGNN checkpoint directory requested for reporting.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score


PROJECT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
MULTITASK = PROJECT / "multitask"
DATA_DIR = PROJECT / "Data" / "Inorganic_datasets"
CKPT_BASE = PROJECT / "checkpoints"

MODEL_ORDER = [
    "CGCNN",
    "ALIGNN",
    "M3GNet",
    "MACE",
    "MACE+M3GNet",
    "MACE+ALIGNN",
]

CKPT_DIRS = {
    "CGCNN": CKPT_BASE / "cgcnn_mt_inorg",
    "ALIGNN": CKPT_BASE / "alignn_mt_inorg_before_newsplit_rerun_20260705_052653",
    "M3GNet": CKPT_BASE / "m3gnet_mt_inorg",
    "MACE": CKPT_BASE / "mace_mt_inorg_pretrained_aligned",
    "MACE+M3GNet": CKPT_BASE / "mace_m3gnet_inorg_oldm3gnet_20260705_081640",
    "MACE+ALIGNN": CKPT_BASE / "dual_backbone_inorg_preAlignSplit_20260702_080248",
}

FOLD_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
MODEL_TITLES = {
    "CGCNN": "CGCNN",
    "ALIGNN": "ALIGNN",
    "M3GNet": "M3GNet",
    "MACE": "MACE",
    "MACE+M3GNet": "MACE+M3GNet",
    "MACE+ALIGNN": "MACE+ALIGNN",
}


class SimpleNormalizer:
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std)

    def denorm(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * (self.std + 1e-8) + self.mean


def setup_paths() -> None:
    sys.path.insert(0, str(MULTITASK))
    sys.path.insert(0, str(PROJECT / "cgcnn"))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_metric_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    return {str(row["metric"]).strip(): float(row["value"]) for _, row in df.iterrows()}


def collect_fold_metrics(out_dir: Path) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        ckpt_dir = CKPT_DIRS[model]
        for fold in range(5):
            metrics = read_metric_csv(ckpt_dir / f"fold{fold}_results.csv")
            rows.append({"model": model, "fold": fold, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "six_model_fold_metrics_best_bg_mace_alignn.csv", index=False)
    return df


def old_crystal_split(dataset, k_folds: int = 5, fold: int = 0, val_ratio: float = 0.1, seed: int = 42):
    """Original train_crystal_mt split used by the reported ALIGNN checkpoint.

    data_crystal_mt.py was later patched to prefer splits_mace_alignn when the
    files exist. The standalone ALIGNN checkpoint in CKPT_DIRS predates that
    patch; using the aligned split for this checkpoint leaks train samples into
    the test fold and does not reproduce fold*_results.csv.
    """
    from sklearn.model_selection import StratifiedKFold

    known_idx = [i for i, sample in enumerate(dataset.samples) if sample["gt"] >= 0]
    unknown_idx = [i for i, sample in enumerate(dataset.samples) if sample["gt"] < 0]
    known_labels = [dataset.samples[i]["gt"] for i in known_idx]
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    rel_train_val, rel_test = list(skf.split(known_idx, known_labels))[fold]

    test_idx = [known_idx[i] for i in rel_test]
    train_val_known = [known_idx[i] for i in rel_train_val]
    n_val = max(1, int(len(train_val_known) * val_ratio))
    rng = np.random.RandomState(seed + fold)
    rng.shuffle(train_val_known)
    val_idx = train_val_known[:n_val]
    train_idx = train_val_known[n_val:] + unknown_idx
    return train_idx, val_idx, test_idx


def append_rows(rows: list[dict], model: str, fold: int, ids, bg_true, bg_pred, gt_true, gt_pred, eh_true, eh_pred) -> None:
    for i, mid in enumerate(ids):
        rows.append(
            {
                "model": model,
                "fold": fold,
                "material_id": str(mid),
                "bg_true": float(bg_true[i]),
                "bg_pred": float(bg_pred[i]),
                "gt_true": int(gt_true[i]),
                "gt_pred": int(gt_pred[i]),
                "eh_true": int(eh_true[i]),
                "eh_pred": int(eh_pred[i]),
            }
        )


@torch.no_grad()
def export_cgcnn(rows: list[dict], device: torch.device) -> None:
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import SubsetRandomSampler
    from train_mt_v10 import (
        CachedGraphDatasetV7,
        CrystalGraphConvNetMTV10,
        Normalizer,
        apply_rules,
        collate_pool_mt_v7,
        load_aligned_split_indices,
    )

    ckpt_root = CKPT_DIRS["CGCNN"]
    sample_ckpt = torch.load(
        ckpt_root / "fold_0_seed_42" / "best_composite.pth.tar",
        map_location="cpu",
        weights_only=False,
    )
    args = sample_ckpt["args"]
    with open(Path(args["data_dir"]) / "id_prop.csv") as f:
        id_prop_data = list(csv.reader(f))
    random.seed(123)
    random.shuffle(id_prop_data)

    dataset = CachedGraphDatasetV7(
        args["cache_dir"],
        id_prop_data,
        merge_metal_indirect=args.get("merge_metal_indirect", True),
        augment_noise=0.0,
        data_dir=None,
        augment_perturb=args.get("augment_perturb", 0.03),
        augment_scale=args.get("augment_scale", 0.02),
        augment_structure_prob=0.0,
    )
    dataset.augment = False

    (sample_atom, sample_nbr, _), _, _ = dataset[0]
    for fold in range(5):
        ckpt = torch.load(
            ckpt_root / f"fold_{fold}_seed_42" / "best_composite.pth.tar",
            map_location=device,
            weights_only=False,
        )
        args = ckpt["args"]
        _, _, test_idx = load_aligned_split_indices(dataset.id_prop_data, args["data_dir"], fold)
        loader = DataLoader(
            dataset,
            sampler=SubsetRandomSampler(test_idx),
            batch_size=64,
            collate_fn=collate_pool_mt_v7,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model = CrystalGraphConvNetMTV10(
            sample_atom.shape[-1],
            sample_nbr.shape[-1],
            atom_fea_len=args["atom_fea_len"],
            n_conv=args["n_conv"],
            h_fea_len=args["h_fea_len"],
            n_h=args["n_h"],
            n_gap_classes=dataset.n_gap_classes,
            n_eh_classes=2,
            dropout=args["dropout"],
            n_attn_heads=args["n_attn_heads"],
            use_cosine_classifier=args["use_cosine_classifier"],
            cosine_temp=args["cosine_temp"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        normalizer = Normalizer()
        normalizer.load_state_dict(ckpt["normalizer_bg"])

        for inputs, targets, ids in loader:
            atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            bg_p, gt_logp, eh_logp = model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            bg_pred = normalizer.denorm(bg_p.detach().cpu()).reshape(-1).numpy()
            bg_true = targets["bandgap"].reshape(-1).numpy()
            gt_pred = apply_rules(torch.from_numpy(bg_pred), gt_logp.detach().cpu(), True).numpy()
            gt_true = targets["gaptype"].reshape(-1).numpy()
            eh_pred = eh_logp.argmax(1).detach().cpu().numpy()
            eh_true = targets["eh_label"].reshape(-1).numpy()
            append_rows(rows, "CGCNN", fold, ids, bg_true, bg_pred, gt_true, gt_pred, eh_true, eh_pred)
        print(f"CGCNN fold {fold}: {len(test_idx)} OOF rows")


@torch.no_grad()
def export_crystal_model(rows: list[dict], model_name: str, device: torch.device) -> None:
    from torch_geometric.loader import DataLoader
    from data_crystal_mt import CrystalMultiTaskDataset, stratified_kfold_split
    from model_alignn_pyg import ALIGNNMultiTaskPyG
    from model_m3gnet_pyg import M3GNetMultiTaskPyG

    ckpt_dir = CKPT_DIRS[model_name]
    sample_ckpt = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args = sample_ckpt["args"]
    dataset = CrystalMultiTaskDataset(
        root_dir=args["data_dir"],
        cache_dir=args.get("cache_dir"),
        radius=args.get("radius", 8.0),
        max_num_nbr=args.get("max_num_nbr", 12),
        dist_bins=args.get("dist_bins", 80),
        angle_bins=args.get("angle_bins", 40),
        log_bg=args.get("log_bg", True),
        merge_metal_indirect=args.get("merge_metal_indirect", True),
        eh_threshold=args.get("eh_threshold", 0.1),
    )

    for fold in range(5):
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)
        args = ckpt["args"]
        if model_name == "ALIGNN":
            _, _, test_idx = old_crystal_split(
                dataset,
                k_folds=args.get("k_folds", 5),
                fold=fold,
                val_ratio=args.get("val_ratio", 0.1),
                seed=args.get("seed", 42),
            )
        else:
            _, _, test_idx = stratified_kfold_split(
                dataset,
                k_folds=args.get("k_folds", 5),
                fold=fold,
                val_ratio=args.get("val_ratio", 0.15),
                seed=args.get("seed", 42),
            )
        test_set = [dataset[i] for i in test_idx]
        test_ids = [dataset.samples[i]["id"] for i in test_idx]
        loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0, follow_batch=["x"])

        if model_name == "ALIGNN":
            model = ALIGNNMultiTaskPyG(
                atom_input_dim=94,
                edge_input_dim=args.get("dist_bins", 80),
                angle_input_dim=args.get("angle_bins", 40),
                hidden_dim=args.get("hidden_dim", 256),
                n_alignn_layers=args.get("n_alignn_layers", 4),
                n_gcn_layers=args.get("n_gcn_layers", 4),
                n_gap_classes=2,
                n_eh_classes=2,
                dropout=args.get("dropout", 0.3),
            ).to(device)
        else:
            model = M3GNetMultiTaskPyG(
                atom_input_dim=94,
                edge_input_dim=args.get("dist_bins", 80),
                angle_input_dim=args.get("angle_bins", 40),
                hidden_dim=args.get("hidden_dim", 128),
                n_blocks=args.get("n_blocks", 3),
                n_gap_classes=2,
                n_eh_classes=2,
                dropout=args.get("dropout", 0.3),
            ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        normalizer = SimpleNormalizer(ckpt["normalizer_mean"], ckpt["normalizer_std"])

        cursor = 0
        for batch in loader:
            batch = batch.to(device)
            bg_p, gt_logp, eh_logp = model(batch)
            bg_log_pred = normalizer.denorm(bg_p.detach().cpu().reshape(-1))
            bg_log_true = batch.bg.detach().cpu().reshape(-1)
            if args.get("log_bg", True):
                bg_pred = torch.expm1(bg_log_pred).numpy()
                bg_true = torch.expm1(bg_log_true).numpy()
            else:
                bg_pred = bg_log_pred.numpy()
                bg_true = bg_log_true.numpy()
            n = len(bg_pred)
            ids = test_ids[cursor : cursor + n]
            cursor += n
            append_rows(
                rows,
                model_name,
                fold,
                ids,
                bg_true,
                bg_pred,
                batch.gt.detach().cpu().reshape(-1).numpy(),
                gt_logp.argmax(1).detach().cpu().numpy(),
                batch.eh.detach().cpu().reshape(-1).numpy(),
                eh_logp.argmax(1).detach().cpu().numpy(),
            )
        print(f"{model_name} fold {fold}: {len(test_idx)} OOF rows")


@torch.no_grad()
def export_mace_family(rows: list[dict], device: torch.device) -> None:
    from mace.calculators import mace_mp
    from torch_geometric.loader import DataLoader
    from data_mace_mt import MACECrystalDataset, stratified_kfold_split as mace_split
    from data_dual_backbone import DualBackboneDataset, stratified_kfold_split as dual_split
    from model_alignn_pyg import ALIGNNMultiTaskPyG
    from model_dual_backbone import DualBackboneMultiTask
    from model_m3gnet_pyg import M3GNetMultiTaskPyG
    from model_mace_m3gnet import DualBackboneMACEM3GNet
    from model_mace_mt import MACEMultiTask

    print("Loading MACE-MP small backbone")
    calc = mace_mp(model="small", default_dtype="float32", device=str(device))
    mace_base = calc.models[0]
    z_table = calc.z_table
    r_max = calc.r_max

    export_single_mace(rows, device, mace_base, z_table, r_max, DataLoader, MACECrystalDataset, mace_split)
    export_dual_alignn(rows, device, mace_base, DataLoader, DualBackboneDataset, dual_split, ALIGNNMultiTaskPyG, DualBackboneMultiTask)
    export_dual_m3gnet(rows, device, mace_base, DataLoader, DualBackboneDataset, dual_split, M3GNetMultiTaskPyG, DualBackboneMACEM3GNet)


def _denorm_logged_bg(bg_pred, bg_true, normalizer: SimpleNormalizer, log_bg: bool):
    pred = normalizer.denorm(bg_pred.detach().cpu().reshape(-1))
    true = bg_true.detach().cpu().reshape(-1)
    if log_bg:
        return torch.expm1(true).numpy(), torch.expm1(pred).numpy()
    return true.numpy(), pred.numpy()


@torch.no_grad()
def export_single_mace(rows, device, mace_base, z_table, r_max, DataLoader, MACECrystalDataset, mace_split) -> None:
    ckpt_dir = CKPT_DIRS["MACE"]
    sample_ckpt = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args0 = sample_ckpt["args"]
    dataset = MACECrystalDataset(
        args0["data_dir"],
        z_table,
        r_max,
        cache_dir=args0.get("cache_dir"),
        merge_metal_indirect=args0.get("merge_metal_indirect", True),
        log_bg=args0.get("log_bg", True),
    )

    from model_mace_mt import MACEMultiTask

    for fold in range(5):
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)
        args = ckpt["args"]
        _, _, test_idx = mace_split(
            dataset,
            k_folds=args.get("k_folds", 5),
            fold=fold,
            val_ratio=args.get("val_ratio", 0.15),
            seed=args.get("seed", 42),
        )
        test_set = [dataset[i] for i in test_idx]
        test_ids = [dataset.samples[i]["mp_id"] for i in test_idx]
        loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
        model = MACEMultiTask(
            mace_base,
            h_fea_len=args.get("h_fea_len", 256),
            n_gap_classes=2,
            dropout=args.get("dropout", 0.3),
            n_attn_heads=args.get("n_attn_heads", 8),
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        normalizer = SimpleNormalizer(ckpt["normalizer_mean"], ckpt["normalizer_std"])

        cursor = 0
        for batch in loader:
            batch = batch.to(device)
            bg_p, gt_logp, eh_logp = model(batch.to_dict())
            bg_true, bg_pred = _denorm_logged_bg(bg_p, batch.bg, normalizer, args.get("log_bg", True))
            n = len(bg_pred)
            ids = test_ids[cursor : cursor + n]
            cursor += n
            append_rows(
                rows,
                "MACE",
                fold,
                ids,
                bg_true,
                bg_pred,
                batch.gt.detach().cpu().reshape(-1).numpy(),
                gt_logp.argmax(1).detach().cpu().numpy(),
                batch.eh.detach().cpu().reshape(-1).numpy(),
                eh_logp.argmax(1).detach().cpu().numpy(),
            )
        print(f"MACE fold {fold}: {len(test_idx)} OOF rows")


@torch.no_grad()
def export_dual_alignn(rows, device, mace_base, DataLoader, DualBackboneDataset, dual_split, ALIGNNMultiTaskPyG, DualBackboneMultiTask) -> None:
    ckpt_dir = CKPT_DIRS["MACE+ALIGNN"]
    sample_ckpt = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args0 = sample_ckpt["args"]
    dataset = DualBackboneDataset(
        args0["data_dir"],
        args0["mace_cache_dir"],
        args0["crystal_cache_dir"],
        merge_metal_indirect=args0.get("merge_metal_indirect", True),
        log_bg=args0.get("log_bg", True),
    )

    for fold in range(5):
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)
        args = ckpt["args"]
        _, _, test_idx = dual_split(
            dataset,
            k_folds=args.get("k_folds", 5),
            fold=fold,
            val_ratio=args.get("val_ratio", 0.15),
            seed=args.get("seed", 42),
        )
        test_set = [dataset[i] for i in test_idx]
        test_ids = [dataset.samples[i]["mp_id"] for i in test_idx]
        loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
        alignn_model = ALIGNNMultiTaskPyG(
            atom_input_dim=94,
            edge_input_dim=80,
            angle_input_dim=40,
            hidden_dim=256,
            n_alignn_layers=4,
            n_gcn_layers=4,
            n_gap_classes=2,
            n_eh_classes=2,
            dropout=args.get("dropout", 0.3),
        )
        model = DualBackboneMultiTask(
            mace_base,
            alignn_model,
            h_fea_len=args.get("h_fea_len", 256),
            n_gap_classes=2,
            dropout=args.get("dropout", 0.3),
            n_attn_heads=args.get("n_attn_heads", 8),
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        normalizer = SimpleNormalizer(ckpt["normalizer_mean"], ckpt["normalizer_std"])

        cursor = 0
        for batch in loader:
            batch = batch.to(device)
            bg_p, gt_logp, eh_logp = model(batch)
            bg_true, bg_pred = _denorm_logged_bg(bg_p, batch.bg, normalizer, args.get("log_bg", True))
            n = len(bg_pred)
            ids = test_ids[cursor : cursor + n]
            cursor += n
            append_rows(
                rows,
                "MACE+ALIGNN",
                fold,
                ids,
                bg_true,
                bg_pred,
                batch.gt.detach().cpu().reshape(-1).numpy(),
                gt_logp.argmax(1).detach().cpu().numpy(),
                batch.eh.detach().cpu().reshape(-1).numpy(),
                eh_logp.argmax(1).detach().cpu().numpy(),
            )
        print(f"MACE+ALIGNN fold {fold}: {len(test_idx)} OOF rows")


@torch.no_grad()
def export_dual_m3gnet(rows, device, mace_base, DataLoader, DualBackboneDataset, dual_split, M3GNetMultiTaskPyG, DualBackboneMACEM3GNet) -> None:
    ckpt_dir = CKPT_DIRS["MACE+M3GNet"]
    sample_ckpt = torch.load(ckpt_dir / "fold0_best.pt", map_location="cpu", weights_only=False)
    args0 = sample_ckpt["args"]
    dataset = DualBackboneDataset(
        args0["data_dir"],
        args0["mace_cache_dir"],
        args0["crystal_cache_dir"],
        merge_metal_indirect=args0.get("merge_metal_indirect", True),
        log_bg=args0.get("log_bg", True),
    )

    for fold in range(5):
        ckpt = torch.load(ckpt_dir / f"fold{fold}_best.pt", map_location=device, weights_only=False)
        args = ckpt["args"]
        _, _, test_idx = dual_split(
            dataset,
            k_folds=args.get("k_folds", 5),
            fold=fold,
            val_ratio=args.get("val_ratio", 0.15),
            seed=args.get("seed", 42),
        )
        test_set = [dataset[i] for i in test_idx]
        test_ids = [dataset.samples[i]["mp_id"] for i in test_idx]
        loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
        m3gnet_model = M3GNetMultiTaskPyG(
            atom_input_dim=94,
            edge_input_dim=80,
            angle_input_dim=40,
            hidden_dim=128,
            n_blocks=3,
            n_gap_classes=2,
            n_eh_classes=2,
            dropout=args.get("dropout", 0.3),
        )
        model = DualBackboneMACEM3GNet(
            mace_base,
            m3gnet_model,
            h_fea_len=args.get("h_fea_len", 256),
            n_gap_classes=2,
            dropout=args.get("dropout", 0.3),
            n_attn_heads=args.get("n_attn_heads", 8),
            use_cosine_classifier=args.get("use_cosine_classifier", True),
            cosine_temp=args.get("cosine_temp", 0.1),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        normalizer = SimpleNormalizer(ckpt["normalizer_mean"], ckpt["normalizer_std"])

        cursor = 0
        for batch in loader:
            batch = batch.to(device)
            bg_p, gt_logp, eh_logp = model(batch)
            bg_true, bg_pred = _denorm_logged_bg(bg_p, batch.bg, normalizer, args.get("log_bg", True))
            n = len(bg_pred)
            ids = test_ids[cursor : cursor + n]
            cursor += n
            append_rows(
                rows,
                "MACE+M3GNet",
                fold,
                ids,
                bg_true,
                bg_pred,
                batch.gt.detach().cpu().reshape(-1).numpy(),
                gt_logp.argmax(1).detach().cpu().numpy(),
                batch.eh.detach().cpu().reshape(-1).numpy(),
                eh_logp.argmax(1).detach().cpu().numpy(),
            )
        print(f"MACE+M3GNet fold {fold}: {len(test_idx)} OOF rows")


def export_oof(out_dir: Path, reuse: bool = False) -> pd.DataFrame:
    oof_path = out_dir / "six_model_oof_predictions_best_bg_mace_alignn.csv"
    if reuse and oof_path.exists():
        print(f"Reusing {oof_path}")
        return pd.read_csv(oof_path)

    setup_paths()
    device = get_device()
    print(f"Device: {device}")
    rows: list[dict] = []
    export_cgcnn(rows, device)
    export_crystal_model(rows, "ALIGNN", device)
    export_crystal_model(rows, "M3GNet", device)
    export_mace_family(rows, device)
    df = pd.DataFrame(rows)
    df.to_csv(oof_path, index=False)
    print(f"Saved {len(df)} OOF rows to {oof_path}")
    return df


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.jpg", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_fitting(oof: pd.DataFrame, out_dir: Path) -> None:
    all_true = oof["bg_true"].to_numpy(float)
    all_pred = oof["bg_pred"].to_numpy(float)
    lo = min(0.0, np.nanmin([all_true.min(), all_pred.min()])) - 0.05
    hi = np.nanmax([all_true.max(), all_pred.max()]) + 0.20

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.1), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        m = oof[oof["model"] == model]
        for fold in range(5):
            mf = m[m["fold"] == fold]
            ax.scatter(
                mf["bg_true"],
                mf["bg_pred"],
                s=9,
                alpha=0.45,
                color=FOLD_COLORS[fold],
                edgecolors="none",
                label=f"Fold {fold}",
            )
        ax.plot([lo, hi], [lo, hi], color="#222222", lw=0.9, ls="--", label="Ideal")
        mae = np.mean(np.abs(m["bg_true"].to_numpy(float) - m["bg_pred"].to_numpy(float)))
        r2 = r2_score(m["bg_true"], m["bg_pred"])
        ax.text(
            0.04,
            0.95,
            f"MAE = {mae:.3f} eV\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#D0D0D0", alpha=0.9),
        )
        ax.set_title(MODEL_TITLES[model], fontsize=9, fontweight="bold")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, lw=0.35, color="#D8D8D8", alpha=0.7)

    for ax in axes[3:]:
        ax.set_xlabel("DFT bandgap (eV)")
    for ax in axes[::3]:
        ax.set_ylabel("Predicted bandgap (eV)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Bandgap fitting across reported five-fold test splits", y=0.995, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save_figure(fig, out_dir, "six_model_fitting_best_bg_mace_alignn")


def plot_metric_bars(metrics: pd.DataFrame, out_dir: Path, task: str) -> None:
    if task == "gt":
        acc_col, f1_col = "gt_acc", "gt_f1"
        title = "Gap-type classification across reported five-fold test splits"
        stem = "six_model_gap_type_best_bg_mace_alignn"
        acc_color, f1_color = "#6EA9CB", "#99BFAB"
    else:
        acc_col, f1_col = "eh_acc", "eh_f1"
        title = "Stability classification across reported five-fold test splits"
        stem = "six_model_stability_best_bg_mace_alignn"
        acc_color, f1_color = "#CD97B1", "#9FADED"

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), sharey=True)
    axes = axes.ravel()
    x = np.arange(5)
    width = 0.34
    for ax, model in zip(axes, MODEL_ORDER):
        m = metrics[metrics["model"] == model].sort_values("fold")
        acc = m[acc_col].to_numpy(float)
        f1 = m[f1_col].to_numpy(float)
        bars_acc = ax.bar(x - width / 2, acc, width, color=acc_color, label="Accuracy", alpha=0.9)
        bars_f1 = ax.bar(x + width / 2, f1, width, color=f1_color, label="F1", alpha=0.9)
        mean_acc = float(np.mean(acc))
        mean_f1 = float(np.mean(f1))
        ax.axhline(mean_acc, color=acc_color, lw=0.8, ls="--", alpha=0.85)
        ax.axhline(mean_f1, color=f1_color, lw=0.8, ls=":", alpha=0.90)
        for bar in list(bars_acc) + list(bars_f1):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                rotation=90,
            )
        ax.text(
            0.02,
            0.05,
            f"mean Acc={mean_acc:.3f}\nmean F1={mean_f1:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
        )
        ax.set_title(MODEL_TITLES[model], fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i}" for i in x])
        ax.set_ylim(0, 1.06)
        ax.grid(axis="y", lw=0.35, color="#D8D8D8", alpha=0.7)

    for ax in axes[3:]:
        ax.set_xlabel("Test fold")
    for ax in axes[::3]:
        ax.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(title, y=0.995, fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save_figure(fig, out_dir, stem)


def validate_oof_against_metrics(oof: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        for fold in range(5):
            pred = oof[(oof["model"] == model) & (oof["fold"] == fold)]
            metric = metrics[(metrics["model"] == model) & (metrics["fold"] == fold)].iloc[0]
            mae = np.mean(np.abs(pred["bg_true"].to_numpy(float) - pred["bg_pred"].to_numpy(float)))
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "oof_bg_mae": mae,
                    "reported_bg_mae": float(metric["bg_mae"]),
                    "abs_diff": abs(mae - float(metric["bg_mae"])),
                    "n": int(len(pred)),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "six_model_oof_metric_check_best_bg_mace_alignn.csv", index=False)
    print(df.to_string(index=False))
    max_diff = float(df["abs_diff"].max())
    if max_diff > 5e-3:
        print(f"WARNING: max OOF vs reported BG MAE difference is {max_diff:.6f}")
    else:
        print(f"OOF BG MAE matches reported metrics within {max_diff:.6f}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results" / "best_bg_six_model_figures")
    parser.add_argument("--reuse-oof", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    metrics = collect_fold_metrics(args.out_dir)
    if args.figures_only:
        oof = pd.read_csv(args.out_dir / "six_model_oof_predictions_best_bg_mace_alignn.csv")
    else:
        oof = export_oof(args.out_dir, reuse=args.reuse_oof)

    validate_oof_against_metrics(oof, metrics, args.out_dir)
    plot_fitting(oof, args.out_dir)
    plot_metric_bars(metrics, args.out_dir, "gt")
    plot_metric_bars(metrics, args.out_dir, "eh")
    print(f"Figures and source data saved to {args.out_dir}")


if __name__ == "__main__":
    main()
