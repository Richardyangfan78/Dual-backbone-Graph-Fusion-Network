#!/usr/bin/env python3
"""Evaluate released DBGFN checkpoints on the fixed five-fold test splits."""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from data_crystal_mt import CrystalMultiTaskDataset
from data_dual_backbone import DualBackboneDataset, stratified_kfold_split
from data_mace_mt import MACECrystalDataset
from model_alignn_pyg import ALIGNNMultiTaskPyG
from model_dual_backbone import DualBackboneMultiTask


ROOT = Path(__file__).resolve().parents[1]
ALIGNN_KWARGS = {
    "hidden_dim": 256,
    "n_alignn_layers": 4,
    "n_gcn_layers": 4,
    "dropout": 0.3,
    "edge_input_dim": 80,
    "angle_input_dim": 40,
    "n_gap_classes": 2,
    "n_eh_classes": 2,
}
DUAL_KWARGS = {
    "h_fea_len": 256,
    "n_attn_heads": 8,
    "dropout": 0.3,
    "n_gap_classes": 2,
    "n_eh_classes": 2,
    "use_cosine_classifier": True,
    "cosine_temp": 0.1,
}


def extract_structures(data_dir: Path) -> None:
    cif_dir = data_dir / "cifs"
    expected = sum(1 for _ in (data_dir / "id_prop.csv").open(encoding="utf-8"))
    if cif_dir.exists() and len(list(cif_dir.glob("*.cif"))) == expected:
        return
    archive = data_dir / "structures.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"Missing {archive}")
    data_root = data_dir.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (data_dir / member.name).resolve()
            if data_root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        bundle.extractall(data_dir)
    found = len(list(cif_dir.glob("*.cif")))
    if found != expected:
        raise RuntimeError(f"Expected {expected} CIFs after extraction; found {found}.")


def build_graph_cache(data_dir: Path, mace_base) -> tuple[Path, Path]:
    from mace.tools import AtomicNumberTable

    mace_cache = data_dir / "mace_cached_graphs"
    alignn_cache = data_dir / "alignn_cached_graphs"
    expected = sum(1 for _ in (data_dir / "id_prop.csv").open(encoding="utf-8"))
    z_table = AtomicNumberTable([int(z) for z in mace_base.atomic_numbers])

    if len(list(mace_cache.glob("*.pt"))) != expected:
        print("Building MACE graph cache...")
        dataset = MACECrystalDataset(
            str(data_dir), z_table=z_table, r_max=6.0, cache_dir=str(mace_cache)
        )
        for index in range(len(dataset)):
            _ = dataset[index]
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"  MACE {index + 1}/{len(dataset)}")

    if len(list(alignn_cache.glob("*.pt"))) != expected:
        print("Building ALIGNN graph cache...")
        dataset = CrystalMultiTaskDataset(
            str(data_dir),
            cache_dir=str(alignn_cache),
            radius=8.0,
            max_num_nbr=12,
            dist_bins=80,
            angle_bins=40,
        )
        for index in range(len(dataset)):
            _ = dataset[index]
            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"  ALIGNN {index + 1}/{len(dataset)}")
    return mace_cache, alignn_cache


def macro_f1(targets: list[int], predictions: list[int]) -> float:
    scores = []
    for label in sorted(set(targets) | set(predictions)):
        tp = sum(a == label and b == label for a, b in zip(targets, predictions))
        fp = sum(a != label and b == label for a, b in zip(targets, predictions))
        fn = sum(a == label and b != label for a, b in zip(targets, predictions))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_fold(model, loader, mean: float, std: float, device):
    bg_true, bg_pred, gt_true, gt_pred, eh_true, eh_pred = [], [], [], [], [], []
    model.eval()
    model.set_cond_weight(1.0)
    for batch in loader:
        batch = batch.to(device)
        bg_out, gt_out, eh_out = model(batch)
        predicted_log_bg = bg_out.squeeze(-1) * (std + 1e-8) + mean
        bg_pred.extend(torch.expm1(predicted_log_bg).cpu().tolist())
        bg_true.extend(torch.expm1(batch.bg).cpu().tolist())
        gt_pred.extend(gt_out.argmax(dim=1).cpu().tolist())
        gt_true.extend(batch.gt.cpu().tolist())
        eh_pred.extend(eh_out.argmax(dim=1).cpu().tolist())
        eh_true.extend(batch.eh.cpu().tolist())
    return {
        "bg_mae": sum(abs(a - b) for a, b in zip(bg_true, bg_pred)) / len(bg_true),
        "gt_acc": sum(a == b for a, b in zip(gt_true, gt_pred)) / len(gt_true),
        "gt_f1": macro_f1(gt_true, gt_pred),
        "eh_acc": sum(a == b for a, b in zip(eh_true, eh_pred)) / len(eh_true),
        "eh_f1": macro_f1(eh_true, eh_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "Data" / "Dataset")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "Checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "Results" / "reproduced_checkpoint_metrics.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="extract CIFs and build both graph caches without evaluating checkpoints",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    extract_structures(args.data_dir)

    from mace.calculators import mace_mp

    print(f"Loading MACE-MP-0 small on {device}...")
    mace_base = mace_mp(
        model="small", default_dtype="float32", device=str(device)
    ).models[0]
    mace_cache, alignn_cache = build_graph_cache(args.data_dir, mace_base)
    if args.prepare_only:
        print(f"Graph caches are ready: {mace_cache} and {alignn_cache}")
        return
    dataset = DualBackboneDataset(
        str(args.data_dir), str(mace_cache), str(alignn_cache)
    )

    rows = []
    for fold in args.folds:
        checkpoint_path = args.checkpoint_dir / f"fold{fold}_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing {checkpoint_path}; run python Checkpoint/download.py"
            )
        _, _, test_indices = stratified_kfold_split(dataset, fold=fold, seed=42)
        loader = DataLoader(
            [dataset[index] for index in test_indices],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        alignn_model = ALIGNNMultiTaskPyG(**ALIGNN_KWARGS)
        model = DualBackboneMultiTask(mace_base, alignn_model, **DUAL_KWARGS)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        metrics = evaluate_fold(
            model,
            loader,
            float(checkpoint["normalizer_mean"]),
            float(checkpoint["normalizer_std"]),
            device,
        )
        rows.append({"fold": fold, **metrics})
        print(
            f"fold {fold}: BG MAE={metrics['bg_mae']:.6f}, "
            f"GT F1={metrics['gt_f1']:.6f}, TS F1={metrics['eh_f1']:.6f}"
        )
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "bg_mae", "gt_acc", "gt_f1", "eh_acc", "eh_f1"])
        writer.writeheader()
        writer.writerows(rows)
    means = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("bg_mae", "gt_acc", "gt_f1", "eh_acc", "eh_f1")
    }
    print(
        "mean: "
        f"BG MAE={means['bg_mae']:.6f}, GT Acc={means['gt_acc']:.6f}, "
        f"GT F1={means['gt_f1']:.6f}, TS Acc={means['eh_acc']:.6f}, "
        f"TS F1={means['eh_f1']:.6f}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
