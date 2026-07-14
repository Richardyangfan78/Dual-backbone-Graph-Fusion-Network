# Reviewer Reproducibility Workflow

This repository supports two levels of reproduction.

## 1. Verify Reported Metrics And Screening Counts

This route does not require GPUs or model checkpoints.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall multitask scripts classical_ml_baseline
```

Inspect:

- `MODEL_STATUS.md` for the official checkpoint provenance and summary metrics.
- `results/best_bg_six_model_figures/six_model_fold_metrics_best_bg_mace_alignn.csv` for fold-level metrics.
- `results/best_bg_six_model_figures/six_model_oof_predictions_best_bg_mace_alignn.csv` for out-of-fold predictions.
- `Data/predict_final_main6_manifest.csv` for the 12,018-compound screening library.
- `exports/final_candidates_screen_direct_stable_bg_0p5_1p1/final_candidates_453_manifest.csv` for the final candidate set.

## 2. Rebuild The Dual-Backbone Model

Use a CUDA-capable environment. The exact package versions on the original CSCS run were controlled by the cluster environment; install CUDA-compatible `torch` and `torch-geometric` wheels for your system before installing the remaining packages.

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT/multitask:$PYTHONPATH
export DATA=$PROJECT_ROOT/Data/Inorganic_datasets
mkdir -p checkpoints logs
```

Train or provide backbone checkpoints:

- MACE pretraining output: `checkpoints/mace_pretrain/pretrained_best.pt`
- ALIGNN fold checkpoints: `checkpoints/alignn_mt_inorg/fold0_best.pt` through `fold4_best.pt`

Then run the fusion model:

```bash
for FOLD in 0 1 2 3 4; do
  python multitask/train_dual_backbone.py "$DATA" \
    --mace-cache-dir "$DATA/mace_cached_graphs" \
    --crystal-cache-dir "$DATA/crystal_cached_graphs" \
    --checkpoint-dir checkpoints/dual_backbone_inorg \
    --mace-pretrained checkpoints/mace_pretrain/pretrained_best.pt \
    --alignn-checkpoint-dir checkpoints/alignn_mt_inorg \
    --fold "$FOLD" --k-folds 5 --seed 42 \
    --h-fea-len 256 --n-attn-heads 8 --dropout 0.3 --log-bg \
    --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0 \
    --label-smoothing 0.06 --consistency-weight 0.05 \
    --gt-minority-boost 2.0 --use-cosine-classifier --cosine-temp 0.1 \
    --warmup-epochs 10 --cosine-T0 120 --cosine-Tmult 2 --patience 80 \
    --epochs 300 --batch-size 32 --lr 0.0005 \
    --mace-backbone-lr 0.000005 --alignn-backbone-lr 0.00001 \
    --unfreeze-epoch 30 --unfreeze-epoch-2 50 --unfreeze-layers 1 \
    --val-ratio 0.15 --min-save-epoch 30 \
    --gt-composite-weight 0.5 --eh-composite-weight 0.3 \
    --cond-anneal-epochs 40 --merge-metal-indirect \
    --mtl-clamp-min -3.0 --mtl-clamp-max 5.0 \
    --mace-model small --workers 4 --print-freq 10
done
```

The script writes `fold{N}_results.csv` and `fold{N}_best.pt` under the selected checkpoint directory.

## 3. Run Inference On New Structures

```bash
python scripts/predict_inorg_dual_backbone.py path/to/cifs predictions.csv \
  --checkpoint-dir checkpoints/dual_backbone_inorg \
  --alignn-checkpoint-dir checkpoints/alignn_mt_inorg \
  --device cuda
```

The output columns are `formula`, `bg_type`, `bg_eV`, `bg_std_eV`, `ehull`, `screen_pass`, and `source`.
