#!/bin/bash
#SBATCH --job-name=MACE_ft_v3
#SBATCH --account=a131
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_ft_v3_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_ft_v3_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/mace_finetune_v3
PRETRAINED=${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== MACE Finetune v3 (5 fixes) — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "Pre-trained: ${PRETRAINED}"

cd ${PROJECT}/multitask

# v3 strategy:
#   Fix 1: min_save_epoch=30, GT composite weight=0.5, EH=0.3
#   Fix 2: Progressive unfreeze: last 1 layer @30, all @50
#   Fix 3: LR=0.0005, weight_decay=5e-4, grad_clip=2.0
#   Fix 4: GT conditioning annealing over 40 epochs
#   Fix 5: val_ratio=0.15 (303 samples vs 202)
python train_mace_mt.py "${DATA_DIR}" \
    --cache-dir "${DATA_DIR}/mace_cached_graphs" \
    --checkpoint-dir "${CKPT_DIR}" \
    --pretrained-checkpoint "${PRETRAINED}" \
    --fold ${FOLD} \
    --k-folds 5 \
    --seed 42 \
    --h-fea-len 256 \
    --n-attn-heads 8 \
    --dropout 0.3 \
    --log-bg \
    --huber-delta 0.4 \
    --focal-gamma-gt 3.0 \
    --focal-gamma-eh 2.0 \
    --label-smoothing 0.06 \
    --consistency-weight 0.05 \
    --gt-minority-boost 2.0 \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --warmup-epochs 10 \
    --cosine-T0 120 \
    --cosine-Tmult 2 \
    --patience 80 \
    --epochs 300 \
    --batch-size 32 \
    --lr 0.0005 \
    --backbone-lr 0.000005 \
    --unfreeze-epoch 30 \
    --unfreeze-epoch-2 50 \
    --unfreeze-layers 1 \
    --val-ratio 0.15 \
    --min-save-epoch 30 \
    --gt-composite-weight 0.5 \
    --eh-composite-weight 0.3 \
    --cond-anneal-epochs 40 \
    --merge-metal-indirect \
    --mtl-clamp-min -3.0 \
    --mtl-clamp-max 5.0 \
    --mace-model small \
    --workers 4 \
    --print-freq 10

EXIT_CODE=$?
echo "Fold ${FOLD} exit code: ${EXIT_CODE}  End: $(date)"
