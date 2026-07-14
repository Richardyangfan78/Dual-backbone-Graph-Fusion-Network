#!/bin/bash
#SBATCH --job-name=MACE_ft_v2
#SBATCH --account=a131
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_ft_v2_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_ft_v2_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/mace_finetune_v2
PRETRAINED=${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== MACE Finetune v2 (frozen backbone) — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "Pre-trained: ${PRETRAINED}"

cd ${PROJECT}/multitask

# Key strategy changes vs v1:
#   - Backbone NEVER unfreezes (--unfreeze-epoch 9999)
#   - Higher head lr (0.001 vs 0.0003)
#   - More patience (150 vs 100) to compensate for noisy 202-sample val set
#   - Longer cosine period (T0=200) for head-only training
#   - Slightly higher dropout (0.4)
#   - Lower label smoothing (0.04) — avoid under-confident head with frozen backbone
python train_mace_mt.py "${DATA_DIR}" \
    --cache-dir "${DATA_DIR}/mace_cached_graphs" \
    --checkpoint-dir "${CKPT_DIR}" \
    --pretrained-checkpoint "${PRETRAINED}" \
    --fold ${FOLD} \
    --k-folds 5 \
    --seed 42 \
    --h-fea-len 256 \
    --n-attn-heads 8 \
    --dropout 0.4 \
    --log-bg \
    --huber-delta 0.4 \
    --focal-gamma-gt 3.0 \
    --focal-gamma-eh 2.0 \
    --label-smoothing 0.04 \
    --consistency-weight 0.05 \
    --gt-minority-boost 2.0 \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --warmup-epochs 15 \
    --cosine-T0 200 \
    --cosine-Tmult 1 \
    --patience 150 \
    --epochs 500 \
    --batch-size 32 \
    --lr 0.001 \
    --backbone-lr 0.000001 \
    --unfreeze-epoch 9999 \
    --unfreeze-layers 0 \
    --val-ratio 0.1 \
    --merge-metal-indirect \
    --mtl-clamp-min -3.0 \
    --mtl-clamp-max 5.0 \
    --mace-model small \
    --workers 4 \
    --print-freq 10

EXIT_CODE=$?
echo "Fold ${FOLD} exit code: ${EXIT_CODE}  End: $(date)"
