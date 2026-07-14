#!/bin/bash
#SBATCH --job-name=MACE_finetune
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --signal=USR1@120
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_ft_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_ft_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/mace_finetune
PRETRAINED=${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== MACE Fine-tune — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "Pre-trained model: ${PRETRAINED}"

cd ${PROJECT}/multitask

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
    --patience 100 \
    --epochs 300 \
    --batch-size 32 \
    --lr 0.0003 \
    --backbone-lr 0.000005 \
    --unfreeze-epoch 20 \
    --unfreeze-layers 1 \
    --val-ratio 0.1 \
    --merge-metal-indirect \
    --mtl-clamp-min -3.0 \
    --mtl-clamp-max 5.0 \
    --mace-model small \
    --workers 4 \
    --print-freq 10

EXIT_CODE=$?
echo "Fold ${FOLD} exit code: ${EXIT_CODE}  End: $(date)"
