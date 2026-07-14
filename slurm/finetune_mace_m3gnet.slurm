#!/bin/bash
#SBATCH --job-name=MACE_M3GNet
#SBATCH --account=a131
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gh
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_m3gnet_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_m3gnet_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/mace_m3gnet
MACE_PRETRAINED=${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt
M3GNET_CKPT_DIR=${PROJECT}/checkpoints/m3gnet_mt

FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export HF_HOME=/path/to/huggingface-cache

echo "=== MACE+M3GNet Ablation — Fold ${FOLD} ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"

cd ${PROJECT}/multitask

python train_mace_m3gnet.py "${DATA_DIR}" \
    --mace-cache-dir    "${DATA_DIR}/mace_cached_graphs" \
    --crystal-cache-dir "${DATA_DIR}/crystal_cached_graphs" \
    --checkpoint-dir    "${CKPT_DIR}" \
    --mace-pretrained   "${MACE_PRETRAINED}" \
    --m3gnet-checkpoint-dir "${M3GNET_CKPT_DIR}" \
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
    --unfreeze-epoch 30 \
    --unfreeze-epoch-2 50 \
    --unfreeze-layers 1 \
    --mace-backbone-lr 5e-6 \
    --m3gnet-backbone-lr 1e-5 \
    --epochs 300 \
    --batch-size 32 \
    --lr 5e-4 \
    --workers 4 \
    --print-freq 10

echo "Done: $(date)"
