#!/bin/bash
#SBATCH --job-name=ALIGNN_mt
#SBATCH --account=a131
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/alignn_pyg_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/alignn_pyg_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/alignn_mt
FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== ALIGNN Multi-task (PyG) — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"

cd ${PROJECT}/multitask

python train_crystal_mt.py "${DATA_DIR}" \
    --model alignn \
    --cache-dir "${DATA_DIR}/crystal_cached_graphs" \
    --checkpoint-dir "${CKPT_DIR}" \
    --fold ${FOLD} \
    --k-folds 5 \
    --seed 42 \
    --hidden-dim 256 \
    --n-alignn-layers 4 \
    --n-gcn-layers 4 \
    --dropout 0.3 \
    --radius 8.0 \
    --max-num-nbr 12 \
    --dist-bins 80 \
    --angle-bins 40 \
    --epochs 500 \
    --batch-size 32 \
    --lr 0.0005 \
    --weight-decay 1e-4 \
    --patience 120 \
    --huber-delta 0.4 \
    --focal-gamma-gt 3.0 \
    --focal-gamma-eh 2.0 \
    --label-smoothing 0.06 \
    --consistency-weight 0.05 \
    --gt-minority-boost 2.0 \
    --log-bg \
    --merge-metal-indirect \
    --workers 4 \
    --print-freq 10

EXIT_CODE=$?
echo "Fold ${FOLD} exit code: ${EXIT_CODE}  End: $(date)"
