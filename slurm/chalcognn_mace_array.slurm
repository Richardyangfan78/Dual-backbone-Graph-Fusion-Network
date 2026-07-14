#!/bin/bash
#SBATCH --job-name=ChalcoGNN_MACE
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --signal=USR1@120
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/mace_mt

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}

echo "=== ChalcoGNN MACE — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}  Node: $(hostname)  Start: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python -c "import mace; print('MACE:', mace.__version__); import e3nn; print('e3nn:', e3nn.__version__)"

cd ${PROJECT}/multitask

python train_mace_mt.py "${DATA_DIR}" \
    --cache-dir "${DATA_DIR}/mace_cached_graphs" \
    --checkpoint-dir "${CKPT_DIR}" \
    --fold ${FOLD} \
    --k-folds 5 \
    --seed 42 \
    --h-fea-len 256 \
    --n-attn-heads 8 \
    --dropout 0.3 \
    --log-bg \
    --huber-delta 0.4 \
    --focal-gamma-gt 4.0 \
    --focal-gamma-eh 2.0 \
    --label-smoothing 0.08 \
    --consistency-weight 0.05 \
    --gt-minority-boost 1.5 \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --warmup-epochs 10 \
    --cosine-T0 100 \
    --cosine-Tmult 2 \
    --patience 80 \
    --epochs 300 \
    --batch-size 32 \
    --lr 0.0005 \
    --backbone-lr 0.00001 \
    --unfreeze-epoch 30 \
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
