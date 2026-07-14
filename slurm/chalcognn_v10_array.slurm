#!/bin/bash
#SBATCH --job-name=ChalcoGNN_v10
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --signal=USR1@120
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/v10_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/v10_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/multitask
CKPT_DIR=${PROJECT}/checkpoints/multitask_v10

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/cgcnn:${PROJECT}/multitask:${PYTHONPATH}

echo "=== ChalcoGNN v10 — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Array Task: ${SLURM_ARRAY_TASK_ID}  Node: $(hostname)  Start: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

cd ${PROJECT}/multitask

python train_mt_v10.py "${DATA_DIR}" \
    --cache-dir "${DATA_DIR}/cached_graphs" \
    --checkpoint-dir "${CKPT_DIR}" \
    --fold ${FOLD} \
    --k-folds 5 \
    --seed 42 \
    --atom-fea-len 160 \
    --h-fea-len 320 \
    --n-conv 8 \
    --n-attn-heads 8 \
    --dropout 0.35 \
    --log-bg \
    --huber-delta 0.4 \
    --bg-reweight 0.3 \
    --focal-gamma-gt 4.0 \
    --focal-gamma-eh 2.0 \
    --label-smoothing 0.08 \
    --consistency-weight 0.05 \
    --augment-noise 0.03 \
    --augment-structure \
    --augment-perturb 0.03 \
    --augment-scale 0.02 \
    --augment-structure-prob 0.4 \
    --warmup-epochs 20 \
    --cosine-T0 150 \
    --cosine-Tmult 2 \
    --patience 150 \
    --epochs 800 \
    --batch-size 48 \
    --lr 0.0008 \
    --val-ratio 0.1 \
    --dynamic-weights \
    --mtl-clamp-min -3.0 \
    --mtl-clamp-max 5.0 \
    --use-weighted-sampler \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --gt-minority-boost 1.5 \
    --merge-metal-indirect \
    --robust-norm \
    --workers 8 \
    --print-freq 20

EXIT_CODE=$?
echo "Fold ${FOLD} exit code: ${EXIT_CODE}  End: $(date)"
