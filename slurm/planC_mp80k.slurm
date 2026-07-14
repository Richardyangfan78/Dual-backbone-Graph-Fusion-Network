#!/bin/bash
#SBATCH --job-name=planC_80k
#SBATCH --account=a131
#SBATCH --array=0-4
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/planC_80k_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/planC_80k_fold%a_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/mp_bandgap_v2
CKPT_DIR=${PROJECT}/checkpoints/mace_transformer_80k
PRETRAINED=${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt

FOLD=${SLURM_ARRAY_TASK_ID}

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== Plan C (80K MP): MACE + Graph Transformer — Fold ${FOLD} ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "Dataset: ${DATA_DIR} (81K samples)"
echo "Pre-trained: ${PRETRAINED}"

cd ${PROJECT}/multitask

python train_mace_transformer.py "${DATA_DIR}" \
    --cache-dir "${DATA_DIR}/mace_cached_graphs" \
    --checkpoint-dir "${CKPT_DIR}" \
    --pretrained-checkpoint "${PRETRAINED}" \
    --fold ${FOLD} \
    --k-folds 5 \
    --seed 42 \
    --h-fea-len 256 \
    --dropout 0.3 \
    --log-bg \
    --n-transformer-layers 3 \
    --n-transformer-heads 4 \
    --d-ff 512 \
    --transformer-dropout 0.15 \
    --huber-delta 0.4 \
    --focal-gamma-gt 3.0 \
    --focal-gamma-eh 2.0 \
    --label-smoothing 0.06 \
    --consistency-weight 0.05 \
    --gt-minority-boost 2.0 \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --warmup-epochs 5 \
    --cosine-T0 60 \
    --cosine-Tmult 2 \
    --patience 30 \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.0005 \
    --backbone-lr 0.000005 \
    --unfreeze-epoch 15 \
    --unfreeze-epoch-2 30 \
    --unfreeze-layers 1 \
    --val-ratio 0.10 \
    --min-save-epoch 10 \
    --gt-composite-weight 0.5 \
    --eh-composite-weight 0.3 \
    --cond-anneal-epochs 20 \
    --merge-metal-indirect \
    --mtl-clamp-min -3.0 \
    --mtl-clamp-max 5.0 \
    --mace-model small \
    --workers 4 \
    --print-freq 5

EXIT_CODE=$?
echo "Fold ${FOLD} exit code: ${EXIT_CODE}  End: $(date)"
