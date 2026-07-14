#!/bin/bash
#SBATCH --job-name=MACE_pretrain_chalco
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_pretrain_chalco_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_pretrain_chalco_fold%a_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
FOLD=${SLURM_ARRAY_TASK_ID}
DATA_DIR=${PROJECT}/Data/Inorganic_datasets/chalco_pretrain_fold${FOLD}
CKPT_DIR=${PROJECT}/checkpoints/mace_pretrain_chalco_fold${FOLD}
SHARED_CACHE=${PROJECT}/Data/Inorganic_datasets/mace_cached_graphs

mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== MACE pretrain on CHALCOHALIDE-ONLY data (fold ${FOLD} train split, no leakage) === $(date)"
echo "Data dir (labels): ${DATA_DIR}  ($(wc -l < ${DATA_DIR}/id_prop.csv) samples)"
echo "Cache dir (reused, shared, pre-computed): ${SHARED_CACHE}"

cd ${PROJECT}/multitask
python pretrain_mace_mp.py "${DATA_DIR}" \
    --cache-dir "${SHARED_CACHE}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --seed 42 \
    --h-fea-len 256 \
    --n-attn-heads 8 \
    --dropout 0.2 \
    --epochs 80 \
    --batch-size 64 \
    --lr 0.0003 \
    --backbone-lr 0.000005 \
    --unfreeze-epoch 10 \
    --unfreeze-layers 1 \
    --warmup-epochs 5 \
    --val-ratio 0.05 \
    --log-bg \
    --merge-metal-indirect \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --mace-model small \
    --workers 8 \
    --print-freq 5

EXIT_CODE=$?
echo "Fold ${FOLD} pretrain exit code: ${EXIT_CODE}  End: $(date)"
