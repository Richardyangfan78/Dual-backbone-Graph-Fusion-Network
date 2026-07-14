#!/bin/bash
#SBATCH --job-name=DualBB_inorg
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=logs/dual_inorg_fold%a_%j.out
#SBATCH --error=logs/dual_inorg_fold%a_%j.err
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA="${DATA:-${PROJECT}/Data/Inorganic_datasets}"
CKPT_DIR="${CKPT_DIR:-${PROJECT}/checkpoints/dual_backbone_inorg}"
ALIGNN_DIR="${ALIGNN_DIR:-${PROJECT}/checkpoints/alignn_mt_inorg}"
MACE_PRETRAINED="${MACE_PRETRAINED:-${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt}"
VENV="${VENV:-${PROJECT}/.venv}"
FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
else
    echo "No virtual environment found at ${VENV}; using current Python environment."
fi
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
echo "=== MACE+ALIGNN inorg Fold ${FOLD} === $(date)"
cd ${PROJECT}/multitask
python train_dual_backbone.py ${DATA}     --mace-cache-dir ${DATA}/mace_cached_graphs     --crystal-cache-dir ${DATA}/crystal_cached_graphs     --checkpoint-dir ${CKPT_DIR}     --mace-pretrained ${MACE_PRETRAINED}     --alignn-checkpoint-dir ${ALIGNN_DIR}     --fold ${FOLD} --k-folds 5 --seed 42     --h-fea-len 256 --n-attn-heads 8 --dropout 0.3 --log-bg     --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0     --label-smoothing 0.06 --consistency-weight 0.05 --gt-minority-boost 2.0     --use-cosine-classifier --cosine-temp 0.1     --warmup-epochs 10 --cosine-T0 120 --cosine-Tmult 2 --patience 80     --epochs 300 --batch-size 32 --lr 0.0005     --mace-backbone-lr 0.000005 --alignn-backbone-lr 0.00001     --unfreeze-epoch 30 --unfreeze-epoch-2 50 --unfreeze-layers 1     --val-ratio 0.15 --min-save-epoch 30     --gt-composite-weight 0.5 --eh-composite-weight 0.3 --cond-anneal-epochs 40     --merge-metal-indirect --mtl-clamp-min -3.0 --mtl-clamp-max 5.0     --mace-model small --workers 4 --print-freq 10
echo "Done Fold ${FOLD}: $(date)"
