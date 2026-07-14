#!/bin/bash
#SBATCH --job-name=DualBB_chalcoPT
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/dual_chalcoPT_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/dual_chalcoPT_fold%a_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
CKPT_DIR=${PROJECT}/checkpoints/dual_backbone_inorg_chalcopretrain
ALIGNN_DIR=${PROJECT}/checkpoints/alignn_mt_inorg
FOLD=${SLURM_ARRAY_TASK_ID}
MACE_PRETRAINED=${PROJECT}/checkpoints/mace_pretrain_chalco_fold${FOLD}/pretrained_best.pt
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
echo "=== MACE(chalco-pretrained)+ALIGNN inorg Fold ${FOLD} === $(date)"
echo "MACE pretrained checkpoint (fold-specific, chalco-only, no leakage): ${MACE_PRETRAINED}"
if [ ! -f "${MACE_PRETRAINED}" ]; then
  echo "ERROR: pretrained checkpoint not found: ${MACE_PRETRAINED}"
  exit 1
fi
cd ${PROJECT}/multitask
python train_dual_backbone.py ${DATA}     --mace-cache-dir ${DATA}/mace_cached_graphs     --crystal-cache-dir ${DATA}/crystal_cached_graphs     --checkpoint-dir ${CKPT_DIR}     --mace-pretrained ${MACE_PRETRAINED}     --alignn-checkpoint-dir ${ALIGNN_DIR}     --fold ${FOLD} --k-folds 5 --seed 42     --h-fea-len 256 --n-attn-heads 8 --dropout 0.3 --log-bg     --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0     --label-smoothing 0.06 --consistency-weight 0.05 --gt-minority-boost 2.0     --use-cosine-classifier --cosine-temp 0.1     --warmup-epochs 10 --cosine-T0 120 --cosine-Tmult 2 --patience 80     --epochs 300 --batch-size 32 --lr 0.0005     --mace-backbone-lr 0.000005 --alignn-backbone-lr 0.00001     --unfreeze-epoch 30 --unfreeze-epoch-2 50 --unfreeze-layers 1     --val-ratio 0.15 --min-save-epoch 30     --gt-composite-weight 0.5 --eh-composite-weight 0.3 --cond-anneal-epochs 40     --merge-metal-indirect --mtl-clamp-min -3.0 --mtl-clamp-max 5.0     --mace-model small --workers 4 --print-freq 10
echo "Done Fold ${FOLD}: $(date)"
