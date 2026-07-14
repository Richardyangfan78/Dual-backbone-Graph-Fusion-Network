#!/bin/bash
#SBATCH --job-name=MACE_pt_aligned
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_pt_aligned_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_pt_aligned_fold%a_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
CKPT_DIR=${PROJECT}/checkpoints/mace_mt_inorg_pretrained_aligned
FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
echo "=== MACE-alone (MP-bandgap pretrained, aligned split) Fold ${FOLD} === $(date)"
echo "Purpose: isolate ALIGNN's marginal contribution vs dual_backbone_inorg (Group A)"
echo "Same MACE pretrain source + same MACE-branch schedule as Group A, minus ALIGNN+fusion"
cd ${PROJECT}/multitask
python train_mace_mt.py ${DATA}     --cache-dir ${DATA}/mace_cached_graphs     --checkpoint-dir ${CKPT_DIR}     --pretrained-checkpoint ${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt     --fold ${FOLD} --k-folds 5 --seed 42     --h-fea-len 256 --n-attn-heads 8 --dropout 0.3 --log-bg     --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0     --label-smoothing 0.06 --consistency-weight 0.05 --gt-minority-boost 2.0     --use-cosine-classifier --cosine-temp 0.1     --warmup-epochs 10 --cosine-T0 120 --cosine-Tmult 2 --patience 80     --epochs 300 --batch-size 32 --lr 0.0005 --backbone-lr 0.000005     --unfreeze-epoch 30 --unfreeze-epoch-2 50 --unfreeze-layers 1     --val-ratio 0.15 --min-save-epoch 30     --gt-composite-weight 0.5 --eh-composite-weight 0.3 --cond-anneal-epochs 40     --merge-metal-indirect --mtl-clamp-min -3.0 --mtl-clamp-max 5.0     --mace-model small --workers 4 --print-freq 10
echo "Done Fold ${FOLD}: $(date)"
