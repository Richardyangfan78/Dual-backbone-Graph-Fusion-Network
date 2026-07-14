#!/bin/bash
#SBATCH --job-name=MACE_inorg
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=3-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_inorg_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_inorg_fold%a_%j.err
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
CKPT_DIR=${PROJECT}/checkpoints/mace_mt_inorg
FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
echo "=== MACE inorg Fold ${FOLD} === $(date)"
cd ${PROJECT}/multitask
python train_mace_mt.py ${DATA}     --cache-dir ${DATA}/mace_cached_graphs     --checkpoint-dir ${CKPT_DIR}     --fold ${FOLD} --k-folds 5 --seed 42     --h-fea-len 256 --n-attn-heads 8 --dropout 0.35 --log-bg     --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0     --label-smoothing 0.06 --consistency-weight 0.05 --gt-minority-boost 2.0     --use-cosine-classifier --cosine-temp 0.1     --warmup-epochs 15 --cosine-T0 150 --cosine-Tmult 2 --patience 100     --epochs 400 --batch-size 32 --lr 0.0005 --backbone-lr 0.00003     --unfreeze-epoch 40 --unfreeze-layers 0 --val-ratio 0.1     --merge-metal-indirect --mtl-clamp-min -3.0 --mtl-clamp-max 5.0     --mace-model small --workers 4 --print-freq 10
echo "Done Fold ${FOLD}: $(date)"
