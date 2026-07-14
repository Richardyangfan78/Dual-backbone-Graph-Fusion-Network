#!/bin/bash
#SBATCH --job-name=M3GNet_inorg
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/m3gnet_inorg_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/m3gnet_inorg_fold%a_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
CKPT_DIR=${PROJECT}/checkpoints/m3gnet_mt_inorg
FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
echo "=== M3GNet inorg Fold ${FOLD} === $(date)"
cd ${PROJECT}/multitask
python train_crystal_mt.py ${DATA}     --model m3gnet --cache-dir ${DATA}/crystal_cached_graphs     --checkpoint-dir ${CKPT_DIR}     --fold ${FOLD} --k-folds 5 --seed 42     --hidden-dim 128 --n-blocks 3 --dropout 0.3     --radius 8.0 --max-num-nbr 12 --dist-bins 80 --angle-bins 40     --epochs 500 --batch-size 32 --lr 0.0008 --weight-decay 1e-4 --patience 120     --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0     --label-smoothing 0.06 --consistency-weight 0.05 --gt-minority-boost 2.0     --log-bg --merge-metal-indirect --val-ratio 0.15 --workers 4 --print-freq 10
echo "Done Fold ${FOLD}: $(date)"
