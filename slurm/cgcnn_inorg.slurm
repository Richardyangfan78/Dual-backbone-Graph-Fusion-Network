#!/bin/bash
#SBATCH --job-name=CGCNN_inorg
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/cgcnn_inorg_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/cgcnn_inorg_fold%a_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
CKPT_DIR=${PROJECT}/checkpoints/cgcnn_mt_inorg
FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
echo "=== CGCNN inorg Fold ${FOLD} === $(date)"
cd ${PROJECT}/multitask
python train_mt_v10.py ${DATA}     --cache-dir ${DATA}/cached_graphs     --checkpoint-dir ${CKPT_DIR}     --fold ${FOLD} --k-folds 5 --seed 42     --atom-fea-len 160 --h-fea-len 320 --n-conv 8 --n-attn-heads 8     --dropout 0.35 --log-bg --huber-delta 0.4     --bg-reweight 0.3 --focal-gamma-gt 4.0 --focal-gamma-eh 2.0     --label-smoothing 0.08 --consistency-weight 0.05     --augment-noise 0.03 --augment-structure --augment-perturb 0.03     --augment-scale 0.02 --augment-structure-prob 0.4     --warmup-epochs 20 --cosine-T0 150 --cosine-Tmult 2 --patience 150     --epochs 800 --batch-size 48 --lr 0.0008 --val-ratio 0.15     --dynamic-weights --mtl-clamp-min -3.0 --mtl-clamp-max 5.0     --use-weighted-sampler --use-cosine-classifier --cosine-temp 0.1     --gt-minority-boost 1.5 --merge-metal-indirect --robust-norm --workers 8 --print-freq 20
echo "Done Fold ${FOLD}: $(date)"
