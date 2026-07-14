#!/bin/bash
#SBATCH --job-name=MACEM3G_inorg
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --array=0-4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/macem3g_inorg_fold%a_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/macem3g_inorg_fold%a_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
CKPT_DIR=${PROJECT}/checkpoints/mace_m3gnet_inorg
M3GNET_DIR=${PROJECT}/checkpoints/m3gnet_mt_inorg
FOLD=${SLURM_ARRAY_TASK_ID}
mkdir -p ${PROJECT}/logs ${CKPT_DIR}
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
echo "=== MACE+M3GNet inorg Fold ${FOLD} === $(date)"
cd ${PROJECT}/multitask
python train_mace_m3gnet.py ${DATA}     --mace-cache-dir ${DATA}/mace_cached_graphs     --crystal-cache-dir ${DATA}/crystal_cached_graphs     --checkpoint-dir ${CKPT_DIR}     --mace-pretrained ${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt     --m3gnet-checkpoint-dir ${M3GNET_DIR}     --fold ${FOLD} --k-folds 5 --seed 42     --h-fea-len 256 --n-attn-heads 8 --dropout 0.3 --log-bg     --huber-delta 0.4 --focal-gamma-gt 3.0 --focal-gamma-eh 2.0     --label-smoothing 0.06 --consistency-weight 0.05 --gt-minority-boost 2.0     --use-cosine-classifier --cosine-temp 0.1     --warmup-epochs 10 --cosine-T0 120 --cosine-Tmult 2 --patience 80     --unfreeze-epoch 30 --unfreeze-epoch-2 50 --unfreeze-layers 1     --mace-backbone-lr 5e-6 --m3gnet-backbone-lr 1e-5     --epochs 300 --batch-size 32 --lr 5e-4 --val-ratio 0.15 --workers 4 --print-freq 10
echo "Done Fold ${FOLD}: $(date)"
