#!/bin/bash
#SBATCH --job-name=OOF_probs
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/oof_probs_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/oof_probs_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== MACE+ALIGNN OOF probs ==="
python ${PROJECT}/multitask/extract_oof_probs.py ${DATA} \
    --ckpt-dir   ${PROJECT}/checkpoints/dual_backbone_inorg \
    --alignn-dir ${PROJECT}/checkpoints/alignn_mt_inorg \
    --mace-pretrained ${PROJECT}/checkpoints/mace_pretrain/pretrained_best.pt \
    --out-csv    ${PROJECT}/exports/oof_probs_mace_alignn.csv

echo "Done"
