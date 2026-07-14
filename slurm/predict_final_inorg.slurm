#!/bin/bash
#SBATCH --job-name=pred_final
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/pred_final_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/pred_final_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== MACE+ALIGNN (dual_backbone_inorg) on predict_final === $(date)"
echo "Structures: $(ls ${PROJECT}/Data/predict_final/*.cif | wc -l)"
cd ${PROJECT}/multitask

python ${PROJECT}/scripts/predict_inorg_dual_backbone.py \
    ${PROJECT}/Data/predict_final \
    ${PROJECT}/predictions/predictions_final_inorg.csv \
    --device cuda

echo "=== Done === $(date)"
