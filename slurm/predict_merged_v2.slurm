#!/bin/bash
#SBATCH --job-name=pred_merged_v2
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/pred_merged_v2_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/pred_merged_v2_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
export MPLBACKEND=Agg

echo "=== MACE+ALIGNN merged prediction (venv2 + pbc fix) === $(date)"
echo "Structures: $(ls ${PROJECT}/Data/predict_merged/*.cif | wc -l)"
cd ${PROJECT}/multitask

python ${PROJECT}/scripts/predict_jacs_nometal.py \
    "${PROJECT}/Data/predict_merged" \
    "${PROJECT}/predictions/predictions_merged_final.csv" \
    --device cuda

echo "=== Done === $(date)"
