#!/bin/bash
#SBATCH --job-name=merge_predict
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/merge_predict_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/merge_predict_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
export MPLBACKEND=Agg

echo "=== Step 1: Merge datasets === $(date)"
python ${PROJECT}/scripts/merge_datasets.py

echo ""
echo "=== Step 2: Count merged structures ==="
ls ${PROJECT}/Data/predict_merged/*.cif | wc -l

echo ""
echo "=== Step 3: Run MACE+ALIGNN prediction on merged dataset === $(date)"
cd ${PROJECT}/multitask
python ${PROJECT}/scripts/predict_jacs_nometal.py \
    "${PROJECT}/Data/predict_merged" \
    "${PROJECT}/predictions/predictions_merged_final.csv" \
    --device cuda

echo ""
echo "=== Done === $(date)"
