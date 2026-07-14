#!/bin/bash
#SBATCH --job-name=predict_jacs
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_jacs_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_jacs_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
INPUT_DIR=${PROJECT}/Data/predict_jacs
OUTPUT_CSV=${PROJECT}/predictions_jacs_final.csv

source ${PROJECT}/venv/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== JACS Final Screening — MACE+ALIGNN DualBackbone ==="
echo "Node: $(hostname)  Start: $(date)"
echo "Input: ${INPUT_DIR}"
echo "Structures: $(ls ${INPUT_DIR}/*.cif 2>/dev/null | wc -l)"
echo "Output: ${OUTPUT_CSV}"

cd ${PROJECT}/multitask
python ${PROJECT}/scripts/predict_jacs_final.py \
    "${INPUT_DIR}" \
    "${OUTPUT_CSV}" \
    --device cuda

echo "Exit: $?  End: $(date)"
