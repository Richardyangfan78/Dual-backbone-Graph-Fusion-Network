#!/bin/bash
#SBATCH --job-name=predict_large
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_large_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_large_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
INPUT_DIR=${PROJECT}/Data/predict_large
OUTPUT_CSV=${PROJECT}/predictions_large.csv

source ${PROJECT}/venv/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== DualBackbone Large-Scale Inference ==="
echo "Node: $(hostname)  Start: $(date)"
echo "Input: ${INPUT_DIR}"
echo "Structures: $(ls ${INPUT_DIR}/*.cif | wc -l)"
echo "Output: ${OUTPUT_CSV}"

cd ${PROJECT}/multitask
python predict_dual_backbone.py "${INPUT_DIR}" "${OUTPUT_CSV}" --device cuda

echo "Exit: $?  End: $(date)"
