#!/bin/bash
#SBATCH --job-name=predict_dual
#SBATCH --account=a131
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_dual_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_dual_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
INPUT_DIR="${PROJECT}/Data/chalcohalide POSCAR"
OUTPUT_CSV=${PROJECT}/predictions_dual_backbone.csv

source ${PROJECT}/venv/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== DualBackbone Inference ==="
echo "Node: $(hostname)  Start: $(date)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_CSV}"

cd ${PROJECT}/multitask
python predict_dual_backbone.py "${INPUT_DIR}" "${OUTPUT_CSV}" --device cuda

echo "Exit: $?  End: $(date)"
