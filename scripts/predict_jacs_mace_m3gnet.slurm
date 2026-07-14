#!/bin/bash
#SBATCH --job-name=predict_jacs_m3g
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_jacs_m3g_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_jacs_m3g_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
INPUT_DIR=${PROJECT}/Data/predict_jacs_relaxed
OUTPUT_CSV=${PROJECT}/predictions_jacs_relaxed_mace_m3gnet.csv

source ${PROJECT}/venv/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== JACS Screening (post-MLIP relax) — MACE+M3GNet eh01 ==="
echo "Node: $(hostname)  Start: $(date)"
echo "Structures: $(ls ${INPUT_DIR}/*.cif 2>/dev/null | wc -l)"

cd ${PROJECT}/multitask
python ${PROJECT}/scripts/predict_jacs_mace_m3gnet.py     "${INPUT_DIR}"     "${OUTPUT_CSV}"     --device cuda

echo "Exit: $?  End: $(date)"
