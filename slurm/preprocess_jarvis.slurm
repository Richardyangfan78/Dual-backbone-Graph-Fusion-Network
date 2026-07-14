#!/bin/bash
#SBATCH --job-name=preprocess_jarvis
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/preprocess_jarvis_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/preprocess_jarvis_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== Preprocess JARVIS chalcohalide === $(date)"
python ${PROJECT}/scripts/preprocess_jarvis.py
echo "=== Done === $(date)"
