#!/bin/bash
#SBATCH --job-name=GateAnalysisV2
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gate_analysis_v2_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gate_analysis_v2_%j.err

set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
mkdir -p ${PROJECT}/logs ${PROJECT}/results/gate_analysis
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PROJECT}/scripts:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== Gate analysis v2 (official checkpoint, current 1768-compound dataset) === $(date)"
python ${PROJECT}/scripts/extract_gate_analysis.py
echo "Done: $(date)"
