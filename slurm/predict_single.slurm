#!/bin/bash
#SBATCH --job-name=predict_single
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_single_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/predict_single_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv/bin/activate
pip install typing_extensions six --quiet --no-deps 2>/dev/null || true
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache
export MPLBACKEND=Agg

echo "=== Single structure prediction: AgBiSeCl2 ==="
echo "Node: $(hostname)  Start: $(date)"

cd ${PROJECT}/multitask
python ${PROJECT}/scripts/predict_jacs_eh01.py     ${PROJECT}/Data/predict_single     ${PROJECT}/predictions_single_AgBiSeCl2.csv     --device cuda

echo "Exit: $?  End: $(date)"
