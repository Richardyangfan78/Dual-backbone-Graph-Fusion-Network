#!/bin/bash
#SBATCH --job-name=model_plots
#SBATCH --account=a131
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/model_plots_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/model_plots_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
mkdir -p ${PROJECT}/results/model_plots

python3 /path/to/Dual-backbone-Graph-Fusion-Network/multitask/generate_model_plots.py

echo "Done: $(date)"
