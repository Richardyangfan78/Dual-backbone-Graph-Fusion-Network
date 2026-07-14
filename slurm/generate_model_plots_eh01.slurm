#!/bin/bash
#SBATCH --job-name=gen_model_plots_eh01
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gen_model_plots_eh01_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gen_model_plots_eh01_%j.err

export MPLBACKEND=Agg

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

cd ${PROJECT}/multitask
mkdir -p ${PROJECT}/logs ${PROJECT}/results/model_plots_eh01

echo "Node: $(hostname)  Start: $(date)"
python ${PROJECT}/scripts/generate_model_plots_eh01.py
echo "Exit: $?  End: $(date)"
