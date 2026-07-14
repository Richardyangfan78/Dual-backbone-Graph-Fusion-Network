#!/bin/bash
#SBATCH --job-name=preproc_inorg
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/preproc_inorg_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/preproc_inorg_%j.err
set -e
PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA=${PROJECT}/Data/Inorganic_datasets
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
echo "=== Preprocessing inorg graphs === $(date)"
cd ${PROJECT}/multitask
python preprocess_graphs.py ${DATA} ${DATA}/cached_graphs
echo "Done: $(date)"
