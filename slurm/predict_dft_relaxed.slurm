#!/bin/bash
#SBATCH --job-name=predict_dft
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/slurm/predict_dft_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/slurm/predict_dft_%j.err

cd /path/to/Dual-backbone-Graph-Fusion-Network
source venv2/bin/activate

python scripts/predict_jacs_eh01.py     Data/predict_dft_relaxed/     predictions/predictions_jacs_relaxed_v3.csv     --device cuda

echo 'Done!'
