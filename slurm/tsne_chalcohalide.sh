#!/bin/bash
#SBATCH --job-name=tsne_chalco
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=a131
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/tsne_chalco_%j.log

source /path/to/Dual-backbone-Graph-Fusion-Network/venv2/bin/activate
cd /path/to/Dual-backbone-Graph-Fusion-Network
MPLBACKEND=Agg python classical_ml_baseline/tsne_chalcohalide.py
