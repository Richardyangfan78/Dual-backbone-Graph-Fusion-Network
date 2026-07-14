#!/bin/bash
#SBATCH --job-name=filter_only
#SBATCH --account=a131
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/filter_only_%j.out

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv/bin/activate
python3 ${PROJECT}/scripts/filter_clean_v2.py
