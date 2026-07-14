#!/bin/bash
#SBATCH --job-name=filter_clean
#SBATCH --account=a131
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/filter_clean_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/filter_clean_%j.err

source /path/to/Dual-backbone-Graph-Fusion-Network/venv/bin/activate
python /path/to/Dual-backbone-Graph-Fusion-Network/scripts/filter_clean.py
