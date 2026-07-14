#!/bin/bash
#SBATCH --job-name=tradml_export
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=a131
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/tradml_export_%j.log

set -euo pipefail

BASE=/path/to/Dual-backbone-Graph-Fusion-Network
source "$BASE/.venv/bin/activate"
cd "$BASE"

export MPLBACKEND=Agg
python scripts/export_traditional_ml_results.py
