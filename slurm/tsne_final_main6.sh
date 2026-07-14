#!/bin/bash
#SBATCH --job-name=tsne_m6_final
#SBATCH --partition=debug
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=a131
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/tsne_m6_final_%j.log

set -euo pipefail

BASE=/path/to/Dual-backbone-Graph-Fusion-Network
source "$BASE/.venv/bin/activate"
cd "$BASE"

export MPLBACKEND=Agg
python scripts/prepare_tsne_final_main6_features.py
python scripts/tsne_chalcohalide_final_main6.py
python scripts/tsne_ideal_final_main6.py
