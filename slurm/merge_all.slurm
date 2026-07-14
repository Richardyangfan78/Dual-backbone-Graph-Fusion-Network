#!/bin/bash
#SBATCH --job-name=merge_all
#SBATCH --account=a131
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/merge_all_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/merge_all_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv/bin/activate
python3 ${PROJECT}/scripts/merge_all.py

echo 'Creating tar.gz...'
tar -czf ${PROJECT}/all_predictions.tar.gz -C ${PROJECT}/Data all_cifs/
echo 'Done. Size:'
du -sh ${PROJECT}/all_predictions.tar.gz
echo 'CIF count:'
ls ${PROJECT}/Data/all_cifs/ | wc -l
