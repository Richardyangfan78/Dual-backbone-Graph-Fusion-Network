#!/bin/bash
#SBATCH --job-name=relax_jacs
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/relax_jacs_%A_%a.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/relax_jacs_%A_%a.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network

source ${PROJECT}/venv/bin/activate
export PYTHONUNBUFFERED=1

echo "=== MACE-MP-0 Relaxation — Chunk ${SLURM_ARRAY_TASK_ID}/7 ==="
echo "Node: $(hostname)  Start: $(date)"

python ${PROJECT}/scripts/relax_jacs.py \
    --chunk ${SLURM_ARRAY_TASK_ID} \
    --total-chunks 8 \
    --fmax 0.05 \
    --max-steps 500 \
    --model medium \
    --device cuda

echo "Exit: $?  End: $(date)"
