#!/bin/bash
#SBATCH --job-name=gate_analysis
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --constraint=gh
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gate_analysis_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gate_analysis_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
mkdir -p ${PROJECT}/logs ${PROJECT}/results/gate_analysis

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export HF_HOME=/path/to/huggingface-cache
export MACE_CACHE=/path/to/mace-cache

echo "=== Gate Analysis: MACE+ALIGNN Dual Backbone ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"

cd ${PROJECT}/multitask

python analyze_gate.py \
    --data-dir    ${PROJECT}/Data/multitask \
    --ckpt-dir    ${PROJECT}/checkpoints/dual_backbone \
    --alignn-ckpt-dir ${PROJECT}/checkpoints/alignn_mt \
    --cif-dir     ${PROJECT}/Data/multitask \
    --out-dir     ${PROJECT}/results/gate_analysis \
    --k-folds 5 \
    --seed 42 \
    --batch-size 64 \
    --workers 4

echo "Done: $(date)"
