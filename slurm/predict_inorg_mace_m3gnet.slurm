#!/bin/bash
#SBATCH --job-name=pred_inorg_m3g
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/pred_inorg_m3g_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/pred_inorg_m3g_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== MACE+M3GNet (inorg) prediction: 576 structures === $(date)"
cd ${PROJECT}/multitask
python ${PROJECT}/scripts/predict_inorg_mace_m3gnet.py \
    ${PROJECT}/Data/predict_dft_relaxed \
    ${PROJECT}/predictions/predictions_inorg_mace_m3gnet.csv \
    --device cuda
echo "Done: $(date)"
