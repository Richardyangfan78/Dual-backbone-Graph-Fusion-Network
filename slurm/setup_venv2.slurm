#!/bin/bash
#SBATCH --job-name=setup_venv2
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/setup_venv2_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/setup_venv2_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV2=$PROJECT/venv2

echo "=== Creating venv2 === $(date)"
python3.11 -m venv $VENV2
$VENV2/bin/pip install --upgrade pip setuptools wheel

echo "=== Installing PyTorch 2.5.1+cu124 === $(date)"
$VENV2/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing torch_geometric 2.7.0 === $(date)"
$VENV2/bin/pip install torch_geometric==2.7.0
$VENV2/bin/pip install torch_scatter torch_sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html || true

echo "=== Installing mace-torch === $(date)"
$VENV2/bin/pip install mace-torch==0.3.13

echo "=== Installing other deps === $(date)"
$VENV2/bin/pip install pymatgen==2025.10.7 ase==3.28.0 numpy e3nn==0.4.4 scipy matplotlib tqdm

echo "=== Verifying === $(date)"
$VENV2/bin/python3 -c "
import torch, mace, pymatgen, torch_geometric, ase
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('torch_geometric:', torch_geometric.__version__)
print('mace:', mace.__version__)
print('ALL OK')
"
echo "=== Done === $(date)"
