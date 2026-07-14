#!/bin/bash
#SBATCH --job-name=MACE_pretrain
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --signal=USR1@120
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_pretrain_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/mace_pretrain_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
VENV=${PROJECT}/venv
DATA_DIR=${PROJECT}/Data/mp_bandgap_v2
CKPT_DIR=${PROJECT}/checkpoints/mace_pretrain

mkdir -p ${PROJECT}/logs ${CKPT_DIR}

source ${VENV}/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1

echo "=== MACE Pre-training on MP Bandgap Data ==="
echo "Job ID: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

cd ${PROJECT}/multitask

# Step 1: Cache MACE graphs if not already done
CACHE_DIR="${DATA_DIR}/mace_cached_graphs"
CACHE_COUNT=$(ls ${CACHE_DIR}/*.pt 2>/dev/null | wc -l)
TOTAL=$(wc -l < ${DATA_DIR}/id_prop.csv)
echo "Cached graphs: ${CACHE_COUNT}/${TOTAL}"

if [ "${CACHE_COUNT}" -lt "${TOTAL}" ]; then
    echo "Caching MACE graphs..."
    python -c "
import sys, os, time, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
from data_mace_mt import MACECrystalDataset
from mace.calculators import mace_mp
calc = mace_mp(model='small', default_dtype='float32', device='cpu')
dataset = MACECrystalDataset(
    root_dir='${DATA_DIR}', z_table=calc.z_table, r_max=calc.r_max,
    cache_dir='${CACHE_DIR}',
)
t0 = time.time()
errors = 0
for i in range(len(dataset)):
    try:
        _ = dataset[i]
    except Exception as e:
        errors += 1
        if errors <= 5: print(f'Error {i}: {e}')
    if (i+1) % 5000 == 0:
        rate = (i+1)/(time.time()-t0)
        print(f'  {i+1}/{len(dataset)} ({rate:.0f}/s, ETA {(len(dataset)-i-1)/rate:.0f}s)')
print(f'Done: {len(dataset)-errors} cached, {errors} errors, {time.time()-t0:.0f}s')
"
    echo "Caching complete."
fi

# Step 2: Pre-train
echo ""
echo "=== Starting Pre-training ==="
python pretrain_mace_mp.py "${DATA_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --seed 42 \
    --h-fea-len 256 \
    --n-attn-heads 8 \
    --dropout 0.2 \
    --epochs 80 \
    --batch-size 64 \
    --lr 0.0003 \
    --backbone-lr 0.000005 \
    --unfreeze-epoch 10 \
    --unfreeze-layers 1 \
    --warmup-epochs 5 \
    --val-ratio 0.05 \
    --log-bg \
    --merge-metal-indirect \
    --use-cosine-classifier \
    --cosine-temp 0.1 \
    --mace-model small \
    --workers 8 \
    --print-freq 5

EXIT_CODE=$?
echo "Pre-training exit code: ${EXIT_CODE}  End: $(date)"
