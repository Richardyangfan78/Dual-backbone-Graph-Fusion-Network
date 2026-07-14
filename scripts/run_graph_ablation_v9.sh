#!/bin/bash
# Graph construction ablation for v9
# Runs preprocess + train for each config; compare BG MAE / GT F1 / EH Acc
set -euo pipefail

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
DATA_DIR="${PROJECT}/Data/multitask"
LOG_DIR="${PROJECT}/logs"

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"

# Configs: name radius max_num_nbr step
CONFIGS=(
    "baseline:8:12:0.2"
    "denser:10:16:0.15"
    "sparser:6:8:0.25"
)

for cfg in "${CONFIGS[@]}"; do
    name=$(echo "$cfg" | cut -d: -f1)
    radius=$(echo "$cfg" | cut -d: -f2)
    max_nbr=$(echo "$cfg" | cut -d: -f3)
    step=$(echo "$cfg" | cut -d: -f4)

    CACHE_DIR="${DATA_DIR}/cached_graphs_${name}"
    CKPT_DIR="${PROJECT}/checkpoints/multitask_v9_graph_${name}"
    LOG="${LOG_DIR}/graph_ablation_${name}_$(date +%Y%m%d_%H%M%S).log"

    echo "=== Graph ablation: $name (r=$radius nbr=$max_nbr step=$step) ==="
    echo "Cache: $CACHE_DIR  Checkpoints: $CKPT_DIR"

    python multitask/preprocess_graphs.py "${DATA_DIR}" "${CACHE_DIR}" \
        --radius "$radius" --max-num-nbr "$max_nbr" --step "$step"

    python multitask/train_mt_v9.py "${DATA_DIR}" \
        --cache-dir "${CACHE_DIR}" \
        --checkpoint-dir "${CKPT_DIR}" \
        --epochs 600 --batch-size 64 --k-folds 3 --n-seeds 2 \
        --patience 100 --print-freq 10 \
        2>&1 | tee "$LOG"

    echo "Done: $name"
    echo ""
done

echo "Graph ablation complete. Compare checkpoints/multitask_v9_graph_*"
