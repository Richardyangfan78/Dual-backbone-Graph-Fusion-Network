# Reproducing the DBGFN results

Run all commands from the repository root.

## 1. Clone the repository

```bash
git clone https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network.git
cd Dual-backbone-Graph-Fusion-Network
```

## 2. Recompute the reported metrics

This lightweight check recomputes the five-fold metrics directly from the archived out-of-fold predictions and verifies them against the official fold result files:

```bash
python Model/reproduce_metrics.py
```

The expected five-fold means are:

| Metric | Mean |
|---|---:|
| Band-gap MAE | 0.221780 eV |
| Gap-type accuracy | 0.889694 |
| Gap-type macro-F1 | 0.847636 |
| Stability accuracy | 0.966619 |
| Stability macro-F1 | 0.953317 |

A successful run ends with:

```text
PASS: archived OOF predictions reproduce all reported fold metrics.
```

## 3. Evaluate the released checkpoints

Create a Python 3.10 environment and install the recorded software versions:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r Model/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Download and SHA-256 verify the five official DBGFN checkpoints:

```bash
python Checkpoint/download.py
```

Evaluate all fixed test folds:

```bash
python Model/evaluate.py --device cuda
```

Use `--device cpu` when CUDA is unavailable. On its first run, the evaluator extracts the 1,768 CIF structures, downloads the public MACE-MP-0 small foundation model, and builds the MACE and ALIGNN graph caches. The reproduced fold metrics are saved to `Results/reproduced_checkpoint_metrics.csv`.

## 4. Run five-fold inference on new structures

Place CIF, VASP, POSCAR, or `.poscar` files in one directory, then run:

```bash
python Model/predict.py path/to/structures Results/predictions.csv --device cuda
```

The command reports the ensemble mean and standard deviation of the predicted band gap, the gap type, the stability class, and whether each structure passes the direct-gap/stable/0.5–1.1 eV screen.

## 5. Repeat five-fold training

Download the MACE and fold-specific ALIGNN initialization checkpoints used by DBGFN:

```bash
python Checkpoint/download.py --training-initializers
python Model/evaluate.py --prepare-only --device cuda
```

Train each aligned fold with the reported defaults:

```bash
for fold in 0 1 2 3 4; do
  python Model/train_dual_backbone.py Data/Dataset \
    --fold "$fold" \
    --mace-cache-dir Data/Dataset/mace_cached_graphs \
    --crystal-cache-dir Data/Dataset/alignn_cached_graphs \
    --mace-pretrained Checkpoint/mace_pretrained_best.pt \
    --alignn-checkpoint-dir Checkpoint/ALIGNN \
    --checkpoint-dir Results/RetrainedCheckpoints
done
```

The fixed material IDs for every train, validation, and test partition are stored in `Data/Dataset/splits_mace_alignn`. Exact bitwise weights can vary with GPU hardware and CUDA kernels; checkpoint evaluation above is the exact route for reproducing the reported numerical results.
