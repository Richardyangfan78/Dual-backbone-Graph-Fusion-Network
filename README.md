# DBGFN checkpoint evaluation and inference

This repository provides the five final DBGFN cross-validation checkpoints,
the fixed evaluation partitions, and the code required to reproduce their
archived numerical outputs or run ensemble inference on new structures.

DBGFN combines MACE and ALIGNN crystal-encoder backbones with a learned gated
fusion module and three prediction heads for band gap, gap type, and stability.
Each released checkpoint contains the complete integrated model state:
both crystal encoders, the fusion module, the shared trunk, the prediction
heads, and the band-gap normalization statistics.

## Scope of this release

This is a **checkpoint-evaluation and inference package**. It distributes only
the five final integrated checkpoints and does not claim to reproduce the
original training trajectory from initialization. The archived out-of-fold
predictions and fold result files are included so that the reported checkpoint
outputs can be checked independently of the GPU environment.

An exact material-ID audit of the supervised MACE initialization artifacts
associated with the original training run found 48 of the 1,768 downstream
evaluation materials in that initialization manifest. The archived metrics
below therefore reproduce the released checkpoints, but should not be
interpreted as strict unseen-material holdout estimates. The affected IDs are
listed by evaluation fold in `Data/initialization_overlap_ids.csv`.

## 1. Clone the repository

```bash
git clone https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network.git
cd Dual-backbone-Graph-Fusion-Network
```

## 2. Recompute the archived metrics

This lightweight check recomputes the five-fold metrics directly from the
archived out-of-fold predictions and verifies them against the official fold
result files:

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

## 3. Evaluate the final checkpoints

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

Download and SHA-256 verify the five final DBGFN checkpoints:

```bash
python Checkpoint/download.py
```

Evaluate all fixed test folds:

```bash
python Model/evaluate.py --device cuda
```

Use `--device cpu` when CUDA is unavailable. On its first run, the evaluator
extracts the 1,768 CIF structures, downloads the public MACE-MP-0 small
foundation model needed to instantiate the architecture, and builds the MACE
and ALIGNN graph caches. It then loads the complete state from each released
checkpoint. Reproduced fold metrics are saved to
`Results/reproduced_checkpoint_metrics.csv`.

The material IDs for every fixed train, validation, and test partition are in
`Data/Dataset/splits_mace_alignn`. SHA-256 digests for all released weights are
recorded in `Checkpoint/SHA256SUMS.txt`.

## 4. Run ensemble inference on new structures

Place CIF, VASP, POSCAR, or `.poscar` files in one directory, then run:

```bash
python Model/predict.py path/to/structures Results/predictions.csv --device cuda
```

The command evaluates all five final checkpoints and reports the ensemble mean
and standard deviation of the predicted band gap, the gap type, the stability
class, and whether each structure passes the direct-gap/stable/0.5–1.1 eV
screen.
