# Dual-backbone Graph Fusion Network for Chalcohalide Prediction

This repository contains the released model checkpoints, code, data, and
results supporting the Dual-backbone Graph Fusion Network (DBGFN) study of
chalcohalide property prediction and candidate screening.

The workflow combines MACE and ALIGNN crystal encoders through a learned gated
fusion module. A shared multitask predictor estimates the band gap, gap type
(direct versus indirect/metal), and thermodynamic stability class.

### Repository Overview

The release supports the following tasks:

- Recompute the archived five-fold metrics from out-of-fold predictions
- Download and SHA-256 verify the five final integrated DBGFN checkpoints
- Evaluate each checkpoint on its fixed test partition
- Run five-checkpoint ensemble inference on new crystal structures
- Inspect performance figures, gate analysis, and screened candidates

### Directory Structure

```text
├── README.md
│
├── 01-checkpoints/
│   ├── download.py
│   ├── SHA256SUMS.txt
│   └── fold{0..4}_results.csv
│
├── 02-data/
│   └── dataset/
│       ├── id_prop.csv
│       ├── structures.tar.gz
│       ├── initialization_overlap_ids.csv
│       └── splits_mace_alignn/
│
├── 03-code/
│   ├── model_dual_backbone.py
│   ├── model_mace_mt.py
│   ├── model_alignn_pyg.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── reproduce_metrics.py
│   └── requirements.txt
│
└── 04-results/
    ├── 01-performance/
    ├── 02-gate-analysis/
    └── 03-screening/
```

### Environment Setup

Run all commands from the repository root.

```bash
git clone https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network.git
cd Dual-backbone-Graph-Fusion-Network
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 03-code/requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Reproduce the Archived Metrics

The lightweight route recomputes all metrics from the archived out-of-fold
predictions and verifies them against the five fold result files:

```bash
python 03-code/reproduce_metrics.py
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

### Evaluate the Final Checkpoints

Download and verify the five final DBGFN checkpoints:

```bash
python 01-checkpoints/download.py
```

Evaluate all fixed test folds:

```bash
python 03-code/evaluate.py --device cuda
```

Use `--device cpu` when CUDA is unavailable. On its first run, the evaluator
extracts the 1,768 CIF structures, obtains the public MACE-MP-0 small model
needed to instantiate the architecture, and builds the MACE and ALIGNN graph
caches. It then loads the complete state from each final checkpoint. The
recomputed metrics are written to
`04-results/01-performance/reproduced_checkpoint_metrics.csv`.

### Predict New Structures

Place CIF, VASP, POSCAR, or `.poscar` files in one directory, then run:

```bash
python 03-code/predict.py path/to/structures \
  04-results/predictions.csv --device cuda
```

The command evaluates all five checkpoints and reports the ensemble mean and
standard deviation of the predicted band gap, gap type, stability class, and
the direct-gap/stable/0.5–1.1 eV screening decision.

### Checkpoint Evaluation Scope

This is a checkpoint-evaluation and inference release. It distributes the five
final integrated model states; it does not reproduce the original optimization
trajectory from initialization. Each checkpoint contains both crystal
encoders, the gated fusion module, the shared trunk, the prediction heads, and
the band-gap normalization statistics.

An exact material-ID audit of the supervised MACE initialization artifacts
associated with the original training run identified 48 of the 1,768
downstream evaluation materials in that initialization manifest. The affected
IDs are listed by fold in
`02-data/dataset/initialization_overlap_ids.csv`. Consequently, the archived
numbers reproduce the released checkpoints but should not be interpreted as
strict unseen-material holdout estimates.

### Contact Information

For questions about the repository, please open a GitHub issue or contact the
repository maintainer through the
[project page](https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network).
