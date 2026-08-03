# Dual-backbone Graph Fusion Network for Chalcohalide Prediction

This repository contains the released model checkpoints, code, data, and
results supporting the Dual-backbone Graph Fusion Network (DBGFN) study of
chalcohalide property prediction and candidate screening.

The workflow combines MACE and ALIGNN crystal encoders through a learned gated
fusion module. A shared multitask predictor estimates the band gap, gap type
(direct versus indirect/metal), and thermodynamic stability class.


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

### Predict New Structures

Place CIF, VASP, POSCAR, or `.poscar` files in one directory, then run:

```bash
python 03-code/predict.py path/to/structures \
  04-results/predictions.csv --device cuda
```

The command evaluates all five checkpoints and reports the ensemble mean and
standard deviation of the predicted band gap, gap type, stability class, and
the direct-gap/stable/0.5–1.1 eV screening decision.


### Contact Information

For questions about the repository, please open a GitHub issue or contact the
repository maintainer through the
[project page](https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network).
