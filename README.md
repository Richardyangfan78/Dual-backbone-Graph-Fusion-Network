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
├── requirements.txt
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
│   ├── data.py
│   └── model.py
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
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Download Checkpoints

```bash
python 01-checkpoints/download.py
```

### Predict New Structures

Place CIF, VASP, POSCAR, or `.poscar` files in one directory, then run:

```bash
python 03-code/model.py path/to/structures \
  04-results/predictions.csv --device cuda
```

`data.py` is called automatically to transform each structure into the MACE and
ALIGNN graph inputs. `model.py` loads all five checkpoints, produces the
checkpoint-dependent embeddings, and reports the ensemble mean and standard
deviation of band gap, gap type, stability class, and the
direct-gap/stable/0.5–1.1 eV screening decision.


### Contact Information

For questions about the repository, please open a GitHub issue or contact the
repository maintainer through the
[project page](https://github.com/Richardyangfan78/Dual-backbone-Graph-Fusion-Network).
