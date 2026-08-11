# Dual-backbone Graph Fusion Network (DBGFN)

Inference package for chalcohalide crystal screening. DBGFN combines a
pretrained MACE encoder with an ALIGNN encoder through gated feature fusion to
predict:

- band gap (eV);
- gap type (`Direct` or `Indirect`, with metallic predictions treated as
  indirect); and
- thermodynamic stability class (`Stable` or `Unstable`).

This local version is an **inference-only release**. It loads five released
DBGFN checkpoints and reports their ensemble prediction for new crystal
structures. Training, fixed-split evaluation, and archived-metric reproduction
scripts are intentionally not included in `03-code`.

## Repository layout

```text
├── README.md
├── requirements.txt
├── 01-checkpoints/
│   ├── download.py              # download and SHA-256 verify the five checkpoints
│   └── SHA256SUMS.txt
├── 02-data/dataset/
│   ├── id_prop.csv              # 1,768 reference material labels
│   ├── structures.tar.gz        # 1,768 CIF files collected from Material Project
│   ├── Prediction.zip           # 12,018 MLIP relaxed candidate CIF files
│   └── splits/                  # archived five-fold split metadata
├── 03-code/
│   ├── data.py                  # structure file → MACE + ALIGNN graph inputs
│   └── model.py                 # DBGFN architecture, checkpoint loading, prediction CLI
└── 04-results/
    └── Results.csv              # Predictions for the 12,018 candidates
```

`02-data/dataset` is reference data from the release. It is not needed to
predict a new directory of structures. In particular, `model.py` does not use
`id_prop.csv`, the archived CIF package, or the split files during inference.

## Installation

Run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

The released model expects `mace-torch==0.3.13` and the MACE-MP-0 `small`
backbone specified in `requirements.txt`.

## Download checkpoints

The checkpoint binaries are not committed to the repository. Download them
once; the script verifies every file against the included SHA-256 hashes.
The five assets in the `v1.0.0-reproducibility` GitHub Release are the
official **DBGFN manuscript-screening five-fold ensemble** used for the
reported screening results. It remains a MACE+ALIGNN dual-backbone model.

```bash
python 01-checkpoints/download.py
```

This creates the following ignored files in `01-checkpoints/`:

```text
fold0_best.pt  fold1_best.pt  fold2_best.pt  fold3_best.pt  fold4_best.pt
```

## Predict new structures

Place CIF, VASP, POSCAR, or `.poscar` files directly in one input directory.
The search is not recursive.

```bash
python 03-code/model.py path/to/structures 04-results/predictions.csv --device cuda
```

Use `--device cpu` when CUDA is unavailable. The first run may download the
public MACE-MP-0 `small` base model needed to construct the checkpoint
architecture.

Optional arguments:

```text
--checkpoint-dir PATH   Directory containing fold0_best.pt … fold4_best.pt
--device DEVICE         PyTorch device, for example cuda, cuda:0, or cpu
--mace-model NAME       MACE base model name; use small for the released checkpoints
```

The command runs all five checkpoints. It does not silently fall back to a
smaller ensemble if a checkpoint is unavailable.

### Output CSV

The destination directory is created automatically. Each successfully
processed input file produces one row with:

```text
structure_id, source_file, source_path, formula,
bg_type, bg_eV, bg_std_eV, ehull, screen_pass
```

- `bg_eV` is the five-checkpoint mean predicted band gap.
- `bg_std_eV` is the standard deviation across the five band-gap predictions.
- `ehull` is a predicted stability **class**, not a numerical energy-above-hull
  value.
- `screen_pass` is `Yes` only when the prediction is direct, stable, and has
  `0.5 ≤ bg_eV ≤ 1.1`.

If an individual structure cannot be parsed or converted, valid rows are still
written and the program exits with a non-zero status after listing skipped
files. This makes partial outputs visible to automated workflows.
