# Data Manifest

## Included

| Path | Purpose |
|---|---|
| `Data/Inorganic_datasets/id_prop.csv` | Processed multitask labels: material id, band gap, gap type, and energy above hull. |
| `Data/Inorganic_datasets/atom_init.json` | Atom feature initialization file used by CGCNN-style baselines. |
| `Data/Inorganic_datasets/splits_mace_alignn/` | Aligned five-fold train/validation/test split files for MACE+ALIGNN and MACE+M3GNet comparisons. |
| `Data/cifs_chalcohalide/` | CIF structures for the chalcohalide training set. |
| `Data/predict_final_main6_manifest.csv` | Manifest for the 12,018-structure screening library used in the main text. |
| `Data/predict_final_main6_novel_manifest.csv` | Novel subset after excluding training-set overlaps. |
| `Data/predict_final_main6_training_overlap.csv` | Removed training-set overlaps. |
| `export_splits/` | CSV exports of the train/validation/test splits for reviewer inspection. |
| `exports/final_candidates_screen_direct_stable_bg_0p5_1p1/final_candidates_453_manifest.csv` | Final 453-candidate manifest after direct-gap, band-gap-window, and stability screening. |
| `exports/final_candidates_screen_direct_stable_bg_0p5_1p1_cifs.tar.gz` | CIF archive for the 453 final candidates. |
| `exports/traditional_ml_full_sync/` | Traditional-ML feature matrix, out-of-fold predictions, and metrics. |
| `results/best_bg_six_model_figures/` | OOF predictions and plotting outputs for the six-model comparison. |
| `results/gate_analysis/` | Gate-analysis exports for interpreting dual-backbone fusion. |

## Excluded Generated Artifacts

The following artifacts are not committed because they are generated, large, or hardware-specific:

- `checkpoints/`: fold checkpoint files (`*.pt`, `*.pth`, `*.ckpt`, `*.pth.tar`).
- `Data/**/mace_cached_graphs/`: cached MACE graph tensors.
- `Data/**/crystal_cached_graphs/`: cached crystal/line-graph tensors.
- `logs/`: Slurm and training logs.
- Python virtual environments.

These files can be regenerated from the included CIFs, labels, split files, and scripts.

## Path Sanitization

Some exported CSV files originally contained absolute CSCS paths in provenance columns. Those paths have been replaced with `PROJECT_ROOT/` placeholders so that the manifests remain readable without depending on a private cluster directory.
