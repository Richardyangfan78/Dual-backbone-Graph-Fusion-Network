# Classical ML + matminer  vs  Gate-Fusion GNN  (Chalcohalide inorganic dataset)

Dataset: `Data/Inorganic_datasets`, N=1768 structures. Identical 5-fold StratifiedKFold(seed=42, strat=gap_type*2+stability). Features: 154 matminer descriptors (Magpie ElementProperty + Stoichiometry + ValenceOrbital + IonProperty + TMetalFraction + BandCenter + DensityFeatures + GlobalSymmetry + StructuralComplexity).

## 1) Bandgap regression  (MAE in eV, lower better)

| Model | eV-MAE | RMSE | R2 |
|---|---|---|---|
| Dummy(mean) | 1.034 | 1.348 | -0.061 |
| Ridge | 0.746 | 0.969 | 0.451 |
| KNN | 0.637 | 0.931 | 0.494 |
| RandomForest | 0.527 | 0.763 | 0.660 |
| ExtraTrees | 0.470 | 0.740 | 0.680 |  **<-- best classical**
| HistGBM | 0.496 | 0.764 | 0.659 |
| XGBoost | 0.483 | 0.746 | 0.675 |
| SVM | 0.551 | 0.827 | 0.601 |
| LightGBM | 0.477 | 0.727 | 0.692 |
| **Gate-Fusion GNN** | **0.222** | - | - |  **(formal model)**

## 2) Gap-type classification  (direct vs indirect/metal; macro-F1, higher better)

| Model | Accuracy | macro-F1 |
|---|---|---|
| Dummy(freq) | 0.775 | 0.437 |
| Logistic | 0.648 | 0.589 |
| KNN | 0.783 | 0.617 |
| RandomForest | 0.807 | 0.651 |
| ExtraTrees | 0.791 | 0.652 |
| HistGBM | 0.790 | 0.675 |
| XGBoost | 0.786 | 0.673 |
| SVM | 0.728 | 0.640 |
| LightGBM | 0.795 | 0.684 |  **<-- best classical**
| **Gate-Fusion GNN** | **0.890** | **0.848** |  **(formal model)**

## 3) Stability classification  (E_hull<0.1 stable vs unstable; macro-F1, higher better)

| Model | Accuracy | macro-F1 |
|---|---|---|
| Dummy(freq) | 0.765 | 0.433 |
| Logistic | 0.794 | 0.744 |
| KNN | 0.852 | 0.766 |
| RandomForest | 0.887 | 0.826 |
| ExtraTrees | 0.878 | 0.818 |
| HistGBM | 0.887 | 0.835 |
| XGBoost | 0.893 | 0.844 |  **<-- best classical**
| SVM | 0.847 | 0.794 |
| LightGBM | 0.885 | 0.832 |
| **Gate-Fusion GNN** | **0.967** | **0.953** |  **(formal model)**

## Summary (best classical vs GNN)

| Task | Metric | Best classical | Gate-Fusion GNN | GNN advantage |
|---|---|---|---|---|
| Bandgap | eV-MAE | 0.470 (ExtraTrees) | 0.222 | 53% lower error |
| Gap-type | macro-F1 | 0.684 (LightGBM) | 0.848 | +0.163 |
| Stability | macro-F1 | 0.844 (XGBoost) | 0.953 | +0.109 |