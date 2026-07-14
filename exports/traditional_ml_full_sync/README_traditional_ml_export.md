# Traditional ML export

Source: `classical_ml_baseline/features.pkl` and `baseline_results.json`.

Main files:
- `traditional_ml_metrics.csv`: metrics converted from `baseline_results.json`.
- `traditional_vs_mace_alignn_summary.csv`: best traditional ML vs MACE+ALIGNN summary.
- `gnn_backbones_metrics.csv`: GNN backbone metrics including MACE+ALIGNN.
- `bandgap_oof_predictions.csv`: per-sample out-of-fold bandgap predictions for traditional models.
- `gap_type_oof_predictions.csv`: per-sample out-of-fold direct-vs-other predictions.
- `stability_oof_predictions.csv`: per-sample out-of-fold stable-vs-unstable predictions.
- `bandgap_fitting_all_traditional_models.png/pdf`: true-vs-predicted fitting plots.
- `gap_type_confusion_matrices_traditional_models.png/pdf`: gap-type classification plots.
- `stability_confusion_matrices_traditional_models.png/pdf`: thermodynamic stability plots.
- `traditional_ml_vs_mace_alignn_comparison.png/pdf`: traditional ML vs MACE+ALIGNN comparison.
