#!/usr/bin/env python3
"""Export traditional ML result CSVs and publication-style figures."""
import json
import pickle
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor


BASE = Path("/path/to/Dual-backbone-Graph-Fusion-Network")
ML = BASE / "classical_ml_baseline"
OUT = ML / "export_traditional_ml"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = 8
RANDOM_STATE = 42


def make_pipe(estimator, scale=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    steps.append(("m", estimator))
    return Pipeline(steps)


def load_formula_map():
    detail = BASE / "Data/training_set_composition_detail.csv"
    if not detail.exists():
        return {}
    df = pd.read_csv(detail)
    return {
        str(row.file).replace(".cif", ""): row.formula
        for row in df.itertuples(index=False)
        if hasattr(row, "file") and hasattr(row, "formula")
    }


def metric_rows_from_json():
    with (ML / "baseline_results.json").open() as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "traditional_ml_metrics.csv", index=False)
    return df


def export_gnn_backbone_metrics():
    with (ML / "gnn_backbones.json").open() as f:
        data = json.load(f)
    rows = []
    for model, tasks in data.items():
        for metric, values in tasks.items():
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "mean": values.get("mean"),
                    "std": values.get("std"),
                    "n_folds": values.get("n"),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "gnn_backbones_metrics.csv", index=False)
    return df


def best_classical_comparison(metrics_df, gnn_df):
    rows = []
    mace = "MACE+ALIGNN"
    specs = [
        ("bandgap", "mae_ev", "bg_mae", False, "Bandgap MAE (eV)"),
        ("gap_type", "f1_macro", "gt_f1", True, "Gap-type macro-F1"),
        ("stability", "f1_macro", "eh_f1", True, "Stability macro-F1"),
    ]
    for task, classical_key, gnn_key, higher, label in specs:
        task_df = metrics_df[(metrics_df["task"] == task) & (~metrics_df["model"].str.startswith("Dummy"))].copy()
        best_idx = task_df[classical_key].idxmax() if higher else task_df[classical_key].idxmin()
        best = task_df.loc[best_idx]
        g = gnn_df[(gnn_df["model"] == mace) & (gnn_df["metric"] == gnn_key)].iloc[0]
        rows.append(
            {
                "task": task,
                "metric": label,
                "higher_is_better": higher,
                "best_traditional_model": best["model"],
                "best_traditional_value": best[classical_key],
                "mace_alignn_value": g["mean"],
                "mace_alignn_std": g["std"],
                "absolute_delta_mace_minus_traditional": g["mean"] - best[classical_key],
                "relative_error_reduction_for_bandgap": (
                    1.0 - g["mean"] / best[classical_key] if task == "bandgap" else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "traditional_vs_mace_alignn_summary.csv", index=False)
    return out


def build_models():
    xgb_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=2,
        tree_method="hist",
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
    )
    lgb_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_samples=10,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    reg_models = [
        ("Dummy(mean)", make_pipe(DummyRegressor())),
        ("Ridge", make_pipe(Ridge(alpha=1.0), scale=True)),
        ("KNN", make_pipe(KNeighborsRegressor(n_neighbors=7, weights="distance"), scale=True)),
        ("SVM", make_pipe(SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1), scale=True)),
        ("RandomForest", make_pipe(RandomForestRegressor(n_estimators=500, n_jobs=N_JOBS, random_state=RANDOM_STATE))),
        ("ExtraTrees", make_pipe(ExtraTreesRegressor(n_estimators=500, n_jobs=N_JOBS, random_state=RANDOM_STATE))),
        (
            "HistGBM",
            make_pipe(
                HistGradientBoostingRegressor(
                    max_iter=600,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        ("XGBoost", make_pipe(XGBRegressor(objective="reg:squarederror", **xgb_params))),
        ("LightGBM", make_pipe(LGBMRegressor(objective="regression_l1", **lgb_params))),
    ]
    clf_models = [
        ("Dummy(freq)", make_pipe(DummyClassifier(strategy="most_frequent"))),
        ("Logistic", make_pipe(LogisticRegression(max_iter=3000, class_weight="balanced"), scale=True)),
        ("KNN", make_pipe(KNeighborsClassifier(n_neighbors=7, weights="distance"), scale=True)),
        (
            "SVM",
            make_pipe(SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"), scale=True),
        ),
        (
            "RandomForest",
            make_pipe(
                RandomForestClassifier(
                    n_estimators=500,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                )
            ),
        ),
        (
            "ExtraTrees",
            make_pipe(
                ExtraTreesClassifier(
                    n_estimators=500,
                    n_jobs=N_JOBS,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                )
            ),
        ),
        (
            "HistGBM",
            make_pipe(
                HistGradientBoostingClassifier(
                    max_iter=600,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                )
            ),
        ),
        (
            "XGBoost",
            make_pipe(XGBClassifier(objective="binary:logistic", eval_metric="logloss", **xgb_params)),
        ),
        ("LightGBM", make_pipe(LGBMClassifier(class_weight="balanced", **lgb_params))),
    ]
    return reg_models, clf_models


def oof_regression(X, y, y_log, folds, models):
    pred_df = pd.DataFrame({"true_bandgap_eV": y})
    metrics = []
    for name, model in models:
        print(f"OOF bandgap: {name}", flush=True)
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(folds):
            p = model
            p.fit(X[tr], y_log[tr])
            oof[te] = np.expm1(p.predict(X[te]))
        pred_df[f"pred_{name}"] = oof
        metrics.append(
            {
                "task": "bandgap",
                "model": name,
                "mae_ev": mean_absolute_error(y, oof),
                "rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
                "r2": r2_score(y, oof),
            }
        )
    return pred_df, pd.DataFrame(metrics)


def oof_classification(X, y, folds, models, task):
    pred_df = pd.DataFrame({f"true_{task}": y})
    metrics = []
    for name, model in models:
        print(f"OOF {task}: {name}", flush=True)
        oof = np.full(len(y), -1, dtype=int)
        for fold, (tr, te) in enumerate(folds):
            p = model
            fit_kwargs = {}
            if name in ("HistGBM", "XGBoost"):
                fit_kwargs["m__sample_weight"] = compute_sample_weight("balanced", y[tr])
            p.fit(X[tr], y[tr], **fit_kwargs)
            oof[te] = p.predict(X[te]).astype(int)
        pred_df[f"pred_{name}"] = oof
        metrics.append(
            {
                "task": task,
                "model": name,
                "accuracy": accuracy_score(y, oof),
                "f1_macro": f1_score(y, oof, average="macro", zero_division=0),
            }
        )
    return pred_df, pd.DataFrame(metrics)


def plot_bandgap_fits(pred_df, metrics_df):
    plot_models = ["Ridge", "KNN", "SVM", "RandomForest", "ExtraTrees", "HistGBM", "XGBoost", "LightGBM"]
    y = pred_df["true_bandgap_eV"].to_numpy()
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    max_v = max(float(np.nanmax(y)), *(float(np.nanmax(pred_df[f"pred_{m}"])) for m in plot_models))
    lim = (-0.05, max_v * 1.03)
    for ax, model in zip(axes, plot_models):
        p = pred_df[f"pred_{model}"].to_numpy()
        mrow = metrics_df[metrics_df["model"] == model].iloc[0]
        ax.scatter(y, p, s=9, alpha=0.45, color="#4C72B0", edgecolors="none", rasterized=True)
        ax.plot(lim, lim, color="#333333", lw=1.0, ls="--")
        coef = np.polyfit(y, p, 1)
        xs = np.linspace(lim[0], lim[1], 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="#C44E52", lw=1.2)
        ax.set_title(f"{model}\nMAE={mrow.mae_ev:.3f} eV, R2={mrow.r2:.3f}", fontsize=10)
        ax.grid(alpha=0.25, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
    for ax in axes[4:]:
        ax.set_xlabel("DFT bandgap (eV)")
    for ax in axes[::4]:
        ax.set_ylabel("Predicted bandgap (eV)")
    fig.suptitle("Traditional ML bandgap fitting: out-of-fold predictions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "bandgap_fitting_all_traditional_models.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "bandgap_fitting_all_traditional_models.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_confusion_grid(pred_df, task, labels, title, out_name):
    models = ["Logistic", "KNN", "SVM", "RandomForest", "ExtraTrees", "HistGBM", "XGBoost", "LightGBM"]
    y = pred_df[f"true_{task}"].to_numpy()
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.2))
    axes = axes.ravel()
    for ax, model in zip(axes, models):
        pred = pred_df[f"pred_{model}"].to_numpy()
        cm = confusion_matrix(y, pred, labels=[0, 1])
        acc = accuracy_score(y, pred)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        row_sum = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]*100:.1f}%", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], labels=labels, rotation=25, ha="right")
        ax.set_yticks([0, 1], labels=labels)
        ax.set_title(f"{model}\nAcc={acc:.3f}, F1={f1:.3f}", fontsize=9.5)
    for ax in axes[4:]:
        ax.set_xlabel("Predicted")
    for ax in axes[::4]:
        ax.set_ylabel("True")
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.015, label="Row-normalized fraction")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(OUT / f"{out_name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{out_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_comparison(metrics_df, gnn_df):
    order = [
        ("Ridge", "Logistic"),
        ("KNN", "KNN"),
        ("SVM", "SVM"),
        ("RandomForest", "RandomForest"),
        ("ExtraTrees", "ExtraTrees"),
        ("HistGBM", "HistGBM"),
        ("XGBoost", "XGBoost"),
        ("LightGBM", "LightGBM"),
    ]
    labels = ["Linear", "KNN", "SVM", "RandomForest", "ExtraTrees", "HistGBM", "XGBoost", "LightGBM"]
    specs = [
        ("bandgap", "mae_ev", "mae_std", "bg_mae", False, "Bandgap MAE (eV)"),
        ("gap_type", "f1_macro", "f1_std", "gt_f1", True, "Gap-type macro-F1"),
        ("stability", "f1_macro", "f1_std", "eh_f1", True, "Stability macro-F1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (task, key, std_key, gkey, higher, title) in zip(axes, specs):
        vals, errs = [], []
        for reg_name, clf_name in order:
            model = reg_name if task == "bandgap" else clf_name
            row = metrics_df[(metrics_df["task"] == task) & (metrics_df["model"] == model)].iloc[0]
            vals.append(row[key])
            errs.append(row.get(std_key, np.nan))
        g = gnn_df[(gnn_df["model"] == "MACE+ALIGNN") & (gnn_df["metric"] == gkey)].iloc[0]
        x = np.arange(len(vals))
        best_i = int(np.argmax(vals)) if higher else int(np.argmin(vals))
        colors = ["#4C72B0"] * len(vals)
        colors[best_i] = "#DD8452"
        ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        ax.bar([len(vals) + 0.8], [g["mean"]], yerr=[g["std"]], capsize=3, color="#C44E52", edgecolor="black")
        top = (1.08 if higher else max(vals + [g["mean"]]) * 1.25)
        ax.set_ylim(0, top)
        ax.set_xticks(list(x) + [len(vals) + 0.8], labels + ["MACE+ALIGNN"], rotation=35, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_axisbelow(True)
        for idx, value in enumerate(vals + [g["mean"]]):
            xpos = idx if idx < len(vals) else len(vals) + 0.8
            ax.text(xpos, value + top * 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Traditional ML vs MACE+ALIGNN", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "traditional_ml_vs_mace_alignn_comparison.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "traditional_ml_vs_mace_alignn_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    metrics_df = metric_rows_from_json()
    gnn_df = export_gnn_backbone_metrics()
    best_classical_comparison(metrics_df, gnn_df)

    with (ML / "features.pkl").open("rb") as f:
        d = pickle.load(f)
    X = d["X"]
    bg = d["bg"]
    gt = d["gt"]
    eh = d["eh"]
    ids = d["ids"]
    formula_map = load_formula_map()
    formulas = [formula_map.get(str(i), "") for i in ids]

    gt_m = np.where(gt == 2, 1, gt).astype(int)
    eh_c = (eh >= 0.1).astype(int)
    bg_log = np.log1p(bg)
    strat = gt_m * 2 + eh_c
    folds = list(StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(np.arange(len(bg)), strat))
    fold_id = np.empty(len(bg), dtype=int)
    for k, (_, te) in enumerate(folds):
        fold_id[te] = k

    reg_models, clf_models = build_models()
    base_cols = pd.DataFrame(
        {
            "id": ids,
            "formula": formulas,
            "fold": fold_id,
            "true_bandgap_eV": bg,
            "true_gap_type_binary": gt_m,
            "true_gap_type_label": np.where(gt_m == 0, "Direct", "Indirect_or_Metal"),
            "true_stability_binary": eh_c,
            "true_stability_label": np.where(eh_c == 0, "Stable", "Unstable"),
            "ehull_eV_atom": eh,
        }
    )

    bandgap_pred, bandgap_oof_metrics = oof_regression(X, bg, bg_log, folds, reg_models)
    bandgap_export = pd.concat([base_cols, bandgap_pred.drop(columns=["true_bandgap_eV"])], axis=1)
    bandgap_export.to_csv(OUT / "bandgap_oof_predictions.csv", index=False)
    bandgap_oof_metrics.to_csv(OUT / "bandgap_oof_metrics_recomputed.csv", index=False)
    plot_bandgap_fits(bandgap_pred, bandgap_oof_metrics)

    gap_pred, gap_oof_metrics = oof_classification(X, gt_m, folds, clf_models, "gap_type")
    gap_export = pd.concat([base_cols, gap_pred.drop(columns=["true_gap_type"])], axis=1)
    gap_export.to_csv(OUT / "gap_type_oof_predictions.csv", index=False)
    gap_oof_metrics.to_csv(OUT / "gap_type_oof_metrics_recomputed.csv", index=False)
    plot_confusion_grid(
        gap_pred,
        "gap_type",
        ["Direct", "Indirect/Metal"],
        "Traditional ML gap-type classification: out-of-fold confusion matrices",
        "gap_type_confusion_matrices_traditional_models",
    )

    stability_pred, stability_oof_metrics = oof_classification(X, eh_c, folds, clf_models, "stability")
    stability_export = pd.concat([base_cols, stability_pred.drop(columns=["true_stability"])], axis=1)
    stability_export.to_csv(OUT / "stability_oof_predictions.csv", index=False)
    stability_oof_metrics.to_csv(OUT / "stability_oof_metrics_recomputed.csv", index=False)
    plot_confusion_grid(
        stability_pred,
        "stability",
        ["Stable", "Unstable"],
        "Traditional ML thermodynamic-stability classification: out-of-fold confusion matrices",
        "stability_confusion_matrices_traditional_models",
    )

    plot_comparison(metrics_df, gnn_df)

    readme = OUT / "README_traditional_ml_export.md"
    readme.write_text(
        "# Traditional ML export\n\n"
        "Source: `classical_ml_baseline/features.pkl` and `baseline_results.json`.\n\n"
        "Main files:\n"
        "- `traditional_ml_metrics.csv`: metrics converted from `baseline_results.json`.\n"
        "- `traditional_vs_mace_alignn_summary.csv`: best traditional ML vs MACE+ALIGNN summary.\n"
        "- `gnn_backbones_metrics.csv`: GNN backbone metrics including MACE+ALIGNN.\n"
        "- `bandgap_oof_predictions.csv`: per-sample out-of-fold bandgap predictions for traditional models.\n"
        "- `gap_type_oof_predictions.csv`: per-sample out-of-fold direct-vs-other predictions.\n"
        "- `stability_oof_predictions.csv`: per-sample out-of-fold stable-vs-unstable predictions.\n"
        "- `bandgap_fitting_all_traditional_models.png/pdf`: true-vs-predicted fitting plots.\n"
        "- `gap_type_confusion_matrices_traditional_models.png/pdf`: gap-type classification plots.\n"
        "- `stability_confusion_matrices_traditional_models.png/pdf`: thermodynamic stability plots.\n"
        "- `traditional_ml_vs_mace_alignn_comparison.png/pdf`: traditional ML vs MACE+ALIGNN comparison.\n",
        encoding="utf-8",
    )
    print(f"Export complete -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
