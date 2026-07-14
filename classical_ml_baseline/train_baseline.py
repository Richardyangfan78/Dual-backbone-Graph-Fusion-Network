import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score

BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
d = pickle.load(open(BASE + '/features.pkl', 'rb'))
X, bg, gt, eh = d['X'], d['bg'], d['gt'], d['eh']
n = len(bg)

# --- targets matched to the formal Gate-Fusion model ---
gt_m = np.where(gt == 2, 1, gt).astype(int)   # merge metal(2) -> indirect(1): direct=0 vs other=1
eh_c = (eh >= 0.1).astype(int)                 # stability: stable(<0.1)=0 vs unstable=1
bg_log = np.log1p(bg)                           # regression target (model uses log1p, MAE in eV)
strat = gt_m * 2 + eh_c                         # same stratification label as data_dual_backbone.py

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.arange(n), strat))
print('N=%d  feats=%d  fold test sizes=%s' % (n, X.shape[1], [len(te) for _, te in folds]), flush=True)
print('gap_type: direct=%d other=%d | stability: stable=%d unstable=%d' %
      ((gt_m == 0).sum(), (gt_m == 1).sum(), (eh_c == 0).sum(), (eh_c == 1).sum()), flush=True)

results = []

def reg(name, est, scale=False):
    steps = [('imp', SimpleImputer(strategy='median'))]
    if scale: steps.append(('sc', StandardScaler()))
    steps.append(('m', est))
    oof = np.full(n, np.nan); per = []
    for tr, te in folds:
        p = Pipeline(steps); p.fit(X[tr], bg_log[tr])
        pred = np.expm1(p.predict(X[te])); oof[te] = pred
        per.append(mean_absolute_error(bg[te], pred))
    mae = mean_absolute_error(bg, oof); rmse = float(np.sqrt(np.mean((bg - oof) ** 2)))
    r2 = r2_score(bg, oof)
    print('REG  %-14s  eV-MAE %.4f (perfold %.4f+-%.4f)  RMSE %.4f  R2 %.4f' %
          (name, mae, np.mean(per), np.std(per), rmse, r2), flush=True)
    results.append(dict(task='bandgap', model=name, mae_ev=mae, mae_perfold=float(np.mean(per)),
                        mae_std=float(np.std(per)), rmse=rmse, r2=float(r2)))

def clf(name, est, y, tag, scale=False, bal_sw=False):
    steps = [('imp', SimpleImputer(strategy='median'))]
    if scale: steps.append(('sc', StandardScaler()))
    steps.append(('m', est))
    oof = np.full(n, -1); pf1 = []; pac = []
    for tr, te in folds:
        p = Pipeline(steps)
        if bal_sw:
            p.fit(X[tr], y[tr], m__sample_weight=compute_sample_weight('balanced', y[tr]))
        else:
            p.fit(X[tr], y[tr])
        pred = p.predict(X[te]); oof[te] = pred
        pf1.append(f1_score(y[te], pred, average='macro', zero_division=0))
        pac.append(accuracy_score(y[te], pred))
    f1 = f1_score(y, oof, average='macro', zero_division=0); acc = accuracy_score(y, oof)
    print('CLF[%s] %-14s  acc %.4f  macroF1 %.4f (perfold %.4f+-%.4f)' %
          (tag, name, acc, f1, np.mean(pf1), np.std(pf1)), flush=True)
    results.append(dict(task=tag, model=name, acc=float(acc), f1_macro=float(f1),
                        f1_perfold=float(np.mean(pf1)), f1_std=float(np.std(pf1)),
                        acc_perfold=float(np.mean(pac))))

print('\n========== BANDGAP REGRESSION (eV MAE, lower=better) ==========', flush=True)
reg('Dummy(mean)', DummyRegressor())
reg('Ridge', Ridge(alpha=1.0), scale=True)
reg('KNN', KNeighborsRegressor(n_neighbors=7, weights='distance'), scale=True)
reg('RandomForest', RandomForestRegressor(n_estimators=500, n_jobs=8, random_state=42))
reg('ExtraTrees', ExtraTreesRegressor(n_estimators=500, n_jobs=8, random_state=42))
reg('HistGBM', HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=42))

for y, tag in [(gt_m, 'gap_type'), (eh_c, 'stability')]:
    print('\n========== %s CLASSIFICATION (macro-F1, higher=better) ==========' % tag.upper(), flush=True)
    clf('Dummy(freq)', DummyClassifier(strategy='most_frequent'), y, tag)
    clf('Logistic', LogisticRegression(max_iter=3000, class_weight='balanced'), y, tag, scale=True)
    clf('KNN', KNeighborsClassifier(n_neighbors=7, weights='distance'), y, tag, scale=True)
    clf('RandomForest', RandomForestClassifier(n_estimators=500, n_jobs=8, random_state=42,
                                               class_weight='balanced'), y, tag)
    clf('ExtraTrees', ExtraTreesClassifier(n_estimators=500, n_jobs=8, random_state=42,
                                           class_weight='balanced'), y, tag)
    clf('HistGBM', HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05,
                                                  l2_regularization=1.0, random_state=42),
        y, tag, bal_sw=True)

json.dump(results, open(BASE + '/baseline_results.json', 'w'), indent=2)
print('\nSAVED baseline_results.json (%d rows)' % len(results), flush=True)
