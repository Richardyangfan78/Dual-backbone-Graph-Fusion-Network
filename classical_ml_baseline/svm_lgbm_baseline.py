import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score
from sklearn.svm import SVR, SVC
from lightgbm import LGBMRegressor, LGBMClassifier

BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
d = pickle.load(open(BASE + '/features.pkl', 'rb'))
X, bg, gt, eh = d['X'], d['bg'], d['gt'], d['eh']
n = len(bg)
gt_m = np.where(gt == 2, 1, gt).astype(int)
eh_c = (eh >= 0.1).astype(int)
bg_log = np.log1p(bg)
strat = gt_m * 2 + eh_c
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(n), strat))

res = json.load(open(BASE + '/baseline_results.json'))
res = [r for r in res if r['model'] not in ('LightGBM', 'SVM')]   # idempotent

LGB = dict(n_estimators=600, learning_rate=0.05, num_leaves=31, subsample=0.8,
           colsample_bytree=0.8, reg_lambda=1.0, min_child_samples=10,
           n_jobs=8, random_state=42, verbose=-1)

def reg(name, make_pipe):
    oof = np.full(n, np.nan); per = []
    for tr, te in folds:
        p = make_pipe(); p.fit(X[tr], bg_log[tr])
        pred = np.expm1(p.predict(X[te])); oof[te] = pred
        per.append(mean_absolute_error(bg[te], pred))
    mae = mean_absolute_error(bg, oof); rmse = float(np.sqrt(np.mean((bg - oof) ** 2))); r2 = r2_score(bg, oof)
    print('REG  %-10s eV-MAE %.4f (perfold %.4f+-%.4f)  RMSE %.4f  R2 %.4f' %
          (name, mae, np.mean(per), np.std(per), rmse, r2), flush=True)
    res.append(dict(task='bandgap', model=name, mae_ev=mae, mae_perfold=float(np.mean(per)),
                    mae_std=float(np.std(per)), rmse=rmse, r2=float(r2)))

def clf(name, make_pipe, y, tag, sw=False):
    oof = np.full(n, -1); pf1 = []; pac = []
    for tr, te in folds:
        p = make_pipe()
        if sw:
            p.fit(X[tr], y[tr], m__sample_weight=compute_sample_weight('balanced', y[tr]))
        else:
            p.fit(X[tr], y[tr])
        pred = p.predict(X[te]); oof[te] = pred
        pf1.append(f1_score(y[te], pred, average='macro', zero_division=0))
        pac.append(accuracy_score(y[te], pred))
    f1 = f1_score(y, oof, average='macro', zero_division=0); acc = accuracy_score(y, oof)
    print('CLF[%s] %-10s acc %.4f  macroF1 %.4f (perfold %.4f+-%.4f)' %
          (tag, name, acc, f1, np.mean(pf1), np.std(pf1)), flush=True)
    res.append(dict(task=tag, model=name, acc=float(acc), f1_macro=float(f1),
                    f1_perfold=float(np.mean(pf1)), f1_std=float(np.std(pf1)),
                    acc_perfold=float(np.mean(pac))))

# ---------- regression ----------
reg('SVM', lambda: Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc', StandardScaler()),
                             ('m', SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1))]))
reg('LightGBM', lambda: Pipeline([('imp', SimpleImputer(strategy='median')),
                                  ('m', LGBMRegressor(objective='regression_l1', **LGB))]))

# ---------- classification ----------
for y, tag in [(gt_m, 'gap_type'), (eh_c, 'stability')]:
    clf('SVM', lambda: Pipeline([('imp', SimpleImputer(strategy='median')),
                                 ('sc', StandardScaler()),
                                 ('m', SVC(kernel='rbf', C=10.0, gamma='scale',
                                           class_weight='balanced'))]), y, tag)
    clf('LightGBM', lambda: Pipeline([('imp', SimpleImputer(strategy='median')),
                                      ('m', LGBMClassifier(class_weight='balanced', **LGB))]),
        y, tag)

json.dump(res, open(BASE + '/baseline_results.json', 'w'), indent=2)
print('\nAppended SVM + LightGBM rows -> baseline_results.json', flush=True)
