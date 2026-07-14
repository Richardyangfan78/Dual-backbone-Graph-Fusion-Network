import warnings; warnings.filterwarnings('ignore')
import pickle, json, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score
from xgboost import XGBRegressor, XGBClassifier

BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
d = pickle.load(open(BASE + '/features.pkl', 'rb'))
X, bg, gt, eh = d['X'], d['bg'], d['gt'], d['eh']
n = len(bg)
gt_m = np.where(gt == 2, 1, gt).astype(int)
eh_c = (eh >= 0.1).astype(int)
bg_log = np.log1p(bg)
strat = gt_m * 2 + eh_c
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.arange(n), strat))

XP = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
          colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=2,
          tree_method='hist', n_jobs=8, random_state=42)

# load existing results to append
res = json.load(open(BASE + '/baseline_results.json'))
res = [r for r in res if r['model'] != 'XGBoost']   # idempotent

# ---- bandgap regression ----
oof = np.full(n, np.nan); per = []
for tr, te in folds:
    imp = SimpleImputer(strategy='median').fit(X[tr])
    m = XGBRegressor(objective='reg:squarederror', **XP)
    m.fit(imp.transform(X[tr]), bg_log[tr])
    pred = np.expm1(m.predict(imp.transform(X[te]))); oof[te] = pred
    per.append(mean_absolute_error(bg[te], pred))
mae = mean_absolute_error(bg, oof); rmse = float(np.sqrt(np.mean((bg - oof) ** 2))); r2 = r2_score(bg, oof)
print('REG  XGBoost        eV-MAE %.4f (perfold %.4f+-%.4f)  RMSE %.4f  R2 %.4f' %
      (mae, np.mean(per), np.std(per), rmse, r2), flush=True)
res.append(dict(task='bandgap', model='XGBoost', mae_ev=mae, mae_perfold=float(np.mean(per)),
                mae_std=float(np.std(per)), rmse=rmse, r2=float(r2)))

# ---- classification ----
for y, tag in [(gt_m, 'gap_type'), (eh_c, 'stability')]:
    oof = np.full(n, -1); pf1 = []; pac = []
    for tr, te in folds:
        imp = SimpleImputer(strategy='median').fit(X[tr])
        m = XGBClassifier(objective='binary:logistic', eval_metric='logloss', **XP)
        m.fit(imp.transform(X[tr]), y[tr],
              sample_weight=compute_sample_weight('balanced', y[tr]))
        pred = m.predict(imp.transform(X[te])); oof[te] = pred
        pf1.append(f1_score(y[te], pred, average='macro', zero_division=0))
        pac.append(accuracy_score(y[te], pred))
    f1 = f1_score(y, oof, average='macro', zero_division=0); acc = accuracy_score(y, oof)
    print('CLF[%s] XGBoost     acc %.4f  macroF1 %.4f (perfold %.4f+-%.4f)' %
          (tag, acc, f1, np.mean(pf1), np.std(pf1)), flush=True)
    res.append(dict(task=tag, model='XGBoost', acc=float(acc), f1_macro=float(f1),
                    f1_perfold=float(np.mean(pf1)), f1_std=float(np.std(pf1)),
                    acc_perfold=float(np.mean(pac))))

json.dump(res, open(BASE + '/baseline_results.json', 'w'), indent=2)
print('\nAppended XGBoost rows -> baseline_results.json', flush=True)
