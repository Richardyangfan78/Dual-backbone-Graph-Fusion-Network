import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
res = json.load(open(BASE + '/baseline_results.json'))

def get(task, model, key):
    for r in res:
        if r['task'] == task and r['model'] == model:
            return r.get(key)
    return None

# 8 base models; the linear slot is Ridge (regression) / Logistic (classification)
order = [('Linear', 'Ridge', 'Logistic'), ('KNN', 'KNN', 'KNN'), ('SVM', 'SVM', 'SVM'),
         ('RandomForest', 'RandomForest', 'RandomForest'), ('ExtraTrees', 'ExtraTrees', 'ExtraTrees'),
         ('HistGBM', 'HistGBM', 'HistGBM'), ('XGBoost', 'XGBoost', 'XGBoost'),
         ('LightGBM', 'LightGBM', 'LightGBM')]
labels = [o[0] for o in order]

# Gate-Fusion GNN per-fold test metrics (checkpoints/dual_backbone_inorg/foldN_results.csv)
G = dict(bandgap=[0.212203, 0.205738, 0.245390, 0.218833, 0.226735],
         gap_type=[0.883820, 0.821624, 0.865072, 0.823208, 0.844456],
         stability=[0.956900, 0.967983, 0.960654, 0.929116, 0.951934])
gnn_m = {k: float(np.mean(v)) for k, v in G.items()}
gnn_s = {k: float(np.std(v)) for k, v in G.items()}
dummy = dict(bandgap=get('bandgap', 'Dummy(mean)', 'mae_ev'),
             gap_type=get('gap_type', 'Dummy(freq)', 'f1_macro'),
             stability=get('stability', 'Dummy(freq)', 'f1_macro'))

panels = [('bandgap', 'mae_ev', 'mae_std', 'Bandgap MAE  (eV)  ↓ lower = better', False),
          ('gap_type', 'f1_macro', 'f1_std', 'Gap-type  macro-F1  ↑ higher = better', True),
          ('stability', 'f1_macro', 'f1_std', 'Stability  macro-F1  ↑ higher = better', True)]

C_CLS, C_BEST, C_GNN = '#4C72B0', '#DD8452', '#C44E52'
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))

for ax, (task, key, ekey, title, higher) in zip(axes, panels):
    models = [(o[1] if task == 'bandgap' else o[2]) for o in order]
    vals = [get(task, m, key) for m in models]
    errs = [get(task, m, ekey) for m in models]
    best_i = int(np.argmax(vals)) if higher else int(np.argmin(vals))
    colors = [C_CLS] * 8; colors[best_i] = C_BEST
    x = np.arange(8)
    bars = ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor='black',
                  linewidth=0.6, width=0.72, zorder=3)
    gx = 9
    gb = ax.bar([gx], [gnn_m[task]], yerr=[gnn_s[task]], capsize=3, color=C_GNN,
                edgecolor='black', linewidth=0.9, width=0.72, zorder=3)

    allv = vals + [gnn_m[task]]
    top = (1.06 if higher else max(allv) * 1.32)
    ax.set_ylim(0, top)
    # dummy reference
    if dummy[task] <= top:
        ax.axhline(dummy[task], ls=':', color='gray', lw=1.4, zorder=1)
        ax.text(9.7, dummy[task], 'Dummy', va='center', ha='left', fontsize=8, color='gray')
    else:
        ax.text(0.015, 0.975, 'Dummy(mean) = %.2f  (off-scale)' % dummy[task],
                transform=ax.transAxes, va='top', ha='left', fontsize=8.5, color='gray')

    for b, v in zip(list(bars) + list(gb), allv):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, '%.3f' % v,
                ha='center', va='bottom', fontsize=8.2,
                fontweight='bold' if (b in gb) else 'normal')

    ax.set_xticks(list(x) + [gx])
    xl = ax.set_xticklabels(labels + ['Gate-Fusion\nGNN'], rotation=38, ha='right', fontsize=9.2)
    xl[best_i].set_fontweight('bold'); xl[best_i].set_color(C_BEST)
    xl[-1].set_fontweight('bold'); xl[-1].set_color(C_GNN)
    ax.set_title(title, fontsize=11.5, fontweight='bold', pad=8)
    ax.grid(axis='y', ls='--', alpha=0.4, zorder=0); ax.set_axisbelow(True)
    ax.set_ylabel(key.replace('_', ' '), fontsize=9)

leg = [Patch(fc=C_CLS, ec='k', label='Classical ML (matminer feats)'),
       Patch(fc=C_BEST, ec='k', label='Best classical (per task)'),
       Patch(fc=C_GNN, ec='k', label='Gate-Fusion GNN (formal model)')]
fig.legend(handles=leg, loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Traditional ML (matminer 154 features) vs Gate-Fusion GNN\n'
             'Chalcohalide inorganic dataset  (N=1768, identical 5-fold CV, error bars = per-fold std)',
             fontsize=12.5, fontweight='bold')
fig.tight_layout(rect=[0, 0.04, 1, 0.93])
out = BASE + '/model_comparison.png'
fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='white')
print('SAVED', out)
