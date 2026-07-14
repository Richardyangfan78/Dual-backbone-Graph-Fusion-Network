import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
res = json.load(open(BASE + '/baseline_results.json'))
gb = json.load(open(BASE + '/gnn_backbones.json'))

def cget(task, model, key):
    for r in res:
        if r['task'] == task and r['model'] == model:
            return r.get(key)
    return None

order = [('Linear', 'Ridge', 'Logistic'), ('KNN', 'KNN', 'KNN'), ('SVM', 'SVM', 'SVM'),
         ('RandomForest', 'RandomForest', 'RandomForest'), ('ExtraTrees', 'ExtraTrees', 'ExtraTrees'),
         ('HistGBM', 'HistGBM', 'HistGBM'), ('XGBoost', 'XGBoost', 'XGBoost'),
         ('LightGBM', 'LightGBM', 'LightGBM')]
clabels = [o[0] for o in order]
# task -> (classical value key, classical std key, gnn key, higher_better, axis title, dummy)
TASKS = [('bandgap', 'mae_ev', 'mae_std', 'bg_mae', False, 'Bandgap MAE  (eV)  ↓ lower = better'),
         ('gap_type', 'f1_macro', 'f1_std', 'gt_f1', True, 'Gap-type  macro-F1  ↑ higher = better'),
         ('stability', 'f1_macro', 'f1_std', 'eh_f1', True, 'Stability  macro-F1  ↑ higher = better')]
dummy = dict(bandgap=cget('bandgap', 'Dummy(mean)', 'mae_ev'),
             gap_type=cget('gap_type', 'Dummy(freq)', 'f1_macro'),
             stability=cget('stability', 'Dummy(freq)', 'f1_macro'))

def classical_vals(task, vkey, ekey):
    models = [(o[1] if task == 'bandgap' else o[2]) for o in order]
    return ([cget(task, m, vkey) for m in models], [cget(task, m, ekey) for m in models])

def best_classical(task, vkey, higher):
    v, _ = classical_vals(task, vkey, None if True else None)  # vals only
    return (max(v) if higher else min(v))

C_CLS, C_BEST, C_CGCNN, C_FUSE, C_OURS = '#4C72B0', '#DD8452', '#55A868', '#DD8452', '#C44E52'

# ============================================================== FIGURE 1
fig1, axes = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, (task, vkey, ekey, gkey, higher, title) in zip(axes, TASKS):
    vals, errs = classical_vals(task, vkey, ekey)
    cg = gb['CGCNN'][gkey]['mean']; cgs = gb['CGCNN'][gkey]['std']
    best_i = int(np.argmax(vals)) if higher else int(np.argmin(vals))
    colors = [C_CLS] * 8; colors[best_i] = C_BEST
    x = np.arange(8)
    bars = ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor='black', lw=0.6, width=0.72, zorder=3)
    gx = 9
    cb = ax.bar([gx], [cg], yerr=[cgs], capsize=3, color=C_CGCNN, edgecolor='black', lw=0.9, width=0.72, zorder=3)
    allv = vals + [cg]
    top = (1.06 if higher else max(allv) * 1.34); ax.set_ylim(0, top)
    if dummy[task] <= top:
        ax.axhline(dummy[task], ls=':', color='gray', lw=1.3); ax.text(9.7, dummy[task], 'Dummy', va='center', fontsize=8, color='gray')
    else:
        ax.text(0.015, 0.975, 'Dummy(mean)=%.2f (off-scale)' % dummy[task], transform=ax.transAxes, va='top', fontsize=8.3, color='gray')
    star = '*' if task == 'gap_type' else ''
    for b, v, s in zip(list(bars) + list(cb), allv, [''] * 8 + [star]):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, '%.3f%s' % (v, s), ha='center', va='bottom',
                fontsize=8.2, fontweight='bold' if b in cb else 'normal')
    ax.set_xticks(list(x) + [gx])
    xl = ax.set_xticklabels(clabels + ['CGCNN\n(GNN)'], rotation=38, ha='right', fontsize=9.2)
    xl[best_i].set_fontweight('bold'); xl[best_i].set_color(C_BEST)
    xl[-1].set_fontweight('bold'); xl[-1].set_color(C_CGCNN)
    ax.set_title(title, fontsize=11.5, fontweight='bold'); ax.grid(axis='y', ls='--', alpha=0.4); ax.set_axisbelow(True)
leg = [Patch(fc=C_CLS, ec='k', label='Classical ML (matminer feats)'), Patch(fc=C_BEST, ec='k', label='Best classical'),
       Patch(fc=C_CGCNN, ec='k', label='CGCNN (basic GNN)')]
fig1.legend(handles=leg, loc='lower center', ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig1.suptitle('Classical ML (matminer 154 feats) vs CGCNN  —  Chalcohalide inorganic (N=1768, identical 5-fold CV)\n'
              '* CGCNN gap-type F1 uses original-CGCNN eval convention (F1>acc), not directly comparable',
              fontsize=11.8, fontweight='bold')
fig1.tight_layout(rect=[0, 0.04, 1, 0.92])
fig1.savefig(BASE + '/fig1_classical_vs_cgcnn.png', dpi=170, bbox_inches='tight', facecolor='white')
print('SAVED fig1_classical_vs_cgcnn.png')

# ============================================================== FIGURE 2
B = [('CGCNN', 'single'), ('ALIGNN', 'single'), ('M3GNet', 'single'), ('MACE', 'single'),
     ('MACE+M3GNet', 'fuse'), ('MACE+ALIGNN', 'ours')]
bnames = [b[0] for b in B]
cmap = dict(single=C_CLS, fuse=C_FUSE, ours=C_OURS)
fig2, axes = plt.subplots(1, 3, figsize=(15, 5.8))
for ax, (task, vkey, ekey, gkey, higher, title) in zip(axes, TASKS):
    vals = [gb[n][gkey]['mean'] for n in bnames]
    errs = [gb[n][gkey]['std'] for n in bnames]
    colors = [cmap[b[1]] for b in B]
    x = np.arange(6)
    bars = ax.bar(x, vals, yerr=errs, capsize=3, color=colors, edgecolor='black', lw=0.7, width=0.66, zorder=3)
    bc = best_classical(task, vkey, higher)
    ax.axhline(bc, ls='--', color='#8172B3', lw=1.6, zorder=2)
    allv = vals + [bc]
    top = (1.07 if higher else max(allv) * 1.3); ax.set_ylim(0, top)
    ax.text(5.6, bc, 'best\nclassical', va='center', ha='left', fontsize=8, color='#8172B3', fontweight='bold')
    star = ['', '', '', '', '', ''];
    if task == 'gap_type': star[0] = '*'
    for b, v, s in zip(bars, vals, star):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, '%.3f%s' % (v, s), ha='center', va='bottom',
                fontsize=8.6, fontweight='bold' if B[list(bars).index(b)][1] == 'ours' else 'normal')
    ax.set_xticks(x)
    xl = ax.set_xticklabels(bnames, rotation=30, ha='right', fontsize=9.4)
    xl[-1].set_color(C_OURS); xl[-1].set_fontweight('bold')
    ax.set_title(title, fontsize=11.5, fontweight='bold'); ax.grid(axis='y', ls='--', alpha=0.4); ax.set_axisbelow(True)
leg = [Patch(fc=C_CLS, ec='k', label='Single backbone'), Patch(fc=C_FUSE, ec='k', label='Fusion (MACE+M3GNet)'),
       Patch(fc=C_OURS, ec='k', label='Gate-Fusion: MACE+ALIGNN (ours)'),
       plt.Line2D([0], [0], ls='--', color='#8172B3', lw=1.6, label='Best classical ML')]
fig2.legend(handles=leg, loc='lower center', ncol=4, frameon=False, fontsize=9.6, bbox_to_anchor=(0.5, -0.02))
fig2.suptitle('GNN backbones — single vs graph-fusion  (Chalcohalide inorganic, N=1768, identical 5-fold CV)\n'
              'Fusion of MACE + ALIGNN (ours) gives the lowest bandgap error and top-tier classification\n'
              '* CGCNN gap-type F1 uses original-CGCNN eval convention, not directly comparable',
              fontsize=11.6, fontweight='bold')
fig2.tight_layout(rect=[0, 0.04, 1, 0.90])
fig2.savefig(BASE + '/fig2_gnn_backbones.png', dpi=170, bbox_inches='tight', facecolor='white')
print('SAVED fig2_gnn_backbones.png')
