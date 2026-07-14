import json, numpy as np
BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
res = json.load(open(BASE + '/baseline_results.json'))

# --- Gate-Fusion (dual_backbone_inorg) per-fold test metrics (from checkpoints/*/foldN_results.csv) ---
gnn = dict(
    bg_mae=[0.212203, 0.205738, 0.245390, 0.218833, 0.226735],
    gt_acc=[0.920904, 0.867232, 0.901130, 0.869688, 0.889518],
    gt_f1 =[0.883820, 0.821624, 0.865072, 0.823208, 0.844456],
    eh_acc=[0.968927, 0.977401, 0.971751, 0.949008, 0.966006],
    eh_f1 =[0.956900, 0.967983, 0.960654, 0.929116, 0.951934],
)
def ms(x): return np.mean(x), np.std(x)

def reg_row(r): return (r['model'], r['mae_ev'], r['mae_std'], r['rmse'], r['r2'])
def clf_rows(tag): return [(r['model'], r['acc'], r['f1_macro'], r['f1_std'])
                           for r in res if r['task'] == tag]

L = []
L.append('# Classical ML + matminer  vs  Gate-Fusion GNN  (Chalcohalide inorganic dataset)\n')
L.append('Dataset: `Data/Inorganic_datasets`, N=1768 structures. Identical 5-fold '
         'StratifiedKFold(seed=42, strat=gap_type*2+stability). Features: 154 matminer '
         'descriptors (Magpie ElementProperty + Stoichiometry + ValenceOrbital + IonProperty '
         '+ TMetalFraction + BandCenter + DensityFeatures + GlobalSymmetry + StructuralComplexity).\n')

L.append('## 1) Bandgap regression  (MAE in eV, lower better)\n')
L.append('| Model | eV-MAE | RMSE | R2 |')
L.append('|---|---|---|---|')
best_reg = min((r for r in res if r['task'] == 'bandgap' and r['model'] != 'Dummy(mean)'),
               key=lambda r: r['mae_ev'])
for r in res:
    if r['task'] == 'bandgap':
        m, mae, std, rmse, r2 = reg_row(r)
        star = '  **<-- best classical**' if r['model'] == best_reg['model'] else ''
        L.append('| %s | %.3f | %.3f | %.3f |%s' % (m, mae, rmse, r2, star))
mm, ms_ = ms(gnn['bg_mae'])
L.append('| **Gate-Fusion GNN** | **%.3f** | - | - |  **(formal model)**' % mm)
L.append('')

L.append('## 2) Gap-type classification  (direct vs indirect/metal; macro-F1, higher better)\n')
L.append('| Model | Accuracy | macro-F1 |')
L.append('|---|---|---|')
best_gt = max((r for r in res if r['task'] == 'gap_type' and not r['model'].startswith('Dummy')),
              key=lambda r: r['f1_macro'])
for m, acc, f1, std in clf_rows('gap_type'):
    star = '  **<-- best classical**' if m == best_gt['model'] else ''
    L.append('| %s | %.3f | %.3f |%s' % (m, acc, f1, star))
am, _ = ms(gnn['gt_acc']); fm, _ = ms(gnn['gt_f1'])
L.append('| **Gate-Fusion GNN** | **%.3f** | **%.3f** |  **(formal model)**' % (am, fm))
L.append('')

L.append('## 3) Stability classification  (E_hull<0.1 stable vs unstable; macro-F1, higher better)\n')
L.append('| Model | Accuracy | macro-F1 |')
L.append('|---|---|---|')
best_eh = max((r for r in res if r['task'] == 'stability' and not r['model'].startswith('Dummy')),
              key=lambda r: r['f1_macro'])
for m, acc, f1, std in clf_rows('stability'):
    star = '  **<-- best classical**' if m == best_eh['model'] else ''
    L.append('| %s | %.3f | %.3f |%s' % (m, acc, f1, star))
am, _ = ms(gnn['eh_acc']); fm, _ = ms(gnn['eh_f1'])
L.append('| **Gate-Fusion GNN** | **%.3f** | **%.3f** |  **(formal model)**' % (am, fm))
L.append('')

L.append('## Summary (best classical vs GNN)\n')
L.append('| Task | Metric | Best classical | Gate-Fusion GNN | GNN advantage |')
L.append('|---|---|---|---|---|')
L.append('| Bandgap | eV-MAE | %.3f (%s) | %.3f | %.0f%% lower error |' %
         (best_reg['mae_ev'], best_reg['model'], ms(gnn['bg_mae'])[0],
          100 * (1 - ms(gnn['bg_mae'])[0] / best_reg['mae_ev'])))
L.append('| Gap-type | macro-F1 | %.3f (%s) | %.3f | +%.3f |' %
         (best_gt['f1_macro'], best_gt['model'], ms(gnn['gt_f1'])[0],
          ms(gnn['gt_f1'])[0] - best_gt['f1_macro']))
L.append('| Stability | macro-F1 | %.3f (%s) | %.3f | +%.3f |' %
         (best_eh['f1_macro'], best_eh['model'], ms(gnn['eh_f1'])[0],
          ms(gnn['eh_f1'])[0] - best_eh['f1_macro']))

txt = '\n'.join(L)
open(BASE + '/COMPARISON.md', 'w').write(txt)
print(txt)
print('\nSAVED', BASE + '/COMPARISON.md')
