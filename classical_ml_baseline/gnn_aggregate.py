import os, csv, json, numpy as np
CK = '/path/to/Dual-backbone-Graph-Fusion-Network/checkpoints'
BASE = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'
dirs = [('CGCNN', 'cgcnn_mt_inorg'), ('ALIGNN', 'alignn_mt_inorg'),
        ('M3GNet', 'm3gnet_mt_inorg'), ('MACE', 'mace_mt_inorg'),
        ('MACE+M3GNet', 'mace_m3gnet_inorg'), ('MACE+ALIGNN', 'dual_backbone_inorg')]
keys = ['bg_mae', 'gt_acc', 'gt_f1', 'eh_acc', 'eh_f1']
out = {}
for name, d in dirs:
    acc = {k: [] for k in keys}
    for f in range(5):
        p = os.path.join(CK, d, 'fold%d_results.csv' % f)
        if not os.path.exists(p):
            continue
        m = {r[0]: r[1] for r in csv.reader(open(p)) if len(r) == 2}
        for k in keys:
            if k in m:
                try:
                    acc[k].append(float(m[k]))
                except Exception:
                    pass
    out[name] = {k: dict(mean=float(np.mean(v)) if v else None,
                         std=float(np.std(v)) if v else None, n=len(v))
                 for k, v in acc.items()}
    print('%-12s' % name,
          ' '.join('%s=%.4f' % (k, np.mean(acc[k])) if acc[k] else '%s=NA' % k for k in keys),
          '| folds=%d' % len(acc['bg_mae']))
json.dump(out, open(BASE + '/gnn_backbones.json', 'w'), indent=2)
print('SAVED gnn_backbones.json')
