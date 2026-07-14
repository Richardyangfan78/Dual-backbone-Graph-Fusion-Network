"""
Plot ChalcoGNN v7 training curves from log file.
Generates: Train/Val Loss, BG MAE, GT Acc, EH Acc, Composite score per fold/seed.
"""

import re
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

LOG_FILE = "/path/to/Dual-backbone-Graph-Fusion-Network/logs/multitask_v7_20260304_230808.log"
OUT_DIR  = "/path/to/Dual-backbone-Graph-Fusion-Network/logs/plots_v7"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parse log ────────────────────────────────────────────────────────────────
EP_RE = re.compile(
    r"Ep\s+(\d+)\s+lr=[\d.e+-]+\s+dw=\[.*?\]\s+\|\s+"
    r"TrL\s+([-\d.]+)\s+VL\s+([-\d.]+)\s+\|\s+"
    r"BG\s+([\d.]+)\s+GT\s+([\d.]+)/([\d.]+)\s+EH_acc\s+([\d.]+)\s+\|\s+comp=([-\d.]+)"
)
FOLD_RE  = re.compile(r"Fold\s+(\d+)/3")
SEED_RE  = re.compile(r"--- Seed (\d+) ---")
SKIP_RE  = re.compile(r"\[SKIP\].*already complete")
EARLY_RE = re.compile(r"Early stop at epoch (\d+)")

data = {}   # {(fold, seed): {epoch:[], trl:[], vl:[], bg:[], gt_acc:[], gt_f1:[], eh:[], comp:[]}}
cur_fold = cur_seed = None

with open(LOG_FILE) as f:
    for line in f:
        m = FOLD_RE.search(line)
        if m:
            cur_fold = int(m.group(1))
            cur_seed = None
            continue
        m = SEED_RE.search(line)
        if m:
            cur_seed = int(m.group(1))
            key = (cur_fold, cur_seed)
            if key not in data:
                data[key] = dict(epoch=[], trl=[], vl=[], bg=[], gt_acc=[], gt_f1=[], eh=[], comp=[])
            continue
        if SKIP_RE.search(line):
            cur_seed = None          # skip cached fold
            continue
        if cur_fold and cur_seed:
            m = EP_RE.search(line)
            if m:
                ep, trl, vl, bg, gt_acc, gt_f1, eh, comp = m.groups()
                key = (cur_fold, cur_seed)
                data[key]['epoch'].append(int(ep))
                data[key]['trl'].append(float(trl))
                data[key]['vl'].append(float(vl))
                data[key]['bg'].append(float(bg))
                data[key]['gt_acc'].append(float(gt_acc))
                data[key]['gt_f1'].append(float(gt_f1))
                data[key]['eh'].append(float(eh))
                data[key]['comp'].append(float(comp))

folds = sorted(set(f for f, s in data))
seeds = sorted(set(s for f, s in data))
colors = {42: '#2196F3', 179: '#FF5722', 316: '#4CAF50'}
SEED_LABELS = {42: 'Seed 42', 179: 'Seed 179', 316: 'Seed 316'}

# ── Per-fold plots ────────────────────────────────────────────────────────────
metrics = [
    ('trl',    'Train Loss',        'Loss'),
    ('vl',     'Val Loss',          'Loss'),
    ('bg',     'BG MAE (eV)',       'MAE (eV)'),
    ('gt_acc', 'GT Accuracy',       'Accuracy'),
    ('gt_f1',  'GT F1',             'F1'),
    ('eh',     'EH Accuracy',       'Accuracy'),
    ('comp',   'Composite Score',   'Score'),
]

for fold in folds:
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(f'ChalcoGNN v7 — Fold {fold}/3 Training Curves', fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for ax_i, (key_m, title, ylabel) in enumerate(metrics):
        ax = axes[ax_i]
        for seed in seeds:
            k = (fold, seed)
            if k not in data or not data[k]['epoch']:
                continue
            d = data[k]
            ax.plot(d['epoch'], d[key_m], color=colors[seed],
                    label=SEED_LABELS[seed], linewidth=1.5, alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    for ax_i in range(len(metrics), len(axes)):
        axes[ax_i].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'fold{fold}_training_curves.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ── Summary plot: all folds, best metric per fold (composite) ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('ChalcoGNN v7 — All Folds Composite Score & Key Metrics', fontsize=14, fontweight='bold')

fold_colors = {1: '#9C27B0', 2: '#FF9800', 3: '#009688'}
linestyles  = {42: '-', 179: '--', 316: '-.'}

plot_configs = [
    ('comp',   'Composite Score',  'Score'),
    ('bg',     'BG MAE (eV)',      'MAE (eV)'),
    ('gt_acc', 'GT Accuracy',      'Accuracy'),
    ('gt_f1',  'GT F1',            'F1'),
    ('eh',     'EH Accuracy',      'Accuracy'),
]

for ax_i, (key_m, title, ylabel) in enumerate(plot_configs):
    ax = axes.flatten()[ax_i]
    for fold in folds:
        for seed in seeds:
            k = (fold, seed)
            if k not in data or not data[k]['epoch']:
                continue
            d = data[k]
            ax.plot(d['epoch'], d[key_m],
                    color=fold_colors[fold],
                    linestyle=linestyles[seed],
                    linewidth=1.2, alpha=0.75,
                    label=f'F{fold}S{seed}')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Epoch', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=7, ncol=3)

axes.flatten()[-1].set_visible(False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'all_folds_summary.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# ── Loss-only combined plot ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(folds), figsize=(15, 4), sharey=False)
fig.suptitle('ChalcoGNN v7 — Train vs Val Loss per Fold', fontsize=13, fontweight='bold')

for ax, fold in zip(axes, folds):
    for seed in seeds:
        k = (fold, seed)
        if k not in data or not data[k]['epoch']:
            continue
        d = data[k]
        ax.plot(d['epoch'], d['trl'], color=colors[seed], linestyle='-',
                linewidth=1.4, alpha=0.8, label=f'Train S{seed}')
        ax.plot(d['epoch'], d['vl'], color=colors[seed], linestyle=':',
                linewidth=1.2, alpha=0.6, label=f'Val S{seed}')
    ax.set_title(f'Fold {fold}', fontsize=11)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'train_val_loss_all_folds.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

print("\nAll plots generated successfully.")
