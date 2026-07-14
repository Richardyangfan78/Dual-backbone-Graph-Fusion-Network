"""Recompute CGCNN v10 EH F1 for all 5 folds."""
import sys, os, csv, random
sys.path.insert(0, os.path.dirname(__file__))

import torch, numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import f1_score, accuracy_score

from data_mt_v4 import stratified_kfold_v4, stratified_split_from_indices
from model_mt_v10 import CrystalGraphConvNetMTV10
from train_mt_v10 import collate_pool_mt_v7, Normalizer, CachedGraphDatasetV7

DATA_DIR  = "/path/to/Dual-backbone-Graph-Fusion-Network/Data/multitask"
CACHE_DIR = "/path/to/Dual-backbone-Graph-Fusion-Network/Data/multitask/cached_graphs"
CKPT_BASE = "/path/to/Dual-backbone-Graph-Fusion-Network/checkpoints/multitask_v10"
K_FOLDS = 5; SEED = 42; MERGE = True; BATCH = 48; VAL_RATIO = 0.1

device = torch.device("cpu")

with open(os.path.join(DATA_DIR, "id_prop.csv")) as f:
    id_prop_data = list(csv.reader(f))
random.seed(123); random.shuffle(id_prop_data)

dataset = CachedGraphDatasetV7(CACHE_DIR, id_prop_data, merge_metal_indirect=MERGE,
                                augment_noise=0.0, data_dir=None)
print(f"Dataset: {len(dataset)} | n_gap_classes={dataset.n_gap_classes}")

folds = stratified_kfold_v4(dataset.id_prop_data, K_FOLDS, merge_metal_indirect=MERGE)

results = []
for fi in range(K_FOLDS):
    test_idx = folds[fi]
    train_val_idx = [idx for j in range(K_FOLDS) if j != fi for idx in folds[j]]
    train_idx, val_idx = stratified_split_from_indices(
        dataset.id_prop_data, train_val_idx,
        train_ratio=1.0 - VAL_RATIO, random_seed=123 + fi,
        merge_metal_indirect=MERGE)

    ckpt_path = os.path.join(CKPT_BASE, f"fold_{fi}_seed_{SEED}", "best_composite.pth.tar")
    if not os.path.exists(ckpt_path):
        print(f"Fold {fi}: missing {ckpt_path}"); continue

    tr_bg = torch.tensor([float(dataset.id_prop_data[i][1]) for i in train_idx])
    n_bg  = Normalizer(tr_bg, log_transform=False, robust=False)

    (s_af, s_nf, _), _, _ = dataset[0]
    model = CrystalGraphConvNetMTV10(
        s_af.shape[-1], s_nf.shape[-1],
        atom_fea_len=160, n_conv=8, h_fea_len=320, n_h=1,
        n_gap_classes=dataset.n_gap_classes, n_eh_classes=2,
        dropout=0.35, n_attn_heads=8,
        use_cosine_classifier=True, cosine_temp=0.1,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    n_bg.load_state_dict(ckpt["normalizer_bg"])
    model.eval()

    loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_idx),
                        batch_size=BATCH, collate_fn=collate_pool_mt_v7, num_workers=0)
    all_t, all_p = [], []
    with torch.no_grad():
        for batch, targets, _ in loader:
            af, nf, ni, ci = batch[0], batch[1], batch[2], batch[3]
            _, _, p_eh = model(af, nf, ni, ci)
            all_p.append(p_eh.argmax(1).numpy())
            all_t.append(targets["eh_label"].squeeze(-1).numpy())

    t, p = np.concatenate(all_t), np.concatenate(all_p)
    acc = accuracy_score(t, p)
    f1  = f1_score(t, p, average="binary", zero_division=0)
    print(f"Fold {fi}: EH_acc={acc:.4f}  EH_F1={f1:.4f}  (n={len(t)})")
    results.append({"fold": fi, "eh_acc": acc, "eh_f1": f1})

accs = [r["eh_acc"] for r in results]
f1s  = [r["eh_f1"]  for r in results]
print(f"\nCGCNN EH Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  {[round(x,4) for x in accs]}")
print(f"CGCNN EH F1 : {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  {[round(x,4) for x in f1s]}")
