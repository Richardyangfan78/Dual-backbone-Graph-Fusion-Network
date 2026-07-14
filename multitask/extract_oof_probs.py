"""OOF probability extraction for MACE+ALIGNN and ALIGNN dual-backbone models."""
import argparse, csv, os, sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

def extract_dual_backbone(data_dir, ckpt_dir, alignn_dir, mace_pretrained,
                          out_csv, k_folds=5, seed=42):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model_dual_backbone import DualBackboneMultiTask
    from model_alignn_pyg import ALIGNNMultiTaskPyG
    from data_dual_backbone import DualBackboneDataset, stratified_kfold_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DualBackboneDataset(
        data_dir,
        mace_cache_dir=os.path.join(data_dir, "mace_cached_graphs"),
        crystal_cache_dir=os.path.join(data_dir, "crystal_cached_graphs"),
        merge_metal_indirect=True, log_bg=True
    )
    n_samples = len(dataset)
    print(f"Dataset: {n_samples} samples")

    all_rows = []

    for fold in range(k_folds):
        ckpt_path = os.path.join(ckpt_dir, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] fold {fold}: checkpoint not found")
            continue
        print(f"\n--- Fold {fold} ---")

        # Build model with same config as training
        alignn_ckpt = os.path.join(alignn_dir, f"fold{fold}_best.pt")
        alignn_model = ALIGNNMultiTaskPyG(
            hidden_dim=256, n_alignn_layers=4, n_gcn_layers=4,
            dropout=0.3, radius=8.0, max_num_nbr=12,
            dist_bins=80, angle_bins=40
        ).to(device)
        if os.path.exists(alignn_ckpt):
            ac = torch.load(alignn_ckpt, map_location=device, weights_only=False)
            sd = ac.get("model_state_dict", ac.get("state_dict", ac))
            alignn_model.load_state_dict(sd, strict=False)
        alignn_model.eval()

        model = DualBackboneMultiTask(
            mace_model_type="small",
            mace_pretrained_path=mace_pretrained,
            alignn_model=alignn_model,
            h_fea_len=256, n_attn_heads=8, dropout=0.3,
            use_cosine_classifier=True, cosine_temp=0.1
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval()

        _, _, test_idx = stratified_kfold_split(
            dataset, k_folds=k_folds, fold=fold, seed=seed, val_ratio=0.15
        )
        test_set = [dataset[i] for i in test_idx]
        loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                bg_pred, gt_pred, eh_pred = model(batch)

                gt_known = (batch.gt >= 0)
                gt_probs = gt_pred.exp()[:, 1]  # prob of direct gap
                eh_probs = eh_pred.exp()[:, 1]  # prob of stable

                for j in range(batch.num_graphs):
                    gt_true = batch.gt[j].item()
                    eh_true = batch.eh[j].item()
                    mp_id   = batch.mp_id[j] if hasattr(batch, 'mp_id') else f"fold{fold}_{j}"
                    all_rows.append({
                        "fold": fold,
                        "mp_id": mp_id,
                        "true_gt": gt_true,
                        "true_eh": eh_true,
                        "pred_gt_prob": gt_probs[j].item() if gt_known[j] else -1,
                        "pred_eh_prob": eh_probs[j].item(),
                    })

        print(f"  fold {fold}: {len(test_idx)} test samples extracted")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold","mp_id","true_gt","true_eh",
                                                "pred_gt_prob","pred_eh_prob"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows → {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--ckpt-dir",         required=True)
    p.add_argument("--alignn-dir",       required=True)
    p.add_argument("--mace-pretrained",  required=True)
    p.add_argument("--out-csv",          required=True)
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--seed",    type=int, default=42)
    args = p.parse_args()
    extract_dual_backbone(
        args.data_dir, args.ckpt_dir, args.alignn_dir,
        args.mace_pretrained, args.out_csv, args.k_folds, args.seed
    )
