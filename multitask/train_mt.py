"""
Multi-task CGCNN v2 training script.

Key improvements over v1:
  - Uncertainty-based learnable loss weighting (Kendall 2018)
  - Larger backbone matching single-task model capacity
  - Stratified data split to preserve class distribution
  - Inverse-frequency class weights for gap type
  - Cosine annealing LR scheduler
  - Gradient clipping
  - Early stopping
"""
import argparse
import os
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))

from data_mt import CIFDataMultiTask, collate_pool_mt, get_train_val_test_loader
from model_mt import CrystalGraphConvNetMT


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-task CGCNN v2')
    parser.add_argument('data_dir', help='Path to data directory')
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--optim', default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--atom-fea-len', default=128, type=int)
    parser.add_argument('--h-fea-len', default=256, type=int)
    parser.add_argument('--n-conv', default=5, type=int)
    parser.add_argument('--n-h', default=2, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--train-ratio', default=0.8, type=float)
    parser.add_argument('--val-ratio', default=0.1, type=float)
    parser.add_argument('--test-ratio', default=0.1, type=float)
    parser.add_argument('-j', '--workers', default=0, type=int)
    parser.add_argument('--print-freq', default=10, type=int)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--patience', default=80, type=int,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--grad-clip', default=5.0, type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                        help='Use inverse-frequency class weights for gap type')
    parser.add_argument('--checkpoint-dir', default='checkpoints/multitask_v2',
                        type=str)
    return parser.parse_args()


class Normalizer:
    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = torch.mean(tensor).item()
            self.std = torch.std(tensor).item()
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def main():
    args = parse_args()
    use_cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')

    # Load dataset
    dataset = CIFDataMultiTask(args.data_dir)
    print(f'Dataset size: {len(dataset)}')

    # Print class distribution
    class_counts = {}
    for row in dataset.id_prop_data:
        label = int(row[2])
        class_counts[label] = class_counts.get(label, 0) + 1
    print(f'Gap type distribution: {dict(sorted(class_counts.items()))}')

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool_mt,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.workers,
        pin_memory=use_cuda,
        return_test=True,
        stratified=True,
    )

    # Compute normalizers for regression targets
    sample_size = min(500, len(dataset))
    sample_indices = sample(range(len(dataset)), sample_size)
    sample_bandgaps = torch.cat([dataset[i][1]['bandgap'] for i in sample_indices])
    sample_ehulls = torch.cat([dataset[i][1]['ehull'] for i in sample_indices])
    normalizer_bg = Normalizer(sample_bandgaps)
    normalizer_eh = Normalizer(sample_ehulls)
    print(f'Band gap normalizer: mean={normalizer_bg.mean:.4f}, std={normalizer_bg.std:.4f}')
    print(f'E_hull normalizer: mean={normalizer_eh.mean:.4f}, std={normalizer_eh.std:.4f}')

    # Build model
    (sample_atom_fea, sample_nbr_fea, _), _, _ = dataset[0]
    orig_atom_fea_len = sample_atom_fea.shape[-1]
    nbr_fea_len = sample_nbr_fea.shape[-1]

    model = CrystalGraphConvNetMT(
        orig_atom_fea_len, nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        n_gap_classes=3,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # Loss functions
    criterion_reg = nn.MSELoss()
    if args.use_class_weights:
        class_weights = dataset.get_class_weights().to(device)
        print(f'Class weights: {class_weights.tolist()}')
        criterion_cls = nn.NLLLoss(weight=class_weights)
    else:
        criterion_cls = nn.NLLLoss()

    # Optimizer
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=0.9,
                              weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                  eta_min=args.lr * 0.01)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_metrics = run_epoch(
            train_loader, model, criterion_reg, criterion_cls,
            normalizer_bg, normalizer_eh, args, device,
            optimizer=optimizer, is_train=True,
        )

        # Validate
        val_loss, val_metrics = run_epoch(
            val_loader, model, criterion_reg, criterion_cls,
            normalizer_bg, normalizer_eh, args, device,
            optimizer=None, is_train=False,
        )

        scheduler.step()

        # Log learned task weights
        with torch.no_grad():
            w = torch.exp(-model.log_vars).cpu().numpy()
        print(f'Epoch {epoch+1:03d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'BG MAE: {val_metrics["bg_mae"]:.4f} eV | '
              f'GT Acc: {val_metrics["gt_acc"]:.4f} | '
              f'GT F1: {val_metrics["gt_f1"]:.4f} | '
              f'EH MAE: {val_metrics["eh_mae"]:.4f} eV/atom | '
              f'Weights: BG={w[0]:.3f} GT={w[1]:.3f} EH={w[2]:.3f}')

        # Save checkpoint + early stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'normalizer_bg': normalizer_bg.state_dict(),
            'normalizer_eh': normalizer_eh.state_dict(),
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }
        torch.save(checkpoint,
                    os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar'))
        if is_best:
            torch.save(checkpoint,
                        os.path.join(args.checkpoint_dir, 'model_best.pth.tar'))
            print(f'  ** Best model saved (val_loss={val_loss:.4f})')

        if args.patience > 0 and no_improve_count >= args.patience:
            print(f'Early stopping at epoch {epoch+1} '
                  f'(no improvement for {args.patience} epochs)')
            break

    # Test with best model
    print('\n=== Final Test (best model) ===')
    best_ckpt = os.path.join(args.checkpoint_dir, 'model_best.pth.tar')
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        print(f'Loaded best model from epoch {ckpt["epoch"]}')

    test_loss, test_metrics = run_epoch(
        test_loader, model, criterion_reg, criterion_cls,
        normalizer_bg, normalizer_eh, args, device,
        optimizer=None, is_train=False,
    )
    print(f'Test Loss: {test_loss:.4f}')
    print(f'  Band Gap MAE:    {test_metrics["bg_mae"]:.4f} eV')
    print(f'  Gap Type Acc:    {test_metrics["gt_acc"]:.4f}')
    print(f'  Gap Type F1:     {test_metrics["gt_f1"]:.4f}')
    print(f'  E_hull MAE:      {test_metrics["eh_mae"]:.4f} eV/atom')


def run_epoch(data_loader, model, criterion_reg, criterion_cls,
              normalizer_bg, normalizer_eh, args, device,
              optimizer=None, is_train=True):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_bg_pred, all_bg_true = [], []
    all_gt_pred, all_gt_true = [], []
    all_eh_pred, all_eh_true = [], []
    n_batches = 0

    for i, (inputs, targets, _) in enumerate(data_loader):
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = inputs
        if device.type == 'cuda':
            atom_fea = atom_fea.cuda()
            nbr_fea = nbr_fea.cuda()
            nbr_fea_idx = nbr_fea_idx.cuda()

        target_bg = normalizer_bg.norm(targets['bandgap']).to(device)
        target_gt = targets['gaptype'].squeeze(-1).to(device)
        target_eh = normalizer_eh.norm(targets['ehull']).to(device)

        if is_train:
            pred_bg, pred_gt, pred_eh = model(atom_fea, nbr_fea, nbr_fea_idx,
                                              crystal_atom_idx)
        else:
            with torch.no_grad():
                pred_bg, pred_gt, pred_eh = model(atom_fea, nbr_fea, nbr_fea_idx,
                                                  crystal_atom_idx)

        loss_bg = criterion_reg(pred_bg, target_bg)
        loss_gt = criterion_cls(pred_gt, target_gt)
        loss_eh = criterion_reg(pred_eh, target_eh)

        # Uncertainty-weighted loss
        loss = model.compute_weighted_loss(loss_bg, loss_gt, loss_eh)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Collect predictions for metrics
        bg_pred_denorm = normalizer_bg.denorm(pred_bg.detach().cpu())
        eh_pred_denorm = normalizer_eh.denorm(pred_eh.detach().cpu())
        all_bg_pred.append(bg_pred_denorm)
        all_bg_true.append(targets['bandgap'])
        all_gt_pred.append(pred_gt.detach().cpu().argmax(dim=1))
        all_gt_true.append(targets['gaptype'].squeeze(-1))
        all_eh_pred.append(eh_pred_denorm)
        all_eh_true.append(targets['ehull'])

    avg_loss = total_loss / max(n_batches, 1)

    all_bg_pred = torch.cat(all_bg_pred).numpy().flatten()
    all_bg_true = torch.cat(all_bg_true).numpy().flatten()
    all_gt_pred = torch.cat(all_gt_pred).numpy()
    all_gt_true = torch.cat(all_gt_true).numpy()
    all_eh_pred = torch.cat(all_eh_pred).numpy().flatten()
    all_eh_true = torch.cat(all_eh_true).numpy().flatten()

    metrics = {
        'bg_mae': np.mean(np.abs(all_bg_pred - all_bg_true)),
        'gt_acc': accuracy_score(all_gt_true, all_gt_pred),
        'gt_f1': f1_score(all_gt_true, all_gt_pred, average='macro',
                          zero_division=0),
        'eh_mae': np.mean(np.abs(all_eh_pred - all_eh_true)),
    }
    return avg_loss, metrics


if __name__ == '__main__':
    main()
