"""
Multi-task CGCNN v3: Optimized architecture for small datasets.

Key improvements over v2:
  - Right-sized backbone (V1-level capacity, ~90K params)
  - Attention-weighted crystal pooling (learns atom importance)
  - Hierarchical prediction: bandgap pred conditions gap type head
  - No learnable loss weighting (use fixed task weights externally)
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.model import ConvLayer


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over atoms in a crystal."""

    def __init__(self, in_features):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Softplus(),
            nn.Linear(in_features // 2, 1),
        )

    def forward(self, atom_fea, crystal_atom_idx):
        scores = self.gate(atom_fea)
        pooled = []
        for idx_map in crystal_atom_idx:
            fea_i = atom_fea[idx_map]
            w_i = F.softmax(scores[idx_map], dim=0)
            pooled.append((w_i * fea_i).sum(dim=0, keepdim=True))
        return torch.cat(pooled, dim=0)


class CrystalGraphConvNetMTV3(nn.Module):
    """
    Multi-task CGCNN v3 with attention pooling and hierarchical heads.

    Heads:
      1. Band gap regression (1 output)
      2. Gap type classification (3 classes), conditioned on bandgap prediction
      3. Energy above hull regression (1 output)
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=4, h_fea_len=128, n_h=1,
                 n_gap_classes=3, dropout=0.15):
        super().__init__()

        # Shared backbone
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.attn_pool = AttentionPooling(atom_fea_len)
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.act = nn.Softplus()
        self.drop = nn.Dropout(p=dropout)

        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            ])
            self.fc_acts = nn.ModuleList([
                nn.Softplus() for _ in range(n_h - 1)
            ])

        # Head 1: Band gap regression
        self.head_bg = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, 1),
        )

        # Head 2: Gap type classification (conditioned on bandgap prediction)
        self.head_gt = nn.Sequential(
            nn.Linear(h_fea_len + 1, h_fea_len // 2),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, n_gap_classes),
            nn.LogSoftmax(dim=1),
        )

        # Head 3: Energy above hull regression
        self.head_eh = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, 1),
        )

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.embedding(atom_fea)
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.attn_pool(atom_fea, crystal_atom_idx)
        crys_fea = self.act(self.conv_to_fc(self.act(crys_fea)))
        crys_fea = self.drop(crys_fea)

        if hasattr(self, 'fcs'):
            for fc, act in zip(self.fcs, self.fc_acts):
                crys_fea = self.drop(act(fc(crys_fea)))

        bg = self.head_bg(crys_fea)
        # Hierarchical: gap type head receives bandgap prediction (detached)
        gt = self.head_gt(torch.cat([crys_fea, bg.detach()], dim=1))
        eh = self.head_eh(crys_fea)
        return bg, gt, eh
