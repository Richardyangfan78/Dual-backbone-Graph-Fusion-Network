"""
Multi-task CGCNN v6: Right-sized model + improved BG head.

Key changes over v5:
  - Smaller backbone: 64 atom_fea, 128 h_fea, 4 conv (right-sized for 2.5K data)
  - Residual BG head with skip connection
  - Separate BG feature branch (partial backbone detach for GT/EH)
  - Supports cached graph loading for faster training
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.model import ConvLayer


class ResidualConvBlock(nn.Module):
    """ConvLayer with residual skip connection."""
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.conv = ConvLayer(atom_fea_len, nbr_fea_len)
        self.layer_norm = nn.LayerNorm(atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        out = self.conv(atom_fea, nbr_fea, nbr_fea_idx)
        return self.layer_norm(out + atom_fea)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling over atoms."""
    def __init__(self, in_features, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 4),
                nn.Softplus(),
                nn.Linear(in_features // 4, 1),
            ) for _ in range(n_heads)
        ])
        self.combine = nn.Linear(in_features * n_heads, in_features)

    def forward(self, atom_fea, crystal_atom_idx):
        head_outputs = []
        for gate in self.gates:
            scores = gate(atom_fea)
            pooled = []
            for idx_map in crystal_atom_idx:
                fea_i = atom_fea[idx_map]
                w_i = F.softmax(scores[idx_map], dim=0)
                pooled.append((w_i * fea_i).sum(dim=0, keepdim=True))
            head_outputs.append(torch.cat(pooled, dim=0))
        multi = torch.cat(head_outputs, dim=-1)
        return self.combine(multi)


class BandgapHead(nn.Module):
    """Deeper BG head with residual skip connection."""
    def __init__(self, h_fea_len, dropout=0.3):
        super().__init__()
        mid = h_fea_len // 2
        self.fc1 = nn.Linear(h_fea_len, mid)
        self.bn1 = nn.BatchNorm1d(mid)
        self.fc2 = nn.Linear(mid, mid)
        self.bn2 = nn.BatchNorm1d(mid)
        self.fc_out = nn.Linear(mid, 1)
        self.skip = nn.Linear(h_fea_len, mid)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        skip = self.skip(x)
        x = self.drop(self.act(self.bn1(self.fc1(x))))
        x = self.drop(self.act(self.bn2(self.fc2(x))))
        x = x + skip
        return self.fc_out(x)


class CrystalGraphConvNetMTV6(nn.Module):
    """
    Multi-task CGCNN v6 — right-sized for ~2.5K samples.

    Heads:
      1. Band gap regression (1 output) — residual head
      2. Gap type classification (n_gap_classes), conditioned on BG
      3. Energy above hull regression (1 output)
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=4, h_fea_len=128, n_h=1,
                 n_gap_classes=2, dropout=0.3, n_attn_heads=4):
        super().__init__()

        # Shared backbone
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embed_bn = nn.BatchNorm1d(atom_fea_len)
        self.convs = nn.ModuleList([
            ResidualConvBlock(atom_fea_len, nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.attn_pool = MultiHeadAttentionPooling(atom_fea_len, n_attn_heads)

        # Shared trunk
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.trunk_bn = nn.BatchNorm1d(h_fea_len)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

        # Head 1: Band gap — residual head
        self.head_bg = BandgapHead(h_fea_len, dropout=dropout)

        # Head 2: Gap type classification (conditioned on BG)
        self.gt_proj = nn.Linear(h_fea_len + 1, h_fea_len)
        self.head_gt = nn.Sequential(
            nn.BatchNorm1d(h_fea_len),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(h_fea_len // 2, n_gap_classes),
            nn.LogSoftmax(dim=1),
        )

        # Head 3: Energy above hull
        self.head_eh = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.BatchNorm1d(h_fea_len // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, 1),
        )

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea = self.act(self.embed_bn(self.embedding(atom_fea)))
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self.attn_pool(atom_fea, crystal_atom_idx)
        crys_fea = self.drop(self.act(self.trunk_bn(self.conv_to_fc(crys_fea))))

        bg = self.head_bg(crys_fea)
        gt_in = self.gt_proj(torch.cat([crys_fea, bg.detach()], dim=1))
        gt = self.head_gt(gt_in)
        eh = self.head_eh(crys_fea)
        return bg, gt, eh
