"""
Multi-task CGCNN v7 — key changes over v6:

1. Wider/deeper backbone: 128 atom_fea, 256 h_fea, 6 conv, 8-head attention
2. Deeper 3-layer BG head with two skip connections
3. EH task: binary classification (stable EH<0.01 vs unstable) instead of regression
4. GT head: also conditioned on EH prediction
5. Dropout 0.3 (slightly reduced since model is wider and better regularised)
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.model import ConvLayer


class ResidualConvBlock(nn.Module):
    """ConvLayer with residual skip + LayerNorm (same as v6)."""
    def __init__(self, atom_fea_len, nbr_fea_len):
        super().__init__()
        self.conv = ConvLayer(atom_fea_len, nbr_fea_len)
        self.layer_norm = nn.LayerNorm(atom_fea_len)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx):
        out = self.conv(atom_fea, nbr_fea, nbr_fea_idx)
        return self.layer_norm(out + atom_fea)


class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling over atoms (same as v6, configurable heads)."""
    def __init__(self, in_features, n_heads=8):
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


class BandgapHeadV7(nn.Module):
    """3-layer BG head with two residual skip connections."""
    def __init__(self, h_fea_len, dropout=0.3):
        super().__init__()
        mid = h_fea_len // 2
        quarter = h_fea_len // 4
        self.fc1 = nn.Linear(h_fea_len, mid)
        self.bn1 = nn.BatchNorm1d(mid)
        self.fc2 = nn.Linear(mid, mid)
        self.bn2 = nn.BatchNorm1d(mid)
        self.fc3 = nn.Linear(mid, quarter)
        self.bn3 = nn.BatchNorm1d(quarter)
        self.fc_out = nn.Linear(quarter, 1)
        self.skip1 = nn.Linear(h_fea_len, mid)
        self.skip2 = nn.Linear(mid, quarter)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        skip1 = self.skip1(x)
        h = self.drop(self.act(self.bn1(self.fc1(x))))
        h = self.drop(self.act(self.bn2(self.fc2(h))))
        h = h + skip1
        skip2 = self.skip2(h)
        h = self.drop(self.act(self.bn3(self.fc3(h))))
        h = h + skip2
        return self.fc_out(h)


class CrystalGraphConvNetMTV7(nn.Module):
    """
    Multi-task CGCNN v7 — wider/deeper with EH binary classification.

    Heads:
      1. Band gap regression (1 output) — 3-layer residual head
      2. Gap type classification (n_gap_classes), conditioned on BG
      3. EH stability binary classification (stable vs unstable)
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=128, n_conv=6, h_fea_len=256, n_h=1,
                 n_gap_classes=2, n_eh_classes=2, dropout=0.3, n_attn_heads=8):
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

        # Head 1: Band gap — 3-layer residual head
        self.head_bg = BandgapHeadV7(h_fea_len, dropout=dropout)

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

        # Head 3: EH stability binary classification
        self.head_eh = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.BatchNorm1d(h_fea_len // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, h_fea_len // 4),
            nn.BatchNorm1d(h_fea_len // 4),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(h_fea_len // 4, n_eh_classes),
            nn.LogSoftmax(dim=1),
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
