"""
Multi-task CGCNN v10 — GT head with CosineClassifier, increased capacity.
Changes from v9:
1. CosineClassifier replaces LogSoftmax in GT head for better class-imbalance handling
2. Increased capacity: atom_fea_len=160, h_fea_len=320, n_conv=8 (configurable)
3. Learnable temperature in CosineClassifier (init=0.1, safer than fixed 0.05)
"""
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_mt_v7 import (
    BandgapHeadV7,
    MultiHeadAttentionPooling,
    ResidualConvBlock,
)


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature scaling.
    Normalizes both features and class weights to reduce bias toward majority class."""
    def __init__(self, in_features, n_classes, init_temp=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        # Learnable log-temperature: exp(log_temp) = temp
        # Clamped to [0.01, 1.0] during forward to prevent instability
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        temp = self.log_temp.exp().clamp(min=0.01, max=1.0)
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return F.log_softmax(x_norm @ w_norm.T / temp, dim=1)


class CrystalGraphConvNetMTV10(nn.Module):
    """
    Multi-task CGCNN v10 — v9 with CosineClassifier for GT head.

    Changes from v9:
    - GT head uses CosineClassifier instead of LogSoftmax (reduces class imbalance bias)
    - Default capacity increased: atom_fea_len=160, h_fea_len=320, n_conv=8
    - Learnable temperature in CosineClassifier
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=160, n_conv=8, h_fea_len=320, n_h=1,
                 n_gap_classes=2, n_eh_classes=2, dropout=0.35, n_attn_heads=8,
                 use_cosine_classifier=True, cosine_temp=0.1):
        super().__init__()
        self.n_eh_classes = n_eh_classes
        self.use_cosine_classifier = use_cosine_classifier

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.embed_bn = nn.BatchNorm1d(atom_fea_len)
        self.convs = nn.ModuleList([
            ResidualConvBlock(atom_fea_len, nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.attn_pool = MultiHeadAttentionPooling(atom_fea_len, n_attn_heads)

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.trunk_bn = nn.BatchNorm1d(h_fea_len)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=dropout)

        self.head_bg = BandgapHeadV7(h_fea_len, dropout=dropout)

        # GT projection: crys_fea + bg + eh_probs -> h_fea_len
        self.gt_proj = nn.Linear(h_fea_len + 1 + n_eh_classes, h_fea_len)

        # GT head with either CosineClassifier or standard LogSoftmax
        gt_hidden = h_fea_len // 2
        self.gt_layers = nn.Sequential(
            nn.BatchNorm1d(h_fea_len),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len, gt_hidden),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
        )

        if use_cosine_classifier:
            self.gt_classifier = CosineClassifier(gt_hidden, n_gap_classes,
                                                  init_temp=cosine_temp)
        else:
            self.gt_classifier = nn.Sequential(
                nn.Linear(gt_hidden, n_gap_classes),
                nn.LogSoftmax(dim=1),
            )

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
        eh = self.head_eh(crys_fea)
        eh_probs = eh.exp()

        gt_in = self.gt_proj(torch.cat(
            [crys_fea, bg.detach(), eh_probs.detach()], dim=1))
        gt_hidden = self.gt_layers(gt_in)
        gt = self.gt_classifier(gt_hidden)

        return bg, gt, eh

    def get_shared_params(self):
        """Return parameters of the shared backbone (for GradNorm).
        Cached as list to avoid repeated collection."""
        params = []
        params.extend(self.embedding.parameters())
        params.extend(self.embed_bn.parameters())
        for conv in self.convs:
            params.extend(conv.parameters())
        params.extend(self.attn_pool.parameters())
        params.extend(self.conv_to_fc.parameters())
        params.extend(self.trunk_bn.parameters())
        return params
