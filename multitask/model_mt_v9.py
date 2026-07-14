"""
Multi-task CGCNN v9 — GT head conditioned on BG prediction + EH class probabilities.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn

from model_mt_v7 import (
    BandgapHeadV7,
    MultiHeadAttentionPooling,
    ResidualConvBlock,
)


class CrystalGraphConvNetMTV9(nn.Module):
    """
    Like MTV7, but gap-type head sees crystal embedding + BG + EH probabilities.
    """

    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=128, n_conv=6, h_fea_len=256, n_h=1,
                 n_gap_classes=2, n_eh_classes=2, dropout=0.3, n_attn_heads=8):
        super().__init__()
        self.n_eh_classes = n_eh_classes

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

        self.gt_proj = nn.Linear(h_fea_len + 1 + n_eh_classes, h_fea_len)
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
        gt = self.head_gt(gt_in)
        return bg, gt, eh
