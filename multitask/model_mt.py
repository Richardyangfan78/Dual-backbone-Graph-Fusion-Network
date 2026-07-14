"""
Multi-task CGCNN v2: larger shared backbone + task-specific heads +
learnable uncertainty-based loss weighting (Kendall et al. 2018).

Heads:
  1. Band gap regression (1 output)
  2. Gap type classification (3 classes: Direct/Indirect/Metal)
  3. Energy above hull regression (1 output)
"""
from __future__ import print_function, division

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cgcnn'))
from cgcnn.model import ConvLayer


class CrystalGraphConvNetMT(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=128, n_conv=5, h_fea_len=256, n_h=2,
                 n_gap_classes=3, dropout=0.2):
        super().__init__()
        self.n_gap_classes = n_gap_classes

        # ── Shared backbone ──
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
            for _ in range(n_conv)
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=dropout)

        # Shared hidden layers
        if n_h > 1:
            self.fcs = nn.ModuleList([
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            ])
            self.softpluses = nn.ModuleList([
                nn.Softplus() for _ in range(n_h - 1)
            ])

        # ── Head 1: Band gap regression ──
        self.head_bandgap = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, 1),
        )

        # ── Head 2: Gap type classification ──
        self.head_gaptype = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, h_fea_len // 4),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 4, n_gap_classes),
            nn.LogSoftmax(dim=1),
        )

        # ── Head 3: Energy above hull regression ──
        self.head_ehull = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.Softplus(),
            nn.Dropout(p=dropout),
            nn.Linear(h_fea_len // 2, 1),
        )

        # ── Learnable loss weights (uncertainty weighting) ──
        # log(sigma^2) for each task; loss_i = 1/(2*sigma_i^2) * L_i + log(sigma_i)
        self.log_vars = nn.Parameter(torch.zeros(3))

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        # Shared backbone
        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
                crys_fea = self.dropout(crys_fea)

        # 3 heads
        out_bandgap = self.head_bandgap(crys_fea)
        out_gaptype = self.head_gaptype(crys_fea)
        out_ehull = self.head_ehull(crys_fea)
        return out_bandgap, out_gaptype, out_ehull

    def compute_weighted_loss(self, loss_bg, loss_gt, loss_eh):
        """Uncertainty-based multi-task loss weighting (Kendall 2018).
        loss = sum_i [ 1/(2*exp(log_var_i)) * L_i + 0.5 * log_var_i ]
        """
        precision_bg = torch.exp(-self.log_vars[0])
        precision_gt = torch.exp(-self.log_vars[1])
        precision_eh = torch.exp(-self.log_vars[2])

        total = (0.5 * precision_bg * loss_bg + 0.5 * self.log_vars[0] +
                 0.5 * precision_gt * loss_gt + 0.5 * self.log_vars[1] +
                 0.5 * precision_eh * loss_eh + 0.5 * self.log_vars[2])
        return total

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) == \
               atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)
