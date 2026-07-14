"""
Multi-task ALIGNN model.

Uses ALIGNN backbone (graph conv + line graph conv for bond angles)
with 3 task-specific heads:
  1. Band gap regression
  2. Gap type classification (binary: Direct vs Indirect+Metal)
  3. Energy above hull regression
"""
from typing import Tuple, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling

from alignn.models.alignn import (
    ALIGNNConfig,
    ALIGNNConv,
    EdgeGatedGraphConv,
    MLPLayer,
)
from alignn.models.utils import RBFExpansion


class ALIGNNMultiTask(nn.Module):
    """ALIGNN backbone with 3 multi-task heads."""

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn"),
                 n_gap_classes=2, dropout=0.3):
        super().__init__()
        self.config = config
        hf = config.hidden_features

        # --- Shared backbone (same as ALIGNN) ---
        self.atom_embedding = MLPLayer(
            config.atom_input_features, hf)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, hf),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, hf),
        )
        self.alignn_layers = nn.ModuleList([
            ALIGNNConv(hf, hf) for _ in range(config.alignn_layers)
        ])
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConv(hf, hf) for _ in range(config.gcn_layers)
        ])
        self.readout = AvgPooling()

        # --- Task heads ---
        self.drop = nn.Dropout(p=dropout)

        # Head 1: Band gap regression (deeper with skip)
        mid = hf // 2
        self.bg_fc1 = nn.Linear(hf, mid)
        self.bg_bn1 = nn.BatchNorm1d(mid)
        self.bg_fc2 = nn.Linear(mid, mid)
        self.bg_bn2 = nn.BatchNorm1d(mid)
        self.bg_skip = nn.Linear(hf, mid)
        self.bg_out = nn.Linear(mid, 1)

        # Head 2: Gap type classification (conditioned on BG prediction)
        self.gt_proj = nn.Linear(hf + 1, hf)
        self.gt_head = nn.Sequential(
            nn.BatchNorm1d(hf),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hf, mid),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(mid, n_gap_classes),
            nn.LogSoftmax(dim=1),
        )

        # Head 3: Energy above hull regression
        self.eh_head = nn.Sequential(
            nn.Linear(hf, mid),
            nn.BatchNorm1d(mid),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mid, 1),
        )

    def _shared_forward(self, g, lg):
        """Run shared ALIGNN backbone, return crystal-level features."""
        lg = lg.local_var()
        z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        h = self.readout(g, x)
        return h

    def forward(
        self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph]
    ):
        g, lg, lat = g
        h = self._shared_forward(g, lg)
        h = self.drop(h)

        # BG head with residual
        skip = self.bg_skip(h)
        bg = self.drop(F.silu(self.bg_bn1(self.bg_fc1(h))))
        bg = self.drop(F.silu(self.bg_bn2(self.bg_fc2(bg))))
        bg = bg + skip
        bg = self.bg_out(bg)

        # GT head conditioned on BG
        gt_in = self.gt_proj(torch.cat([h, bg.detach()], dim=1))
        gt = self.gt_head(gt_in)

        # EH head
        eh = self.eh_head(h)

        return bg.squeeze(-1), gt, eh.squeeze(-1)
