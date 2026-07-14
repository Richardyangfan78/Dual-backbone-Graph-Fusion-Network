"""
MACE + M3GNet dual-backbone with gated fusion — ablation counterpart to
MACE + ALIGNN (model_dual_backbone.py).

Key difference vs ALIGNN variant:
  - M3GNet hidden_dim=128 → projection to 256-dim before fusion
  - M3GNet uses 3-body gating (simpler than ALIGNN edge-gated line graph)
  - Same crystal_cached_graphs data pipeline as ALIGNN (drop-in replacement)

Purpose: ablation study to show ALIGNN is a better fusion partner than M3GNet
for MACE, justifying the MACE+ALIGNN design choice.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import Dict

from model_mace_mt import (
    MACEFeatureExtractor, AttentionPooling,
    BandgapHead, CosineClassifier,
)
from model_m3gnet_pyg import M3GNetMultiTaskPyG


class M3GNetFeatureExtractor(nn.Module):
    """
    Strip task heads from M3GNet; return [B, out_dim] crystal features.

    M3GNet internal dim is 128.  We project to out_dim=256 to match MACE,
    so GatedFusion receives equal-dimension inputs from both backbones.
    """

    def __init__(self, m3gnet_model: M3GNetMultiTaskPyG, out_dim: int = 256):
        super().__init__()
        self.atom_proj  = m3gnet_model.atom_proj
        self.bond_proj  = m3gnet_model.bond_proj
        self.angle_proj = m3gnet_model.angle_proj
        self.blocks     = m3gnet_model.blocks
        self.drop       = m3gnet_model.drop
        self._hidden    = 128          # M3GNet default
        self.out_dim    = out_dim
        # Project 128 → 256 to match MACE feature dimension
        self.expand = nn.Sequential(
            nn.Linear(self._hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x, edge_index, edge_attr,
                line_edge_index, line_attr, batch, num_graphs):
        """
        Same signature as ALIGNNFeatureExtractor — feeds from DualGraphData
        alignn_* fields (M3GNet reuses the same line-graph data pipeline).
        """
        atom_x  = self.atom_proj(x)
        bond_x  = self.bond_proj(edge_attr)
        if line_attr.size(0) > 0:
            angle_x = self.angle_proj(line_attr)
        else:
            angle_x = torch.zeros(0, atom_x.size(1), device=atom_x.device)

        for block in self.blocks:
            atom_x, bond_x, angle_x = block(
                atom_x, bond_x, angle_x, edge_index, line_edge_index
            )

        h = scatter(atom_x, batch, dim=0, dim_size=num_graphs, reduce="mean")
        h = self.drop(h)
        return self.expand(h)          # [B, out_dim]


class GatedFusion(nn.Module):
    """Identical gated fusion as in model_dual_backbone.py."""

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Linear(dim * 2, dim)

    def forward(self, f_mace, f_m3gnet):
        concat = torch.cat([f_mace, f_m3gnet], dim=1)
        g      = self.gate(concat)
        gated  = g * f_mace + (1 - g) * f_m3gnet
        return gated + self.proj(concat)


class DualBackboneMACEM3GNet(nn.Module):
    """
    MACE + M3GNet dual backbone with gated fusion + multi-task heads.

    Architecture mirrors DualBackboneMultiTask (MACE+ALIGNN) exactly,
    with M3GNet replacing ALIGNN as the angular backbone.
    """

    def __init__(self, mace_model, m3gnet_model,
                 h_fea_len=256, n_gap_classes=2, n_eh_classes=2,
                 dropout=0.3, n_attn_heads=8,
                 use_cosine_classifier=True, cosine_temp=0.1):
        super().__init__()
        self.n_eh_classes = n_eh_classes
        self.register_buffer("cond_weight", torch.tensor(0.0))

        # MACE backbone + attention pooling
        self.mace_backbone = MACEFeatureExtractor(mace_model)
        mace_dim = self.mace_backbone.out_dim          # 256
        self.mace_pool = AttentionPooling(mace_dim, n_attn_heads)

        # M3GNet backbone (projects 64 → 256 internally)
        self.m3gnet_backbone = M3GNetFeatureExtractor(m3gnet_model, out_dim=mace_dim)
        assert mace_dim == self.m3gnet_backbone.out_dim

        # Gated fusion (identical structure to MACE+ALIGNN version)
        self.fusion = GatedFusion(mace_dim, dropout=dropout * 0.5)

        # Trunk
        self.proj      = nn.Linear(mace_dim, h_fea_len)
        self.trunk_bn  = nn.BatchNorm1d(h_fea_len)
        self.act       = nn.SiLU()
        self.drop      = nn.Dropout(dropout)

        # Head 1: Bandgap regression
        self.head_bg = BandgapHead(h_fea_len, dropout=dropout)

        # Head 2: EH stability (binary)
        self.head_eh = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len // 2),
            nn.BatchNorm1d(h_fea_len // 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(h_fea_len // 2, h_fea_len // 4),
            nn.BatchNorm1d(h_fea_len // 4), nn.SiLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(h_fea_len // 4, n_eh_classes),
            nn.LogSoftmax(dim=1),
        )

        # Head 3: Gap type (conditioned on BG + EH with annealing)
        self.gt_proj = nn.Linear(h_fea_len + 1 + n_eh_classes, h_fea_len)
        gt_hidden = h_fea_len // 2
        self.gt_layers = nn.Sequential(
            nn.BatchNorm1d(h_fea_len), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(h_fea_len, gt_hidden), nn.SiLU(), nn.Dropout(dropout * 0.5),
        )
        if use_cosine_classifier:
            self.gt_classifier = CosineClassifier(gt_hidden, n_gap_classes,
                                                  init_temp=cosine_temp)
        else:
            self.gt_classifier = nn.Sequential(
                nn.Linear(gt_hidden, n_gap_classes), nn.LogSoftmax(dim=1),
            )

    def set_cond_weight(self, w: float):
        self.cond_weight.fill_(min(max(w, 0.0), 1.0))

    def forward(self, data):
        batch_idx  = data.batch
        num_graphs = int(batch_idx.max().item()) + 1

        # ── MACE branch ──────────────────────────────────────────────
        mace_dict = {
            "positions":   data.positions,
            "node_attrs":  data.mace_node_attrs,
            "edge_index":  data.edge_index,
            "shifts":      data.shifts,
            "unit_shifts": data.unit_shifts,
            "cell":        data.cell,
            "batch":       data.batch,
            "ptr":         data.ptr,
        }
        if hasattr(data, "pbc"):
            mace_dict["pbc"] = data.pbc
        if hasattr(data, "head"):
            mace_dict["head"] = data.head
        node_feats_mace = self.mace_backbone(mace_dict)       # [N, 256]
        f_mace = self.mace_pool(node_feats_mace, batch_idx, num_graphs)

        # ── M3GNet branch (reuses alignn_* fields — same line graph) ─
        f_m3gnet = self.m3gnet_backbone(
            x              = data.alignn_x,
            edge_index     = data.alignn_edge_index,
            edge_attr      = data.alignn_edge_attr,
            line_edge_index= data.alignn_line_edge_index,
            line_attr      = data.alignn_line_attr,
            batch          = batch_idx,
            num_graphs     = num_graphs,
        )

        # ── Fusion ───────────────────────────────────────────────────
        fused = self.fusion(f_mace, f_m3gnet)

        # ── Trunk ────────────────────────────────────────────────────
        crys_fea = self.drop(self.act(self.trunk_bn(self.proj(fused))))

        # ── Task heads ───────────────────────────────────────────────
        bg = self.head_bg(crys_fea)
        eh = self.head_eh(crys_fea)
        eh_probs = eh.exp()

        cw = self.cond_weight.item()
        bg_cond = bg.detach() * cw
        eh_cond = eh_probs.detach() * cw
        gt_in   = self.gt_proj(torch.cat([crys_fea, bg_cond, eh_cond], dim=1))
        gt      = self.gt_classifier(self.gt_layers(gt_in))

        return bg, gt, eh

    # ── Parameter group helpers (mirrors DualBackboneMultiTask) ──────

    def get_mace_backbone_params(self):
        return list(self.mace_backbone.parameters())

    def get_m3gnet_backbone_params(self):
        return list(self.m3gnet_backbone.parameters())

    def get_head_params(self):
        mace_ids   = {id(p) for p in self.mace_backbone.parameters()}
        m3g_ids    = {id(p) for p in self.m3gnet_backbone.parameters()}
        exclude    = mace_ids | m3g_ids
        return [p for p in self.parameters() if id(p) not in exclude]

    def freeze_mace_backbone(self):
        for p in self.mace_backbone.parameters():
            p.requires_grad = False

    def freeze_m3gnet_backbone(self):
        for p in self.m3gnet_backbone.parameters():
            p.requires_grad = False

    def unfreeze_mace_backbone(self, last_n_layers=None):
        if last_n_layers is None:
            for p in self.mace_backbone.parameters():
                p.requires_grad = True
        else:
            n = len(self.mace_backbone.interactions)
            for blk in list(self.mace_backbone.interactions)[max(0, n-last_n_layers):]:
                for p in blk.parameters(): p.requires_grad = True
            for blk in list(self.mace_backbone.products)[max(0, n-last_n_layers):]:
                for p in blk.parameters(): p.requires_grad = True

    def unfreeze_m3gnet_backbone(self, last_n_layers=None):
        if last_n_layers is None:
            for p in self.m3gnet_backbone.parameters():
                p.requires_grad = True
        else:
            n = len(self.m3gnet_backbone.blocks)
            for blk in list(self.m3gnet_backbone.blocks)[max(0, n-last_n_layers):]:
                for p in blk.parameters(): p.requires_grad = True
