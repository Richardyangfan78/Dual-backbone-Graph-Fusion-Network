"""
Dual-backbone MACE + ALIGNN fusion model for chalcohalide prediction.

Plan A architecture:
  - MACE backbone (256-dim equivariant per-atom features) → attention pooling
  - ALIGNN backbone (256-dim angular/topological features) → mean pooling
  - Gated fusion: learns to combine local (MACE) and global (ALIGNN) info
  - 3 task heads: BG (regression), GT (classification), EH (classification)

Key insight: MACE excels at BG (local bonding), ALIGNN excels at GT
(global topology). Fusion combines both strengths.
"""
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from typing import Dict

from model_mace_mt import (
    MACEFeatureExtractor, AttentionPooling, BandgapHead, CosineClassifier
)
from model_alignn_pyg import (
    ALIGNNMultiTaskPyG, EdgeGatedConv, ALIGNNLayer
)


class ALIGNNFeatureExtractor(nn.Module):
    """Extract per-crystal features from ALIGNN backbone (strip task heads).

    Runs: atom_proj → bond_proj → angle_proj → ALIGNN layers → GCN layers
          → global mean pooling → [B, hidden_dim]
    """

    def __init__(self, alignn_model: ALIGNNMultiTaskPyG):
        super().__init__()
        self.atom_proj = alignn_model.atom_proj
        self.bond_proj = alignn_model.bond_proj
        self.angle_proj = alignn_model.angle_proj
        self.alignn_layers = alignn_model.alignn_layers
        self.gcn_layers = alignn_model.gcn_layers
        self.drop = alignn_model.drop
        self.out_dim = 256  # hidden_dim

    def forward(self, x, edge_index, edge_attr,
                line_edge_index, line_attr, batch, num_graphs):
        """
        Args:
            x: [N, 94] atom one-hot features
            edge_index: [2, E] crystal graph edges
            edge_attr: [E, 80] RBF distance features
            line_edge_index: [2, T] line graph edges
            line_attr: [T, 40] RBF angle features
            batch: [N] graph assignment
            num_graphs: number of graphs
        Returns:
            crys_fea: [B, 256] crystal-level features
        """
        atom_x = self.atom_proj(x)
        bond_x = self.bond_proj(edge_attr)
        angle_x = self.angle_proj(line_attr)

        for layer in self.alignn_layers:
            atom_x, bond_x, angle_x = layer(
                atom_x, bond_x, angle_x, edge_index, line_edge_index
            )

        for layer in self.gcn_layers:
            atom_x, bond_x = layer(atom_x, edge_index, bond_x)

        # Global mean pooling
        h = scatter(atom_x, batch, dim=0, dim_size=num_graphs, reduce="mean")
        return self.drop(h)


class GatedFusion(nn.Module):
    """Gated fusion of two feature streams.

    gate = sigmoid(W_gate * [f_mace, f_alignn])
    fused = gate * f_mace + (1 - gate) * f_alignn
    Then project through a residual MLP.
    """

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

    def forward(self, f_mace, f_alignn):
        """
        Args:
            f_mace: [B, D] MACE crystal features
            f_alignn: [B, D] ALIGNN crystal features
        Returns:
            fused: [B, D] fused features
        """
        concat = torch.cat([f_mace, f_alignn], dim=1)  # [B, 2D]
        g = self.gate(concat)  # [B, D]
        gated = g * f_mace + (1 - g) * f_alignn  # [B, D]
        projected = self.proj(concat)  # [B, D]
        return gated + projected  # residual


class DualBackboneMultiTask(nn.Module):
    """
    MACE + ALIGNN dual backbone with gated fusion + multi-task heads.

    Architecture:
      MACE backbone → attn pooling → [B, 256]
      ALIGNN backbone → mean pooling → [B, 256]
      → Gated fusion → [B, 256]
      → Trunk → 3 task heads
    """

    def __init__(self, mace_model, alignn_model,
                 h_fea_len=256, n_gap_classes=2, n_eh_classes=2,
                 dropout=0.3, n_attn_heads=8,
                 use_cosine_classifier=True, cosine_temp=0.1):
        super().__init__()
        self.n_eh_classes = n_eh_classes
        self.register_buffer("cond_weight", torch.tensor(0.0))

        # MACE backbone + attention pooling
        self.mace_backbone = MACEFeatureExtractor(mace_model)
        mace_dim = self.mace_backbone.out_dim  # 256
        self.mace_pool = AttentionPooling(mace_dim, n_attn_heads)

        # ALIGNN backbone (extracts features, strips task heads)
        self.alignn_backbone = ALIGNNFeatureExtractor(alignn_model)
        alignn_dim = self.alignn_backbone.out_dim  # 256

        assert mace_dim == alignn_dim, \
            f"Feature dim mismatch: MACE={mace_dim}, ALIGNN={alignn_dim}"

        # Gated fusion
        self.fusion = GatedFusion(mace_dim, dropout=dropout * 0.5)

        # Trunk: project fused features
        self.proj = nn.Linear(mace_dim, h_fea_len)
        self.trunk_bn = nn.BatchNorm1d(h_fea_len)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # Head 1: Bandgap regression
        self.head_bg = BandgapHead(h_fea_len, dropout=dropout)

        # Head 3: EH stability (binary)
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

        # Head 2: Gap type (conditioned on BG + EH with annealing)
        self.gt_proj = nn.Linear(h_fea_len + 1 + n_eh_classes, h_fea_len)
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

    def set_cond_weight(self, w: float):
        self.cond_weight.fill_(min(max(w, 0.0), 1.0))

    def forward(self, data):
        """
        Args:
            data: DualGraphData batch with both MACE and ALIGNN fields
        Returns:
            bg, gt, eh predictions
        """
        batch_idx = data.batch
        num_graphs = int(batch_idx.max().item()) + 1

        # ── MACE branch ──
        # Build dict manually to avoid PyG Data.node_attrs() method collision
        mace_dict = {
            "positions": data.positions,
            "node_attrs": data.mace_node_attrs,  # renamed in DualGraphData
            "edge_index": data.edge_index,
            "shifts": data.shifts,
            "unit_shifts": data.unit_shifts,
            "cell": data.cell,
            "batch": data.batch,
            "ptr": data.ptr,  # needed by MACE prepare_graph
        }
        if hasattr(data, "pbc"):
            mace_dict["pbc"] = data.pbc
        if hasattr(data, "head"):
            mace_dict["head"] = data.head
        node_feats_mace = self.mace_backbone(mace_dict)  # [N, 256]
        f_mace = self.mace_pool(node_feats_mace, batch_idx, num_graphs)

        # ── ALIGNN branch ──
        f_alignn = self.alignn_backbone(
            x=data.alignn_x,
            edge_index=data.alignn_edge_index,
            edge_attr=data.alignn_edge_attr,
            line_edge_index=data.alignn_line_edge_index,
            line_attr=data.alignn_line_attr,
            batch=batch_idx,
            num_graphs=num_graphs,
        )

        # ── Fusion ──
        fused = self.fusion(f_mace, f_alignn)  # [B, 256]

        # ── Trunk ──
        crys_fea = self.drop(self.act(self.trunk_bn(self.proj(fused))))

        # ── Task heads ──
        bg = self.head_bg(crys_fea)
        eh = self.head_eh(crys_fea)
        eh_probs = eh.exp()

        # GT conditioned on BG + EH with annealing
        cw = self.cond_weight.item()
        bg_cond = bg.detach() * cw
        eh_cond = eh_probs.detach() * cw
        gt_in = self.gt_proj(torch.cat([crys_fea, bg_cond, eh_cond], dim=1))
        gt_hidden = self.gt_layers(gt_in)
        gt = self.gt_classifier(gt_hidden)

        return bg, gt, eh

    # ─── Parameter groups for differential learning rates ───

    def get_mace_backbone_params(self):
        return list(self.mace_backbone.parameters())

    def get_alignn_backbone_params(self):
        return list(self.alignn_backbone.parameters())

    def get_head_params(self):
        """All non-backbone parameters (fusion + trunk + heads)."""
        mace_ids = {id(p) for p in self.mace_backbone.parameters()}
        alignn_ids = {id(p) for p in self.alignn_backbone.parameters()}
        backbone_ids = mace_ids | alignn_ids
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def freeze_mace_backbone(self):
        for p in self.mace_backbone.parameters():
            p.requires_grad = False

    def freeze_alignn_backbone(self):
        for p in self.alignn_backbone.parameters():
            p.requires_grad = False

    def unfreeze_mace_backbone(self, last_n_layers=None):
        if last_n_layers is None:
            for p in self.mace_backbone.parameters():
                p.requires_grad = True
        else:
            n = len(self.mace_backbone.interactions)
            for block in list(self.mace_backbone.interactions)[max(0, n - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
            for block in list(self.mace_backbone.products)[max(0, n - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True

    def unfreeze_alignn_backbone(self, last_n_layers=None):
        if last_n_layers is None:
            for p in self.alignn_backbone.parameters():
                p.requires_grad = True
        else:
            # Unfreeze last N GCN layers + last N ALIGNN layers
            n_gcn = len(self.alignn_backbone.gcn_layers)
            for block in list(self.alignn_backbone.gcn_layers)[max(0, n_gcn - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
            n_aln = len(self.alignn_backbone.alignn_layers)
            for block in list(self.alignn_backbone.alignn_layers)[max(0, n_aln - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
