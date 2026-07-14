"""
MACE backbone + Graph Transformer for long-range crystal property prediction.

Plan C architecture:
  - Pre-trained MACE-MP-0 backbone (equivariant per-atom features, 256-dim)
  - Element-aware positional encoding (from node_attrs)
  - 3 Graph Transformer layers with self-attention across all atoms
  - Virtual [CLS] token for crystal-level representation
  - 3 task heads: BG (regression), GT (classification), EH (classification)

Key advantage over plain MACE:
  - Self-attention captures long-range atom-atom interactions
  - Virtual global node aggregates crystal-wide information
  - Better for global properties like gap type (direct/indirect)
"""
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from model_mace_mt import MACEFeatureExtractor, BandgapHead, CosineClassifier


class GraphTransformerLayer(nn.Module):
    """Pre-norm Transformer layer with self-attention for graph nodes."""

    def __init__(self, d_model, n_heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [B, S, d_model] padded sequence (CLS + atoms + padding)
            key_padding_mask: [B, S] True = masked (padding positions)
        """
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        h = self.norm2(x)
        x = x + self.ff(h)
        return x


class MACEGraphTransformer(nn.Module):
    """
    MACE backbone + Graph Transformer for global context.

    Architecture:
      MACE backbone → per-atom features [N, 256]
      → Pad to [B, max_atoms+1, 256] with virtual [CLS] token
      → 3 Graph Transformer layers (self-attention across all atoms + CLS)
      → Extract [CLS] token → [B, 256] crystal representation
      → Trunk → 3 task heads (BG, GT, EH)
    """

    def __init__(self, mace_model, h_fea_len=256, n_gap_classes=2,
                 n_eh_classes=2, dropout=0.3,
                 n_transformer_layers=3, n_transformer_heads=4,
                 d_ff=512, transformer_dropout=0.15,
                 use_cosine_classifier=True, cosine_temp=0.1):
        super().__init__()
        self.n_eh_classes = n_eh_classes
        self.register_buffer("cond_weight", torch.tensor(0.0))

        # MACE backbone
        self.backbone = MACEFeatureExtractor(mace_model)
        mace_dim = self.backbone.out_dim  # 256

        # Virtual global node (CLS token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, mace_dim) * 0.02)

        # Element-aware positional encoding from MACE node_attrs
        self.pos_proj = nn.Sequential(
            nn.Linear(89, mace_dim),  # node_attrs dim for MACE-MP-0 small
            nn.SiLU(),
            nn.Dropout(transformer_dropout),
        )

        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(mace_dim, n_transformer_heads, d_ff,
                                  transformer_dropout)
            for _ in range(n_transformer_layers)
        ])

        # Project to hidden dim
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

    def _pad_and_transform(self, node_feats, node_attrs, batch_idx, num_graphs):
        """Pad per-atom features to [B, max_atoms+1, D] and run transformer."""
        device = node_feats.device
        d = node_feats.size(-1)
        counts = torch.bincount(batch_idx, minlength=num_graphs)
        max_atoms = counts.max().item()
        S = max_atoms + 1  # +1 for CLS token

        # Create padded tensor and mask
        seq = torch.zeros(num_graphs, S, d, device=device)
        mask = torch.ones(num_graphs, S, dtype=torch.bool, device=device)

        # Insert CLS token at position 0
        seq[:, 0, :] = self.cls_token.squeeze(1).expand(num_graphs, -1)
        mask[:, 0] = False

        # Add element-aware positional encoding to atom features
        pos_enc = self.pos_proj(node_attrs)
        node_feats = node_feats + pos_enc

        # Compute local atom indices and place atoms in padded sequence
        cum = torch.zeros(num_graphs + 1, device=device, dtype=torch.long)
        cum[1:] = counts.cumsum(0)
        local_idx = torch.arange(len(batch_idx), device=device) - cum[batch_idx]
        flat_idx = batch_idx * S + 1 + local_idx  # +1 for CLS offset

        seq_flat = seq.reshape(-1, d)
        seq_flat[flat_idx] = node_feats
        seq = seq_flat.reshape(num_graphs, S, d)

        mask_flat = mask.reshape(-1)
        mask_flat[flat_idx] = False
        mask = mask_flat.reshape(num_graphs, S)

        # Run transformer layers
        for layer in self.transformer_layers:
            seq = layer(seq, key_padding_mask=mask)

        # Extract CLS token as crystal representation
        return seq[:, 0, :]  # [B, D]

    def forward(self, data: Dict[str, torch.Tensor]):
        batch_idx = data["batch"]
        num_graphs = int(batch_idx.max().item()) + 1

        # MACE backbone → per-atom features
        node_feats = self.backbone(data)  # [N_total, 256]

        # Graph Transformer with CLS token
        crys_fea = self._pad_and_transform(
            node_feats, data["node_attrs"], batch_idx, num_graphs
        )

        # Trunk
        crys_fea = self.drop(self.act(self.trunk_bn(self.proj(crys_fea))))

        # Task heads
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

    def get_backbone_params(self):
        return list(self.backbone.parameters())

    def get_head_params(self):
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, last_n_layers=None):
        if last_n_layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            n = len(self.backbone.interactions)
            for block in list(self.backbone.interactions)[max(0, n - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
            for block in list(self.backbone.products)[max(0, n - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
