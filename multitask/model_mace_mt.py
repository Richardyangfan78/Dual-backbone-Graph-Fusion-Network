"""
MACE-MP-0 Multi-task model for chalcohalide property prediction.

Architecture:
  - Pre-trained MACE-MP-0 backbone (equivariant message passing)
  - Multi-head attention pooling (atom → crystal)
  - Three task heads: bandgap (regression), gap type (classification), E_hull (classification)

Key advantages over CGCNN:
  - Equivariant representations capture directional bonding
  - Pre-trained on ~150k Materials Project structures
  - 128-dim per-atom features per interaction layer (256-dim concatenated)
"""
from __future__ import print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class MACEFeatureExtractor(nn.Module):
    """Extracts per-atom features from a pre-trained MACE model.

    Runs the MACE backbone (embedding → interactions → products) and returns
    concatenated node features from all interaction layers, bypassing the
    energy readout and force computation.
    """
    def __init__(self, mace_model):
        super().__init__()
        # Copy MACE components (shared references, not copies)
        self.node_embedding = mace_model.node_embedding
        self.radial_embedding = mace_model.radial_embedding
        self.spherical_harmonics = mace_model.spherical_harmonics
        self.interactions = mace_model.interactions
        self.products = mace_model.products
        self.atomic_numbers = mace_model.atomic_numbers

        # Feature dimension: 128 per interaction layer
        self.n_interactions = len(self.interactions)
        self.per_layer_dim = 128  # MACE-MP-0 small
        self.out_dim = self.per_layer_dim * self.n_interactions  # 256 for 2 layers

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            data: dict with keys from MACE's prepare_graph:
                  node_attrs, positions, edge_index, shifts, unit_shifts, cell, batch
        Returns:
            node_feats: [N_atoms, out_dim] concatenated features from all layers
        """
        from mace.modules.utils import prepare_graph

        ctx = prepare_graph(data)
        vectors = ctx.vectors
        lengths = ctx.lengths

        # Node embedding
        node_feats = self.node_embedding(data["node_attrs"])

        # Edge features
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(
            lengths, data["node_attrs"], data["edge_index"], self.atomic_numbers
        )

        # Interaction blocks
        node_feats_list: List[torch.Tensor] = []
        for i, (interaction, product) in enumerate(
            zip(self.interactions, self.products)
        ):
            interaction_out = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
                first_layer=(i == 0),
            )
            node_feats, sc = interaction_out[0], interaction_out[1]
            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)

        # Concatenate all interaction outputs
        return torch.cat(node_feats_list, dim=-1)  # [N_atoms, 256]


class AttentionPooling(nn.Module):
    """Multi-head attention pooling: atom features → crystal features.

    Uses PyG batch indices instead of CGCNN's crystal_atom_idx list.
    """
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

    def forward(self, atom_fea, batch, num_graphs):
        """
        Args:
            atom_fea: [N_atoms, in_features]
            batch: [N_atoms] graph index for each atom
            num_graphs: number of graphs in batch
        Returns:
            crys_fea: [num_graphs, in_features]
        """
        head_outputs = []
        for gate in self.gates:
            scores = gate(atom_fea)  # [N_atoms, 1]
            pooled = []
            for g in range(num_graphs):
                mask = (batch == g)
                fea_g = atom_fea[mask]
                w_g = F.softmax(scores[mask], dim=0)
                pooled.append((w_g * fea_g).sum(dim=0, keepdim=True))
            head_outputs.append(torch.cat(pooled, dim=0))
        multi = torch.cat(head_outputs, dim=-1)
        return self.combine(multi)


class BandgapHead(nn.Module):
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


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with learnable temperature."""
    def __init__(self, in_features, n_classes, init_temp=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp)))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        temp = self.log_temp.exp().clamp(min=0.01, max=1.0)
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return F.log_softmax(x_norm @ w_norm.T / temp, dim=1)


class MACEMultiTask(nn.Module):
    """
    MACE-MP-0 backbone + multi-task heads for chalcohalide prediction.

    Architecture:
      MACE backbone (256-dim per-atom) → attention pooling → trunk →
        ├── BG head (regression)
        ├── EH head (binary classification)
        └── GT head (classification, conditioned on BG + EH with annealing)
    """
    def __init__(self, mace_model,
                 h_fea_len=256, n_gap_classes=2, n_eh_classes=2,
                 dropout=0.3, n_attn_heads=8,
                 use_cosine_classifier=True, cosine_temp=0.1):
        super().__init__()
        self.n_eh_classes = n_eh_classes

        # Conditioning annealing weight: 0→1 over training
        self.register_buffer("cond_weight", torch.tensor(0.0))

        # MACE backbone
        self.backbone = MACEFeatureExtractor(mace_model)
        mace_dim = self.backbone.out_dim  # 256

        # Attention pooling
        self.attn_pool = AttentionPooling(mace_dim, n_attn_heads)

        # Trunk: project to h_fea_len
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

        # Head 2: Gap type (conditioned on BG + EH)
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
        """Set GT conditioning annealing weight (0=no conditioning, 1=full)."""
        self.cond_weight.fill_(min(max(w, 0.0), 1.0))

    def forward(self, data: Dict[str, torch.Tensor]):
        """
        Args:
            data: MACE-format batch dict with 'batch' key for graph indices
        Returns:
            bg: [batch, 1] bandgap prediction
            gt: [batch, n_gap_classes] log-prob gap type
            eh: [batch, n_eh_classes] log-prob E_hull stability
        """
        batch_idx = data["batch"]
        num_graphs = int(batch_idx.max().item()) + 1

        # MACE backbone → per-atom features
        node_feats = self.backbone(data)  # [N_atoms, 256]

        # Attention pooling → crystal features
        crys_fea = self.attn_pool(node_feats, batch_idx, num_graphs)  # [B, 256]

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
        gt_in = self.gt_proj(
            torch.cat([crys_fea, bg_cond, eh_cond], dim=1)
        )
        gt_hidden = self.gt_layers(gt_in)
        gt = self.gt_classifier(gt_hidden)

        return bg, gt, eh

    def get_backbone_params(self):
        """MACE backbone parameters (for differential learning rate)."""
        return list(self.backbone.parameters())

    def get_head_params(self):
        """All non-backbone parameters."""
        backbone_ids = {id(p) for p in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone_ids]

    def freeze_backbone(self):
        """Freeze all MACE backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, last_n_layers=None):
        """Unfreeze MACE backbone.
        If last_n_layers is specified, only unfreeze the last N interaction layers.
        """
        if last_n_layers is None:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            # Unfreeze last N interaction + product layers
            n = len(self.backbone.interactions)
            for block in list(self.backbone.interactions)[max(0, n - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
            for block in list(self.backbone.products)[max(0, n - last_n_layers):]:
                for p in block.parameters():
                    p.requires_grad = True
