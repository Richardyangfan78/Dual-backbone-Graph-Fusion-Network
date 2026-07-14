"""
ALIGNN multi-task model (PyG implementation, no DGL).

Architecture:
  - Atom embedding: 94-dim one-hot → 256-dim
  - Bond encoding: 80-dim Gaussian RBF of distance
  - Angle encoding: 40-dim Gaussian RBF of cos(angle)
  - 4 ALIGNN layers (line graph + crystal graph edge-gated conv)
  - 4 GCN layers (crystal graph only)
  - Global mean pooling
  - 3 task heads: BG regression, GT classification, EH classification

Reference: Choudhary & DeCost, npj Comput. Mater. 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


# ─── Building blocks ──────────────────────────────────────────────────────────

class EdgeGatedConv(nn.Module):
    """
    Edge-Gated Graph Convolution (from ALIGNN).
    Updates both node features AND edge features in one pass.

    h'_i  = LayerNorm(h_i + Σ_j gate_ij * msg_ij)
    e'_ij = softplus(W_src h_i + W_dst h_j + W_e e_ij)
    gate_ij = sigmoid(same linear combination)
    """
    def __init__(self, node_dim: int, edge_dim: int, out_node_dim: int, out_edge_dim: int):
        super().__init__()
        in_dim = 2 * node_dim + edge_dim
        self.gate_fc  = nn.Linear(in_dim, out_edge_dim)
        self.msg_fc   = nn.Linear(in_dim, out_edge_dim)
        self.node_fc  = nn.Linear(node_dim + out_edge_dim, out_node_dim)
        self.node_norm = nn.LayerNorm(out_node_dim)
        # Projections for residual connections
        self.node_res = nn.Linear(node_dim, out_node_dim, bias=False) \
            if node_dim != out_node_dim else nn.Identity()
        self.edge_res = nn.Linear(edge_dim, out_edge_dim, bias=False) \
            if edge_dim != out_edge_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:          [N, node_dim]
            edge_index: [2, E]
            edge_attr:  [E, edge_dim]
        Returns:
            new_x:      [N, out_node_dim]
            new_e:      [E, out_edge_dim]
        """
        src, dst = edge_index
        x_src = x[src]   # [E, node_dim]
        x_dst = x[dst]   # [E, node_dim]
        cat = torch.cat([x_src, x_dst, edge_attr], dim=-1)  # [E, 2*node_dim+edge_dim]

        gate   = torch.sigmoid(self.gate_fc(cat))            # [E, out_edge_dim]
        msg    = F.softplus(self.msg_fc(cat))                 # [E, out_edge_dim]
        new_e  = self.edge_res(edge_attr) + gate * msg        # [E, out_edge_dim]

        # Aggregate gated messages to nodes
        agg = scatter(new_e, dst, dim=0, dim_size=x.size(0), reduce="sum")  # [N, out_edge_dim]
        new_x = self.node_norm(
            self.node_res(x) + F.silu(self.node_fc(torch.cat([x, agg], dim=-1)))
        )
        return new_x, new_e


class ALIGNNLayer(nn.Module):
    """
    One ALIGNN layer: line graph update → crystal graph update.
    Step 1: update bond features using angle information (on line graph)
    Step 2: update atom features using updated bond features (on crystal graph)
    """
    def __init__(self, node_dim: int, edge_dim: int, angle_dim: int):
        super().__init__()
        # Line graph: nodes=bonds, edges=angles
        self.line_conv    = EdgeGatedConv(edge_dim, angle_dim, edge_dim, angle_dim)
        # Crystal graph: nodes=atoms, edges=bonds
        self.crystal_conv = EdgeGatedConv(node_dim, edge_dim, node_dim, edge_dim)

    def forward(self, atom_x, bond_x, angle_x, crystal_edge_index, line_edge_index):
        """
        atom_x:             [N, node_dim]
        bond_x:             [E, edge_dim]    (crystal graph edge features)
        angle_x:            [T, angle_dim]   (line graph edge features)
        crystal_edge_index: [2, E]
        line_edge_index:    [2, T]  (line graph: nodes are crystal edges)
        """
        # Step 1: update bond features using angles (line graph)
        new_bond_x, new_angle_x = self.line_conv(
            bond_x, line_edge_index, angle_x
        )
        # Step 2: update atom features using updated bond features
        new_atom_x, _ = self.crystal_conv(
            atom_x, crystal_edge_index, new_bond_x
        )
        return new_atom_x, new_bond_x, new_angle_x


# ─── ALIGNN Multi-task model ──────────────────────────────────────────────────

class ALIGNNMultiTaskPyG(nn.Module):
    """ALIGNN backbone + 3 task heads (BG regression, GT classification, EH classification)."""

    def __init__(
        self,
        atom_input_dim: int = 94,      # one-hot atomic number
        edge_input_dim: int = 80,      # RBF distance bins
        angle_input_dim: int = 40,     # RBF angle bins
        hidden_dim: int = 256,
        n_alignn_layers: int = 4,
        n_gcn_layers: int = 4,
        n_gap_classes: int = 2,
        n_eh_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Input projections ────────────────────────────────────────
        self.atom_proj  = nn.Sequential(nn.Linear(atom_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.bond_proj  = nn.Sequential(nn.Linear(edge_input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.angle_proj = nn.Sequential(nn.Linear(angle_input_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU())

        angle_dim = hidden_dim // 2

        # ── ALIGNN layers ─────────────────────────────────────────────
        self.alignn_layers = nn.ModuleList([
            ALIGNNLayer(hidden_dim, hidden_dim, angle_dim)
            for _ in range(n_alignn_layers)
        ])

        # ── GCN layers (crystal graph only) ──────────────────────────
        self.gcn_layers = nn.ModuleList([
            EdgeGatedConv(hidden_dim, hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_gcn_layers)
        ])

        self.drop = nn.Dropout(p=dropout)

        # ── Task heads ────────────────────────────────────────────────
        mid = hidden_dim // 2

        # Head 1: Band gap regression
        self.bg_fc1  = nn.Linear(hidden_dim, mid)
        self.bg_bn1  = nn.LayerNorm(mid)
        self.bg_fc2  = nn.Linear(mid, mid)
        self.bg_bn2  = nn.LayerNorm(mid)
        self.bg_skip = nn.Linear(hidden_dim, mid)
        self.bg_out  = nn.Linear(mid, 1)

        # Head 2: Gap type (conditioned on BG)
        self.gt_proj = nn.Linear(hidden_dim + 1, hidden_dim)
        self.gt_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid), nn.SiLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(mid, n_gap_classes), nn.LogSoftmax(dim=1),
        )

        # Head 3: Energy above hull classification
        self.eh_head = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.LayerNorm(mid), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, n_eh_classes), nn.LogSoftmax(dim=1),
        )

    def forward(self, data):
        x          = data.x                  # [N_total, atom_dim]
        edge_index = data.edge_index         # [2, E_total]
        edge_attr  = data.edge_attr          # [E_total, dist_bins]
        line_edge_index = data.line_edge_index  # [2, T_total]
        line_attr  = data.line_attr          # [T_total, angle_bins]
        batch      = data.batch              # [N_total]

        # ── Project inputs ─────────────────────────────────────────
        atom_x  = self.atom_proj(x)          # [N, H]
        bond_x  = self.bond_proj(edge_attr)  # [E, H]
        angle_x = self.angle_proj(line_attr) # [T, H/2]

        # ── ALIGNN layers ──────────────────────────────────────────
        for layer in self.alignn_layers:
            atom_x, bond_x, angle_x = layer(
                atom_x, bond_x, angle_x, edge_index, line_edge_index
            )

        # ── GCN layers ─────────────────────────────────────────────
        for layer in self.gcn_layers:
            atom_x, bond_x = layer(atom_x, edge_index, bond_x)

        # ── Global mean pooling ────────────────────────────────────
        h = scatter(atom_x, batch, dim=0, reduce="mean")  # [B, H]
        h = self.drop(h)

        # ── Task heads ─────────────────────────────────────────────
        skip = self.bg_skip(h)
        bg   = self.drop(F.silu(self.bg_bn1(self.bg_fc1(h))))
        bg   = self.drop(F.silu(self.bg_bn2(self.bg_fc2(bg))))
        bg   = self.bg_out(bg + skip)                      # [B, 1]

        gt   = self.gt_head(self.gt_proj(torch.cat([h, bg.detach()], dim=1)))
        eh   = self.eh_head(h)

        return bg.squeeze(-1), gt, eh
