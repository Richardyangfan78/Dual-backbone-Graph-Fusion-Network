"""
M3GNet multi-task model (PyG implementation, no DGL).

Architecture inspired by:
  Chen & Ong, "A universal graph deep learning interatomic potential...", Nature Comput. Sci. 2022

Key features:
  - Atom embedding: 94-dim one-hot → 64-dim learned embedding
  - Bond encoding: 80-dim Gaussian RBF
  - 3-body interactions via line graph (same as ALIGNN data pipeline)
  - 3 M3GNetBlock layers
  - Atom-wise readout + global mean pooling
  - 3 task heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class SoftplusLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return F.softplus(self.fc(x))


class M3GNetBlock(nn.Module):
    """
    One M3GNet interaction block.

    M3GNet update (simplified, invariant version):
      1. 3-body gating: for each bond (i,j), aggregate angle features
         from triplets (i,j,k) to get a 3-body correction
      2. Bond update: e'_ij = SiLU(W [h_i, h_j, e_ij, e3b_ij])
      3. Atom update: h'_i  = h_i + SiLU(W [h_i, Σ_j e'_ij])
    """
    def __init__(self, node_dim: int, edge_dim: int, angle_dim: int):
        super().__init__()
        # 3-body: aggregate angle info onto each bond
        # For each edge e1=(i→j), sum over all triplets where e1 is the "left" bond
        self.angle_to_bond = nn.Sequential(
            nn.Linear(angle_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        # Gate for 3-body signal
        self.angle_gate = nn.Sequential(
            nn.Linear(angle_dim, edge_dim),
            nn.Sigmoid(),
        )

        # Bond update
        bond_in = 2 * node_dim + 2 * edge_dim  # h_i, h_j, e_ij, e3b_ij
        self.bond_fc = nn.Sequential(
            nn.Linear(bond_in, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.bond_gate = nn.Sequential(
            nn.Linear(bond_in, edge_dim),
            nn.Sigmoid(),
        )

        # Atom update
        atom_in = node_dim + edge_dim
        self.atom_fc = nn.Sequential(
            nn.Linear(atom_in, node_dim),
            nn.LayerNorm(node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, atom_x, bond_x, angle_x, crystal_edge_index, line_edge_index):
        """
        atom_x:             [N, node_dim]
        bond_x:             [E, edge_dim]
        angle_x:            [T, angle_dim]  (line graph edges)
        crystal_edge_index: [2, E]
        line_edge_index:    [2, T]  (line graph: nodes=crystal_edges)
        """
        src, dst = crystal_edge_index   # [E]
        E = bond_x.size(0)

        # ── Step 1: 3-body aggregation onto bonds ─────────────────────
        # line_edge_index[0] = e1 (the "source" edge in line graph = i→j bond)
        # For each bond e1, aggregate angle info from all triplets (e1, e2)
        if line_edge_index.size(1) > 0:
            le_src = line_edge_index[0]  # [T] — which crystal edge is the "left" bond
            e3b_msg   = self.angle_to_bond(angle_x)  # [T, edge_dim]
            e3b_gate  = self.angle_gate(angle_x)     # [T, edge_dim]
            e3b_gated = e3b_gate * e3b_msg            # [T, edge_dim]
            e3b = scatter(e3b_gated, le_src, dim=0, dim_size=E, reduce="mean")  # [E, edge_dim]
        else:
            e3b = torch.zeros(E, bond_x.size(1), device=bond_x.device)

        # ── Step 2: Bond update ───────────────────────────────────────
        x_src = atom_x[src]   # [E, node_dim]
        x_dst = atom_x[dst]   # [E, node_dim]
        bond_in = torch.cat([x_src, x_dst, bond_x, e3b], dim=-1)  # [E, 2*N+2*E_dim]

        gate    = self.bond_gate(bond_in)            # [E, edge_dim]
        new_bond_x = bond_x + gate * self.bond_fc(bond_in)  # residual

        # ── Step 3: Atom update ───────────────────────────────────────
        # Aggregate updated bond messages to atoms
        agg = scatter(new_bond_x, dst, dim=0, dim_size=atom_x.size(0), reduce="mean")  # [N, edge_dim]
        atom_in = torch.cat([atom_x, agg], dim=-1)  # [N, node_dim+edge_dim]
        new_atom_x = atom_x + self.atom_fc(atom_in)  # residual

        return new_atom_x, new_bond_x, angle_x


# ─── M3GNet Multi-task model ─────────────────────────────────────────────────

class M3GNetMultiTaskPyG(nn.Module):
    """M3GNet backbone + 3 task heads."""

    def __init__(
        self,
        atom_input_dim: int = 94,
        edge_input_dim: int = 80,
        angle_input_dim: int = 40,
        hidden_dim: int = 64,
        n_blocks: int = 3,
        n_gap_classes: int = 2,
        n_eh_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Input projections ─────────────────────────────────────────
        self.atom_proj  = nn.Sequential(
            nn.Linear(atom_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.bond_proj  = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.angle_proj = nn.Sequential(
            nn.Linear(angle_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )

        # ── M3GNet blocks ─────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            M3GNetBlock(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(n_blocks)
        ])

        self.drop = nn.Dropout(p=dropout)

        # ── Task heads ────────────────────────────────────────────────
        mid = hidden_dim * 2  # expand for heads

        # BG head
        self.bg_head = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.LayerNorm(mid), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid // 2), nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(mid // 2, 1),
        )
        self.bg_res = nn.Linear(hidden_dim, 1, bias=False)

        # GT head (conditioned on BG)
        self.gt_proj = nn.Linear(hidden_dim + 1, hidden_dim)
        self.gt_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid), nn.SiLU(),
            nn.Linear(mid, n_gap_classes), nn.LogSoftmax(dim=1),
        )

        # EH head
        self.eh_head = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.LayerNorm(mid), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, n_eh_classes), nn.LogSoftmax(dim=1),
        )

    def forward(self, data):
        x               = data.x
        edge_index      = data.edge_index
        edge_attr       = data.edge_attr
        line_edge_index = data.line_edge_index
        line_attr       = data.line_attr
        batch           = data.batch

        # ── Project inputs ─────────────────────────────────────────
        atom_x  = self.atom_proj(x)
        bond_x  = self.bond_proj(edge_attr)
        angle_x = self.angle_proj(line_attr) if line_attr.size(0) > 0 else \
                  torch.zeros(0, atom_x.size(1), device=atom_x.device)

        # ── M3GNet blocks ──────────────────────────────────────────
        for block in self.blocks:
            atom_x, bond_x, angle_x = block(
                atom_x, bond_x, angle_x, edge_index, line_edge_index
            )

        # ── Atom-wise weighting + global pooling ───────────────────
        h = scatter(atom_x, batch, dim=0, reduce="mean")  # [B, H]
        h = self.drop(h)

        # ── Task heads ─────────────────────────────────────────────
        bg = self.bg_res(h) + self.bg_head(h)  # [B, 1]
        gt = self.gt_head(self.gt_proj(torch.cat([h, bg.detach()], dim=1)))
        eh = self.eh_head(h)

        return bg.squeeze(-1), gt, eh
