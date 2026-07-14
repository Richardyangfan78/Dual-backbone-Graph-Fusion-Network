import sys, os
sys.path.insert(0, ".")
import torch
from torchinfo import summary
from mace.calculators import mace_mp
from model_alignn_pyg import ALIGNNMultiTaskPyG
from model_dual_backbone import DualBackboneMultiTask

device = torch.device("cpu")
print("Loading MACE-MP-0...")
calc = mace_mp(model="small", default_dtype="float32", device="cpu")
mace_model = calc.models[0]

print("Creating ALIGNN model...")
alignn_model = ALIGNNMultiTaskPyG(
    atom_input_dim=94, edge_input_dim=80, angle_input_dim=40,
    hidden_dim=256, n_alignn_layers=4, n_gcn_layers=4, dropout=0.3,
)

print("Building DualBackboneMultiTask...")
model = DualBackboneMultiTask(
    mace_model, alignn_model,
    h_fea_len=256, n_gap_classes=2, n_eh_classes=2,
    dropout=0.3, n_attn_heads=8,
)

print()
print("="*80)
print("DualBackboneMultiTask - Architecture Summary")
print("="*80)
summary(model, depth=5, col_names=["num_params", "trainable"])
