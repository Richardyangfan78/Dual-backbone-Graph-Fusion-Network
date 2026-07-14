"""
Extend atom_init.json with three continuous physical features:
  1. Electronegativity (Pauling scale)
  2. SOC strength ξ (eV) — spin-orbit coupling of outermost valence shell
  3. Ionic radius (Å, most common oxidation state, Shannon)

All three features are min-max normalized to [0, 1] to match
the scale of the existing binary features.
"""

import json
import numpy as np
from pymatgen.core.periodic_table import Element, Species

# ── SOC strength ξ (eV), valence-shell spin-orbit coupling ──────────────────
# Sources: relativistic Hartree-Fock/DFT atomic calculations;
# Pyykko & Desclaux (1979), Dolg (2011), and experimental spectroscopy.
# s-block elements have l=0 → ξ≈0; values increase steeply with Z.
SOC = {
     1: 0.000,  2: 0.000,  3: 0.000,  4: 0.000,  5: 0.002,
     6: 0.004,  7: 0.006,  8: 0.009,  9: 0.013, 10: 0.018,
    11: 0.001, 12: 0.002, 13: 0.012, 14: 0.020, 15: 0.030,
    16: 0.045, 17: 0.064, 18: 0.089, 19: 0.004, 20: 0.007,
    21: 0.020, 22: 0.030, 23: 0.040, 24: 0.050, 25: 0.060,
    26: 0.070, 27: 0.080, 28: 0.090, 29: 0.100, 30: 0.005,
    31: 0.100, 32: 0.140, 33: 0.180, 34: 0.230, 35: 0.290,
    36: 0.360, 37: 0.020, 38: 0.030, 39: 0.070, 40: 0.090,
    41: 0.110, 42: 0.140, 43: 0.170, 44: 0.200, 45: 0.240,
    46: 0.280, 47: 0.320, 48: 0.030, 49: 0.270, 50: 0.340,
    51: 0.420, 52: 0.510, 53: 0.620, 54: 0.750, 55: 0.060,
    56: 0.090, 57: 0.080, 58: 0.100, 59: 0.110, 60: 0.130,
    61: 0.140, 62: 0.160, 63: 0.170, 64: 0.190, 65: 0.210,
    66: 0.230, 67: 0.250, 68: 0.280, 69: 0.310, 70: 0.340,
    71: 0.200, 72: 0.350, 73: 0.450, 74: 0.570, 75: 0.700,
    76: 0.840, 77: 0.990, 78: 1.150, 79: 1.320, 80: 1.500,
    81: 0.800, 82: 0.910, 83: 1.500, 84: 1.750, 85: 2.000,
    86: 2.300, 87: 0.200, 88: 0.280, 89: 0.300, 90: 0.350,
    91: 0.430, 92: 0.520, 93: 0.610, 94: 0.710, 95: 0.810,
    96: 0.920, 97: 1.030, 98: 1.150, 99: 1.280, 100: 1.420,
}

# ── Preferred oxidation state for ionic radius lookup ────────────────────────
# Used to pick Shannon ionic radius for the most chemically common form.
PREFERRED_OX = {
     1:  1,  2:  0,  3:  1,  4:  2,  5:  3,  6:  4,  7: -3,  8: -2,
     9: -1, 10:  0, 11:  1, 12:  2, 13:  3, 14:  4, 15:  5, 16: -2,
    17: -1, 18:  0, 19:  1, 20:  2, 21:  3, 22:  4, 23:  5, 24:  3,
    25:  2, 26:  3, 27:  3, 28:  2, 29:  2, 30:  2, 31:  3, 32:  4,
    33:  3, 34: -2, 35: -1, 36:  0, 37:  1, 38:  2, 39:  3, 40:  4,
    41:  5, 42:  6, 43:  4, 44:  4, 45:  3, 46:  2, 47:  1, 48:  2,
    49:  3, 50:  4, 51:  3, 52: -2, 53: -1, 54:  0, 55:  1, 56:  2,
    57:  3, 58:  4, 59:  3, 60:  3, 61:  3, 62:  3, 63:  3, 64:  3,
    65:  3, 66:  3, 67:  3, 68:  3, 69:  3, 70:  3, 71:  3, 72:  4,
    73:  5, 74:  6, 75:  4, 76:  4, 77:  3, 78:  2, 79:  3, 80:  2,
    81:  1, 82:  2, 83:  3, 84:  4, 85: -1, 86:  0, 87:  1, 88:  2,
    89:  3, 90:  4, 91:  5, 92:  6, 93:  5, 94:  4, 95:  3, 96:  3,
    97:  3, 98:  3, 99:  3, 100:  3,
}

# ── Load current atom_init.json ──────────────────────────────────────────────
SRC  = "/path/to/Dual-backbone-Graph-Fusion-Network/Data/multitask/atom_init.json"
DEST = "/path/to/Dual-backbone-Graph-Fusion-Network/Data/multitask/atom_init_extended.json"

with open(SRC) as f:
    orig = json.load(f)

# ── Collect raw continuous values ────────────────────────────────────────────
electro_raw = {}
soc_raw     = {}
radius_raw  = {}

FALLBACK_RADIUS = 1.40  # Å (average covalent radius, used if not found)

for z_str in orig:
    z = int(z_str)
    elem = Element.from_Z(z)

    # 1. Electronegativity (Pauling)
    # Noble gases have no Pauling EN (don't form bonds) → assign 0.0
    en = elem.X
    if en is None or (hasattr(en, '__float__') and np.isnan(float(en))):
        en = 0.0
    electro_raw[z] = float(en)

    # 2. SOC strength
    soc_raw[z] = SOC.get(z, 0.0)

    # 3. Ionic radius (Shannon, most common oxidation state)
    ox = PREFERRED_OX.get(z, 0)
    radius = None
    if ox != 0:
        try:
            sp = Species(elem.symbol, ox)
            radius = sp.ionic_radius  # Å, coordination 6 default
        except Exception:
            pass
    if radius is None:
        # fallback: atomic radius
        radius = elem.atomic_radius
        if radius is None:
            radius = FALLBACK_RADIUS
        else:
            radius = float(radius)
    radius_raw[z] = float(radius)

# ── Min-max normalize each feature to [0, 1] ─────────────────────────────────
def minmax(d):
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    return {k: float((v - lo) / (hi - lo + 1e-8)) for k, v in d.items()}

electro_norm = minmax(electro_raw)
soc_norm     = minmax(soc_raw)
radius_norm  = minmax(radius_raw)

# ── Print summary for key elements ──────────────────────────────────────────
print("=== Raw values for key elements ===")
for sym, z in [("Bi",83),("Pb",82),("Sn",50),("Te",52),("I",53),("Br",35),("Se",34)]:
    print(f"  {sym:3s} (Z={z:3d}): EN={electro_raw[z]:.2f} "
          f"SOC={soc_raw[z]:.3f}eV  IR={radius_raw[z]:.3f}Å"
          f"  → norm [{electro_norm[z]:.3f}, {soc_norm[z]:.3f}, {radius_norm[z]:.3f}]")

# ── Build extended atom_init ─────────────────────────────────────────────────
extended = {}
for z_str, feats in orig.items():
    z = int(z_str)
    new_feats = feats + [
        round(electro_norm[z], 6),
        round(soc_norm[z],     6),
        round(radius_norm[z],  6),
    ]
    extended[z_str] = new_feats

# Verify
sample_z = "83"
assert len(extended[sample_z]) == len(orig[sample_z]) + 3, "Dimension mismatch!"
print(f"\nOriginal dim: {len(orig[sample_z])}  →  Extended dim: {len(extended[sample_z])}")
print(f"Last 3 values for Bi: {extended[sample_z][-3:]}")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(DEST, "w") as f:
    json.dump(extended, f, indent=2)
print(f"\nSaved: {DEST}")
