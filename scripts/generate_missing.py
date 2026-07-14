"""
系统生成缺失的 M-Ch-X 组合：
Type1: A(I)C(III)ChX2  - 用 predict_dft_relaxed 中最近邻模板
Type2: C(III)ChX       - 用 predict_jacs_relaxed 中 BiSI 类模板  
Type3: B(II)2C(III)Ch2X3 - 用 predict_jacs_relaxed 中 Sn2SbSX3 模板
"""
import os, warnings
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Element, Composition
from pymatgen.io.cif import CifWriter
from math import gcd
from functools import reduce
from itertools import product as iproduct
from collections import defaultdict

BASE     = "/path/to/Dual-backbone-Graph-Fusion-Network"
DFT_DIR  = BASE + "/Data/predict_dft_relaxed"
JACS_DIR = BASE + "/Data/predict_jacs_relaxed"
OUT_DIR  = BASE + "/Data/predict_new_generated"
os.makedirs(OUT_DIR, exist_ok=True)

PRED_FINAL = BASE + "/Data/predict_final"
CH = {"S","Se","Te"}
X  = {"F","Cl","Br","I"}

# ── Shannon ionic radii (CN=6 typical) ──
RADII = {
    # A(+1)
    "Li":0.76,"Na":1.02,"K":1.38,"Rb":1.52,"Cs":1.67,"Cu":0.77,"Ag":1.15,"Au":1.37,
    # B(+2)
    "Mg":0.72,"Ca":1.00,"Sr":1.18,"Ba":1.35,"Mn":0.83,"Fe":0.78,"Co":0.75,
    "Ni":0.69,"Zn":0.74,"Sn":0.69,"Pd":0.86,"Pt":0.80,
    # C(+3)
    "Al":0.535,"Ga":0.62,"In":0.80,"Sc":0.745,"Y":0.90,"La":1.032,"Lu":0.861,
    "Bi":1.03,"Sb":0.76,"Ti":0.670,"V":0.640,"Nb":0.640,"Mo":0.650,
    "Zr":0.720,"Hf":0.710,"Ta":0.640,"W":0.600,"Fe":0.645,"Ru":0.680,"Rh":0.665,
    # Ch(-2)
    "S":1.84,"Se":1.98,"Te":2.21,
    # X(-1)
    "F":1.33,"Cl":1.81,"Br":1.96,"I":2.20,
}

# ── Already in predict_final ──
existing = set()
for f in os.listdir(PRED_FINAL):
    if f.endswith(".cif"):
        existing.add(f.replace(".cif",""))

print("Existing in predict_final: " + str(len(existing)))

# ── Element sets ──
A1 = ["Li","Na","K","Rb","Cs","Cu","Ag","Au"]
C3 = ["Al","Ga","In","Sc","Y","La","Lu","Bi","Sb","Ti","V","Nb","Mo","Zr","Hf","Ta","W","Fe","Ru","Rh"]
B2 = ["Mg","Ca","Sr","Ba","Mn","Fe","Co","Ni","Zn","Sn","Pd","Pt"]  # removed Cu (in A1)
CH_l = ["S","Se","Te"]
X_l  = ["F","Cl","Br","I"]

# ── Build Type1 template library from DFT relaxed structures ──
# Map (A,C,Ch,X) -> structure
t1_templates = {}
for fname in os.listdir(DFT_DIR):
    if not fname.endswith(".vasp"): continue
    try:
        struct = Structure.from_file(os.path.join(DFT_DIR, fname))
        comp = struct.composition
        els = {str(e) for e in comp.elements}
        m_els = [e for e in els if e not in CH and e not in X]
        ch_els = [e for e in els if e in CH]
        x_els = [e for e in els if e in X]
        if len(m_els)==2 and len(ch_els)==1 and len(x_els)==1:
            # Identify A(+1) and C(+3)
            # A is the one with smaller count/monovalent
            ch = ch_els[0]; x = x_els[0]
            key = tuple(sorted(m_els)) + (ch, x)
            t1_templates[key] = (fname, struct)
    except: pass

print("Type1 DFT templates loaded: " + str(len(t1_templates)))

def get_radius(el, default=1.0):
    return RADII.get(el, default)

def substitute_and_scale(template_struct, old_els, new_els):
    """Substitute elements and scale lattice by ionic radii ratio."""
    struct = template_struct.copy()
    # Build species map
    species_map = {}
    for old, new in zip(old_els, new_els):
        if old != new:
            species_map[old] = new
    if species_map:
        struct.replace_species(species_map)
    # Scale lattice: volume ~ sum of atomic radii^3
    # Use geometric mean of radii ratios
    vol_scale = 1.0
    for old, new in zip(old_els, new_els):
        r_old = get_radius(old)
        r_new = get_radius(new)
        vol_scale *= (r_new / r_old)
    vol_scale = vol_scale ** (1/len(old_els))
    struct.scale_lattice(struct.volume * vol_scale**3)
    return struct

# ── TYPE 1: A(I)C(III)ChX2 ──
print("\nGenerating Type1 A(I)C(III)ChX2...")
t1_generated = 0
t1_skipped = 0

# Pick best template for each combination (minimize total radii difference)
for a, c, ch, x in iproduct(A1, C3, CH_l, X_l):
    # Formula: A=1, C=1, Ch=1, X=2
    from pymatgen.core import Composition as Comp
    formula = a + c + ch + x + "2"
    try:
        red = Comp(formula).reduced_formula
    except:
        red = formula
    
    if red in existing or formula in existing:
        t1_skipped += 1
        continue
    
    # Find best template (same A or same C preferred)
    best_template = None
    best_score = float('inf')
    for (ta, tc, tch, tx), (tfname, tstruct) in t1_templates.items():
        score = (abs(get_radius(a)-get_radius(ta)) + 
                 abs(get_radius(c)-get_radius(tc)) +
                 abs(get_radius(ch)-get_radius(tch)) +
                 abs(get_radius(x)-get_radius(tx)))
        if score < best_score:
            best_score = score
            best_template = (ta, tc, tch, tx, tstruct)
    
    if best_template is None: continue
    ta, tc, tch, tx, tstruct = best_template
    try:
        new_struct = substitute_and_scale(
            tstruct, [ta, tc, tch, tx], [a, c, ch, x]
        )
        out_path = os.path.join(OUT_DIR, formula + ".cif")
        CifWriter(new_struct).write_file(out_path)
        t1_generated += 1
    except Exception as e:
        pass

print("Type1 generated: " + str(t1_generated) + "  skipped (existing): " + str(t1_skipped))

# ── TYPE 2: C(III)ChX ──
print("\nGenerating Type2 C(III)ChX...")
# Use BiSI from JACS as template (most common)
t2_template = None
for fname in ["BiSI.cif","BiSBr.cif","BiSCl.cif","BiSeI.cif","SbSI.cif"]:
    p = os.path.join(JACS_DIR, fname)
    if os.path.exists(p):
        try:
            t2_template = (fname.replace(".cif",""), Structure.from_file(p))
            print("  Type2 template: " + fname)
            break
        except: pass

# Also check Inorganic_datasets for BiSI
if t2_template is None:
    for fname in os.listdir(BASE + "/Data/Inorganic_datasets"):
        if not fname.endswith(".cif"): continue
        try:
            s = Structure.from_file(os.path.join(BASE+"/Data/Inorganic_datasets", fname))
            if s.composition.reduced_formula == "BiSI":
                t2_template = ("BiSI", s)
                print("  Type2 template from Inorganic: " + fname)
                break
        except: pass

t2_generated = 0
t2_skipped = 0
if t2_template:
    t_name, t_struct = t2_template
    # Identify elements in template
    t_els = {str(e) for e in t_struct.composition.elements}
    t_m = [e for e in t_els if e not in CH and e not in X][0]
    t_ch = [e for e in t_els if e in CH][0]
    t_x  = [e for e in t_els if e in X][0]
    
    for c, ch, x in iproduct(C3, CH_l, X_l):
        formula = c + ch + x
        try:
            red = Comp(formula).reduced_formula
        except: red = formula
        if red in existing or formula in existing:
            t2_skipped += 1; continue
        try:
            new_struct = substitute_and_scale(
                t_struct, [t_m, t_ch, t_x], [c, ch, x]
            )
            CifWriter(new_struct).write_file(os.path.join(OUT_DIR, formula+".cif"))
            t2_generated += 1
        except: pass
    print("Type2 generated: " + str(t2_generated) + "  skipped: " + str(t2_skipped))

# ── TYPE 3: B(II)2C(III)Ch2X3 ──
print("\nGenerating Type3 B(II)2C(III)Ch2X3...")
t3_template = None
for fname in ["Sn2SbS2I3.cif","Sn2SbSe2I3.cif","Pb2SbS2I3.cif"]:
    p = os.path.join(JACS_DIR, fname)
    if os.path.exists(p):
        try:
            t3_template = (fname.replace(".cif",""), Structure.from_file(p))
            print("  Type3 template: " + fname)
            break
        except: pass

t3_generated = 0
t3_skipped = 0
if t3_template:
    t_name, t_struct = t3_template
    t_els = {str(e): int(t_struct.composition[e]) for e in t_struct.composition.elements}
    # Identify roles: B(+2) has count=2, C(+3) has count=1
    m_els_t = {k:v for k,v in t_els.items() if k not in CH and k not in X}
    t_ch = [k for k in t_els if k in CH][0]
    t_x  = [k for k in t_els if k in X][0]
    # B is the one with larger count
    sorted_m = sorted(m_els_t.items(), key=lambda x: -x[1])
    t_b = sorted_m[0][0]  # count=2
    t_c = sorted_m[1][0]  # count=1
    
    for b, c, ch, x in iproduct(B2, C3, CH_l, X_l):
        if b == c: continue  # same element for both sites
        # Formula: 2B + 1C + 2Ch + 3X
        formula = b + "2" + c + ch + "2" + x + "3"
        try:
            red = Comp(formula).reduced_formula
        except: red = formula
        if red in existing or formula in existing:
            t3_skipped += 1; continue
        try:
            new_struct = substitute_and_scale(
                t_struct, [t_b, t_c, t_ch, t_x], [b, c, ch, x]
            )
            CifWriter(new_struct).write_file(os.path.join(OUT_DIR, formula+".cif"))
            t3_generated += 1
        except: pass
    print("Type3 generated: " + str(t3_generated) + "  skipped: " + str(t3_skipped))

total = t1_generated + t2_generated + t3_generated
print("\nTotal generated: " + str(total))
print("Output: " + OUT_DIR)
