"""
Query MP for existing ABX2Y2 chalcohalides, then generate missing ones via ionic-radii substitution.
A = monovalent (Li,Na,K,Rb,Cs,Ag,Cu)
B = trivalent  (Bi,Sb,As,In,Ga)
X = chalcogen  (S,Se,Te)
Y = halogen    (F,Cl,Br,I)
"""
import os, sys, itertools
from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester

API_KEY  = '9y32i0wNJMyxDTPAWo9wBC52gUiMOB8h'
PROJECT  = '/path/to/Dual-backbone-Graph-Fusion-Network'
PRED_DIR = os.path.join(PROJECT, 'Data/predict')
OUT_DIR  = os.path.join(PROJECT, 'Data/predict_new')
os.makedirs(OUT_DIR, exist_ok=True)

A_ELEMENTS = ['Li','Na','K','Rb','Cs','Ag','Cu']
B_ELEMENTS = ['Bi','Sb','As','In','Ga']
X_ELEMENTS = ['S','Se','Te']
Y_ELEMENTS = ['F','Cl','Br','I']

# Ionic radii (Shannon, 6-coord, pm)
RADII = {
    'Li':76,'Na':102,'K':138,'Rb':152,'Cs':167,'Ag':115,'Cu':77,
    'Bi':103,'Sb':76,'As':58,'In':80,'Ga':62,
    'S':184,'Se':198,'Te':221,
    'F':133,'Cl':181,'Br':196,'I':220,
}

# ── 1. Build full target set ──────────────────────────────────────────────────
all_targets = set()
for a,b,x,y in itertools.product(A_ELEMENTS, B_ELEMENTS, X_ELEMENTS, Y_ELEMENTS):
    all_targets.add(f'{a}{b}{x}{y}2')

print(f'Full chemical space: {len(all_targets)} compounds')

# ── 2. Already in predict dir ─────────────────────────────────────────────────
existing = set()
for fn in os.listdir(PRED_DIR):
    if fn.endswith('.cif'):
        existing.add(fn.replace('.cif','').replace('.cif.tmp',''))
print(f'Already have CIFs: {len(existing)}')

# ── 3. Query MP ───────────────────────────────────────────────────────────────
print('Querying Materials Project...')
mp_formulas = set()
with MPRester(API_KEY) as mpr:
    for a,b in itertools.product(A_ELEMENTS, B_ELEMENTS):
        results = mpr.materials.summary.search(
            elements=[a, b],
            num_elements=(4,4),
            fields=['formula_pretty','structure']
        )
        for r in results:
            f = r.formula_pretty
            # Check it matches ABX2Y2 pattern (chalcogen + halogen)
            elems = set(Element(e).symbol for e in r.structure.species)
            chalcogens = elems & {'S','Se','Te'}
            halogens   = elems & {'F','Cl','Br','I'}
            a_elems    = elems & set(A_ELEMENTS)
            b_elems    = elems & set(B_ELEMENTS)
            if chalcogens and halogens and a_elems and b_elems and len(elems)==4:
                mp_formulas.add(f)

print(f'MP has {len(mp_formulas)} relevant chalcohalides: {sorted(mp_formulas)[:10]}...')

# ── 4. Find missing (not in existing CIFs, not in MP) ────────────────────────
missing = all_targets - existing - mp_formulas
print(f'Missing (need to generate): {len(missing)}')

# ── 5. Generate structures via substitution ───────────────────────────────────
def best_template(a, b, x, y):
    """Find closest existing structure as template (same B,X,Y if possible)."""
    # Priority: same B,X,Y with any A; then same A,X,Y with any B
    for ta in ['Ag','Cu','Na','K','Li','Cs']:
        name = f'{ta}{b}{x}{y}2'
        path = os.path.join(PRED_DIR, name+'.cif')
        if os.path.exists(path) and ta != a:
            return path, {'A_old':ta,'A_new':a,'B_old':b,'B_new':b,'X_old':x,'X_new':x,'Y_old':y,'Y_new':y}
    for tb in ['Bi','In','Sb','As','Ga']:
        name = f'{a}{tb}{x}{y}2'
        path = os.path.join(PRED_DIR, name+'.cif')
        if os.path.exists(path) and tb != b:
            return path, {'A_old':a,'A_new':a,'B_old':tb,'B_new':b,'X_old':x,'X_new':x,'Y_old':y,'Y_new':y}
    return None, None

generated, failed = [], []
for formula in sorted(missing):
    # Parse formula: e.g. RbBiSF2 -> a=Rb,b=Bi,x=S,y=F
    a = next((e for e in A_ELEMENTS if formula.startswith(e)), None)
    rest = formula[len(a):]
    b = next((e for e in B_ELEMENTS if rest.startswith(e)), None)
    rest2 = rest[len(b):]
    x = next((e for e in X_ELEMENTS if rest2.startswith(e)), None)
    y = rest2[len(x):].rstrip('2')

    tpl_path, subs = best_template(a,b,x,y)
    if not tpl_path:
        failed.append(formula); continue

    try:
        s = Structure.from_file(tpl_path)
        # Replace species
        replace_map = {}
        if subs['A_old'] != subs['A_new']: replace_map[subs['A_old']] = subs['A_new']
        if subs['B_old'] != subs['B_new']: replace_map[subs['B_old']] = subs['B_new']
        if subs['X_old'] != subs['X_new']: replace_map[subs['X_old']] = subs['X_new']
        if subs['Y_old'] != subs['Y_new']: replace_map[subs['Y_old']] = subs['Y_new']
        s.replace_species(replace_map)

        # Scale lattice by ionic radii ratio
        scale = 1.0
        if subs['A_old'] != subs['A_new']:
            scale *= (RADII[subs['A_new']]/RADII[subs['A_old']])**(1/3)
        if subs['B_old'] != subs['B_new']:
            scale *= (RADII[subs['B_new']]/RADII[subs['B_old']])**(1/3)
        if subs['X_old'] != subs['X_new']:
            scale *= (RADII[subs['X_new']]/RADII[subs['X_old']])**(1/3)
        if subs['Y_old'] != subs['Y_new']:
            scale *= (RADII[subs['Y_new']]/RADII[subs['Y_old']])**(1/3)
        s.scale_lattice(s.volume * scale**3)

        out_path = os.path.join(OUT_DIR, formula+'.cif')
        CifWriter(s).write_file(out_path)
        generated.append(formula)
    except Exception as e:
        failed.append(f'{formula}({e})')

print(f'\nGenerated: {len(generated)}')
print(f'Failed:    {len(failed)}')
if failed: print('  ', failed[:10])
print(f'CIFs saved to: {OUT_DIR}')
