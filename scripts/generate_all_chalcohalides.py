"""
Generate ALL chalcohalide formula types not already in predict dirs or MP:

Type 1 (already done): A(I)B(III)Ch2Y2  → ABX2Y2
  A=Li,Na,K,Rb,Cs,Ag,Cu  B=Bi,Sb,As,In,Ga  Ch=S,Se,Te  Y=Cl,Br,I  [297 done]

Type 2 (NEW): M(III)ChY  → ternary BiSI-type
  M=Bi,Sb,As,In,Ga  Ch=S,Se,Te  Y=Cl,Br,I  [45 total]

Type 3 (NEW): A(II)2B(III)Ch2Y3  → Sn2SbS2I3-type
  A=Sn,Pb,Ge  B=Bi,Sb,In  Ch=S,Se,Te  Y=Cl,Br,I  [81 total]
"""
import os, sys, itertools
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester

API_KEY = '9y32i0wNJMyxDTPAWo9wBC52gUiMOB8h'
PROJECT = '/path/to/Dual-backbone-Graph-Fusion-Network'
PRED_DIR  = os.path.join(PROJECT, 'Data/predict')
NEW_DIR   = os.path.join(PROJECT, 'Data/predict_new')
TYPE2_DIR = os.path.join(PROJECT, 'Data/predict_type2')
TYPE3_DIR = os.path.join(PROJECT, 'Data/predict_type3')
for d in [TYPE2_DIR, TYPE3_DIR]: os.makedirs(d, exist_ok=True)

RADII = {
    # Monovalent
    'Li':76,'Na':102,'K':138,'Rb':152,'Cs':167,'Ag':115,'Cu':77,
    # Divalent
    'Sn':118,'Pb':119,'Ge':73,
    # Trivalent
    'Bi':103,'Sb':76,'As':58,'In':80,'Ga':62,
    # Chalcogen
    'S':184,'Se':198,'Te':221,
    # Halogen
    'Cl':181,'Br':196,'I':220,
}

M3  = ['Bi','Sb','As','In','Ga']
CH  = ['S','Se','Te']
HAL = ['Cl','Br','I']
A2  = ['Sn','Pb','Ge']
B3  = ['Bi','Sb','In']

# ── Query MP for existing structures ─────────────────────────────────────────
print('Querying MP...')
mp_type2, mp_type3 = set(), set()
with MPRester(API_KEY) as mpr:
    # Type 2: ternary M-Ch-Y (3 elements, 4-atom unit cells)
    for m in M3:
        results = mpr.materials.summary.search(
            elements=[m], num_elements=(3,3),
            fields=['formula_pretty','elements']
        )
        for r in results:
            elems = set(r.elements)
            elem_syms = {str(e) for e in elems}
            if (elem_syms & {'S','Se','Te'}) and (elem_syms & {'Cl','Br','I'}) and m in elem_syms:
                mp_type2.add(r.formula_pretty)

    # Type 3: A2B quaternary (4 elements)
    for a,b in itertools.product(A2, B3):
        results = mpr.materials.summary.search(
            elements=[a,b], num_elements=(4,4),
            fields=['formula_pretty','elements']
        )
        for r in results:
            elem_syms = {str(e) for e in r.elements}
            if (elem_syms & {'S','Se','Te'}) and (elem_syms & {'Cl','Br','I'}):
                mp_type3.add(r.formula_pretty)

print(f'MP type2 hits: {len(mp_type2)}: {sorted(mp_type2)[:8]}')
print(f'MP type3 hits: {len(mp_type3)}: {sorted(mp_type3)[:8]}')

# ── Get MP templates via download ─────────────────────────────────────────────
print('Downloading templates...')
templates = {}
with MPRester(API_KEY) as mpr:
    # Best type2 template: BiSI (well-studied)
    for fid in ['mp-22919','mp-675047','mp-23063']:
        try:
            s = mpr.get_structure_by_material_id(fid)
            elems = {str(e) for e in s.species}
            if len(elems)==3:
                key = 'type2_' + '_'.join(sorted(elems))
                templates[key] = s
                print(f'  type2 template: {fid} {elems}')
                break
        except: pass
    # Generic type2 template: BiSI
    results = mpr.materials.summary.search(
        formula='BiSI', fields=['material_id','structure'])
    for r in results:
        templates['type2_BiSI'] = r.structure
        print(f'  BiSI template: {r.material_id}')
        break
    # Generic type3 template: Sn2SbS2I3
    for formula in ['Sn2SbS2I3','Sn2BiS2I3']:
        results = mpr.materials.summary.search(
            formula=formula, fields=['material_id','structure'])
        for r in results:
            templates['type3_Sn2SbS2I3'] = r.structure
            print(f'  type3 template: {r.material_id} ({formula})')
            break
        if 'type3_Sn2SbS2I3' in templates: break

print(f'Templates loaded: {list(templates.keys())}')

# ── Generate type2: M(III)ChY ─────────────────────────────────────────────────
def scale_struct(s, old_elems, new_elems):
    """Scale lattice volume by ionic radii ratio."""
    vol_factor = 1.0
    for old, new in zip(old_elems, new_elems):
        if old != new:
            vol_factor *= (RADII[new]/RADII[old])
    s.scale_lattice(s.volume * vol_factor)
    return s

gen2, skip2, fail2 = [], [], []
tpl2 = templates.get('type2_BiSI')
if tpl2 is None:
    print('WARNING: no type2 template found, skipping type2')
else:
    tpl2_m = [str(e) for e in tpl2.species if str(e) in M3][0]  # Bi
    tpl2_ch = [str(e) for e in tpl2.species if str(e) in CH][0]  # S
    tpl2_y  = [str(e) for e in tpl2.species if str(e) in HAL][0] # I
    print(f'Type2 template: {tpl2_m}{tpl2_ch}{tpl2_y}')

    for m, ch, y in itertools.product(M3, CH, HAL):
        formula = f'{m}{ch}{y}'
        # Skip if in MP
        if any(formula in f for f in mp_type2):
            skip2.append(formula); continue
        # Skip if already generated
        if os.path.exists(os.path.join(TYPE2_DIR, formula+'.cif')):
            skip2.append(formula); continue
        try:
            s = tpl2.copy()
            rmap = {}
            if tpl2_m  != m:  rmap[tpl2_m]  = m
            if tpl2_ch != ch: rmap[tpl2_ch] = ch
            if tpl2_y  != y:  rmap[tpl2_y]  = y
            if rmap: s.replace_species(rmap)
            scale_struct(s,
                [tpl2_m, tpl2_ch, tpl2_y],
                [m, ch, y])
            CifWriter(s).write_file(os.path.join(TYPE2_DIR, formula+'.cif'))
            gen2.append(formula)
        except Exception as e:
            fail2.append(f'{formula}:{e}')

print(f'Type2 generated={len(gen2)} skipped(MP)={len(skip2)} failed={len(fail2)}')
if fail2: print(' ', fail2)

# ── Generate type3: A2BCh2Y3 ──────────────────────────────────────────────────
gen3, skip3, fail3 = [], [], []
tpl3 = templates.get('type3_Sn2SbS2I3')
if tpl3 is None:
    print('WARNING: no type3 template found, skipping type3')
else:
    tpl3_a  = next((str(e) for e in tpl3.species if str(e) in A2), None)
    tpl3_b  = next((str(e) for e in tpl3.species if str(e) in B3), None)
    tpl3_ch = next((str(e) for e in tpl3.species if str(e) in CH), None)
    tpl3_y  = next((str(e) for e in tpl3.species if str(e) in HAL), None)
    print(f'Type3 template: {tpl3_a}2{tpl3_b}{tpl3_ch}2{tpl3_y}3')

    for a, b, ch, y in itertools.product(A2, B3, CH, HAL):
        formula = f'{a}2{b}{ch}2{y}3'
        if any(formula in f for f in mp_type3):
            skip3.append(formula); continue
        if os.path.exists(os.path.join(TYPE3_DIR, formula+'.cif')):
            skip3.append(formula); continue
        try:
            s = tpl3.copy()
            rmap = {}
            if tpl3_a  != a:  rmap[tpl3_a]  = a
            if tpl3_b  != b:  rmap[tpl3_b]  = b
            if tpl3_ch != ch: rmap[tpl3_ch] = ch
            if tpl3_y  != y:  rmap[tpl3_y]  = y
            if rmap: s.replace_species(rmap)
            scale_struct(s,
                [tpl3_a, tpl3_b, tpl3_ch, tpl3_y],
                [a, b, ch, y])
            CifWriter(s).write_file(os.path.join(TYPE3_DIR, formula+'.cif'))
            gen3.append(formula)
        except Exception as e:
            fail3.append(f'{formula}:{e}')

print(f'Type3 generated={len(gen3)} skipped(MP)={len(skip3)} failed={len(fail3)}')
if fail3: print(' ', fail3)

print(f'\n=== TOTAL NEW CIFs: type2={len(gen2)}, type3={len(gen3)} ===')
