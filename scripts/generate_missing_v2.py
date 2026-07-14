"""Generate only the previously-failed F-containing compounds via Cl→F substitution."""
import os, itertools
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

PROJECT  = '/path/to/Dual-backbone-Graph-Fusion-Network'
PRED_DIR = os.path.join(PROJECT, 'Data/predict')
OUT_DIR  = os.path.join(PROJECT, 'Data/predict_new')

A_ELEMENTS = ['Li','Na','K','Rb','Cs','Ag','Cu']
B_ELEMENTS = ['Bi','Sb','As','In','Ga']
X_ELEMENTS = ['S','Se','Te']

RADII = {
    'Li':76,'Na':102,'K':138,'Rb':152,'Cs':167,'Ag':115,'Cu':77,
    'Bi':103,'Sb':76,'As':58,'In':80,'Ga':62,
    'S':184,'Se':198,'Te':221,
    'F':133,'Cl':181,'Br':196,'I':220,
}

generated, failed = [], []

for a, b, x in itertools.product(A_ELEMENTS, B_ELEMENTS, X_ELEMENTS):
    formula = f'{a}{b}{x}F2'
    out_path = os.path.join(OUT_DIR, formula+'.cif')
    if os.path.exists(out_path):
        continue  # already done

    # Use Cl template (closest halogen), then scale for F
    tpl_path = None
    for ta in [a] + [e for e in A_ELEMENTS if e != a]:
        for tb in [b] + [e for e in B_ELEMENTS if e != b]:
            p = os.path.join(PRED_DIR, f'{ta}{tb}{x}Cl2.cif')
            if os.path.exists(p):
                tpl_path = p; ta_use = ta; tb_use = tb; break
        if tpl_path: break

    if not tpl_path:
        failed.append(formula); continue

    try:
        s = Structure.from_file(tpl_path)
        replace_map = {'Cl': 'F'}
        if ta_use != a: replace_map[ta_use] = a
        if tb_use != b: replace_map[tb_use] = b
        s.replace_species(replace_map)

        # Scale: account for all substitutions
        vol_scale = 1.0
        if ta_use != a: vol_scale *= (RADII[a]/RADII[ta_use])
        if tb_use != b: vol_scale *= (RADII[b]/RADII[tb_use])
        vol_scale *= (RADII['F']/RADII['Cl'])**2   # 2 F per formula unit
        # cube root for linear scale, then cube for volume
        s.scale_lattice(s.volume * vol_scale)

        CifWriter(s).write_file(out_path)
        generated.append(formula)
        print(f'  OK: {formula}  (tpl={ta_use}{tb_use}{x}Cl2)')
    except Exception as e:
        failed.append(f'{formula}({e})')

print(f'\nGenerated: {len(generated)}  Failed: {len(failed)}')
if failed: print('Failed:', failed)
