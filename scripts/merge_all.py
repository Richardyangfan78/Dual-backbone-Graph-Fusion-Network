import csv, os, shutil
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifWriter

PROJECT = '/path/to/Dual-backbone-Graph-Fusion-Network'
OUT_CSV = os.path.join(PROJECT, 'predictions_all.csv')
OUT_CIF = os.path.join(PROJECT, 'Data/all_cifs')
os.makedirs(OUT_CIF, exist_ok=True)

def rf(formula):
    try: return Composition(formula).reduced_formula
    except: return formula

def parse_id(mid):
    if '_' in mid and mid.startswith('mp-'):
        return rf(mid.split('_', 1)[1])
    return rf(mid)

def screen(row):
    try:
        bg = float(row['bg_pred_eV'])
        gt = row['gt_pred']
        eh = row['eh_pred']
        return gt == 'Direct' and 0.5 <= bg <= 1.0 and 'Stable' in eh and 'Unstable' not in eh
    except: return False

rows_out = []
seen = set()

LARGE_DIR = os.path.join(PROJECT, 'Data/predict_large')
SN_DIR    = os.path.join(PROJECT, 'sn_structures')
VASP_DIR  = os.path.join(PROJECT, 'Data/chalcohalide POSCAR')

# Source 1: predictions_large.csv (17390)
print('Loading predictions_large.csv...', flush=True)
with open(os.path.join(PROJECT, 'predictions_large.csv')) as f:
    for row in csv.DictReader(f):
        mid = row['material_id']
        formula = parse_id(mid)
        if formula in seen: continue
        seen.add(formula)
        bg = float(row['bg_pred_eV'])
        eh = row['eh_pred']
        rows_out.append({
            'formula':     formula,
            'bg_type':     row['gt_pred'],
            'bg_eV':       round(bg, 4),
            'bg_std_eV':   round(float(row['bg_std_eV']), 4),
            'ehull':       'Stable' if ('Stable' in eh and 'Unstable' not in eh) else 'Unstable',
            'screen_pass': 'Yes' if screen(row) else 'No',
            '_cif_src':    os.path.join(LARGE_DIR, mid + '.cif'),
        })
print(f'  After large: {len(rows_out)}', flush=True)

# Source 2: predictions_dual_backbone.csv (162)
print('Loading predictions_dual_backbone.csv...', flush=True)
with open(os.path.join(PROJECT, 'predictions_dual_backbone.csv')) as f:
    for row in csv.DictReader(f):
        mid = row['material_id']
        formula = rf(mid)
        if formula in seen: continue
        seen.add(formula)
        bg = float(row['bg_pred_eV'])
        eh = row['eh_pred']
        rows_out.append({
            'formula':     formula,
            'bg_type':     row['gt_pred'],
            'bg_eV':       round(bg, 4),
            'bg_std_eV':   round(float(row['bg_std_eV']), 4),
            'ehull':       'Stable' if ('Stable' in eh and 'Unstable' not in eh) else 'Unstable',
            'screen_pass': 'Yes' if screen(row) else 'No',
            '_cif_src':    os.path.join(VASP_DIR, formula + '.vasp'),
        })
print(f'  After dual_backbone: {len(rows_out)}', flush=True)

# Source 3: predictions_sn.csv (54)
print('Loading predictions_sn.csv...', flush=True)
with open(os.path.join(PROJECT, 'predictions_sn.csv')) as f:
    for row in csv.DictReader(f):
        mid = row['material_id']
        formula = rf(mid)
        if formula in seen: continue
        seen.add(formula)
        bg = float(row['bg_pred_eV'])
        eh = row['eh_pred']
        rows_out.append({
            'formula':     formula,
            'bg_type':     row['gt_pred'],
            'bg_eV':       round(bg, 4),
            'bg_std_eV':   round(float(row['bg_std_eV']), 4),
            'ehull':       'Stable' if ('Stable' in eh and 'Unstable' not in eh) else 'Unstable',
            'screen_pass': 'Yes' if screen(row) else 'No',
            '_cif_src':    os.path.join(SN_DIR, formula + '.cif'),
        })
print(f'  After sn: {len(rows_out)}', flush=True)

# Write CSV
fields = ['formula','bg_type','bg_eV','bg_std_eV','ehull','screen_pass']
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow({k: r[k] for k in fields})
print(f'CSV written: {OUT_CSV}', flush=True)

# Collect CIFs
print('\nCollecting CIF files...', flush=True)
copied = converted = skipped = 0
for i, row in enumerate(rows_out):
    if i % 2000 == 0 and i > 0:
        print(f'  {i}/{len(rows_out)} done...', flush=True)
    formula = row['formula']
    cif_out = os.path.join(OUT_CIF, formula + '.cif')
    src = row['_cif_src']
    if src.endswith('.cif') and os.path.exists(src):
        shutil.copy2(src, cif_out)
        copied += 1
    elif src.endswith('.vasp') and os.path.exists(src):
        try:
            CifWriter(Structure.from_file(src)).write_file(cif_out)
            converted += 1
        except:
            skipped += 1
    else:
        skipped += 1

total = len(rows_out)
passed = sum(1 for r in rows_out if r['screen_pass'] == 'Yes')
print(f'\n=== SUMMARY ===')
print(f'Total unique: {total}')
print(f'  large:          {sum(1 for r in rows_out if r["_cif_src"].startswith(LARGE_DIR))}')
print(f'  dual_backbone:  {sum(1 for r in rows_out if r["_cif_src"].startswith(VASP_DIR))}')
print(f'  sn:             {sum(1 for r in rows_out if r["_cif_src"].startswith(SN_DIR))}')
print(f'Screen PASS:  {passed}')
print(f'CIF copied:   {copied}')
print(f'VASP->CIF:    {converted}')
print(f'Skipped:      {skipped}')
