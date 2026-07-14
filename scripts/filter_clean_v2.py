import csv, os
from pymatgen.core import Composition

PROJECT = '/path/to/Dual-backbone-Graph-Fusion-Network'
FORMULA_FILE = os.path.join(PROJECT, 'Data/mp_all_formulas.txt')
PRED_CSV = os.path.join(PROJECT, 'predictions_large.csv')
OUT_CSV  = os.path.join(PROJECT, 'predictions_clean.csv')

print('Loading MP formulas...', flush=True)
mp_formulas = set()
with open(FORMULA_FILE) as f:
    for i, line in enumerate(f):
        formula = line.strip()
        if not formula:
            continue
        try:
            mp_formulas.add(Composition(formula).reduced_formula)
        except Exception:
            pass
print(f'MP known formulas: {len(mp_formulas)}', flush=True)

def extract_comp(mid):
    parts = mid.split('_', 1)
    if len(parts) == 2:
        try:
            return Composition(parts[1])
        except Exception:
            pass
    return None

results = []
stats = {'total': 0, 'direct': 0, 'stable': 0, 'bg': 0, 'in_mp': 0, 'clean': 0}

with open(PRED_CSV) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        stats['total'] += 1
        bg = float(row['bg_pred_eV'])
        if row['gt_pred'] != 'Direct':
            continue
        stats['direct'] += 1
        if 'Unstable' in row['eh_pred']:
            continue
        stats['stable'] += 1
        if not (0.5 <= bg <= 1.0):
            continue
        stats['bg'] += 1
        comp = extract_comp(row['material_id'])
        if comp is None:
            continue
        rf = comp.reduced_formula
        if rf in mp_formulas:
            stats['in_mp'] += 1
            continue
        stats['clean'] += 1
        row['reduced_formula'] = rf
        results.append(row)

out_fields = list(fieldnames) + ['reduced_formula']
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    w.writerows(results)

print('=== FILTER SUMMARY ===')
print('Total:              ' + str(stats['total']))
print('Direct gap:         ' + str(stats['direct']))
print('+ Stable:           ' + str(stats['stable']))
print('+ BG 0.5-1.0eV:     ' + str(stats['bg']))
print('- In MP (excluded): ' + str(stats['in_mp']))
print('= CLEAN:            ' + str(stats['clean']))
print('Saved: ' + OUT_CSV)
results.sort(key=lambda r: float(r['bg_pred_eV']))
print('')
print('--- All results (sorted by BG) ---')
for r in results:
    mid = r['material_id']
    bg_val = float(r['bg_pred_eV'])
    bg_std = float(r['bg_std_eV'])
    rf = r['reduced_formula']
    print('  ' + mid[:55].ljust(55) + '  BG=' + str(round(bg_val,3)) + '+-' + str(round(bg_std,3)) + '  ' + rf)
