import os, csv, glob
from pymatgen.core import Composition

PROJECT = '/path/to/Dual-backbone-Graph-Fusion-Network'
PRED_CSV = os.path.join(PROJECT, 'predictions_large.csv')
OUT_CSV  = os.path.join(PROJECT, 'predictions_clean.csv')

def get_formula_from_cif(path):
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                if '_chemical_formula_structural' in line:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        return parts[1].strip("'\"")
    except: pass
    return None

def to_reduced(formula_str):
    try: return Composition(formula_str).reduced_formula
    except: return None

def build_formula_set(cif_dir, label):
    formulas = set()
    cifs = glob.glob(os.path.join(cif_dir, '*.cif'))
    print(f'[{label}] scanning {len(cifs)} CIFs...', flush=True)
    for i, p in enumerate(cifs):
        if i % 20000 == 0 and i > 0:
            print(f'  {i}/{len(cifs)}...', flush=True)
        f = get_formula_from_cif(p)
        if f:
            rf = to_reduced(f)
            if rf: formulas.add(rf)
    print(f'[{label}] done: {len(formulas)} unique formulas', flush=True)
    return formulas

chalco_dir = os.path.join(PROJECT, 'Data/cifs_chalcohalide')
mp_v2_dir  = os.path.join(PROJECT, 'Data/mp_bandgap_v2/cifs')
mp_formulas = build_formula_set(chalco_dir, 'chalco_train')
mp_formulas |= build_formula_set(mp_v2_dir, 'mp_bandgap_v2')
print(f'Total known MP formulas: {len(mp_formulas)}', flush=True)

def extract_comp(mid):
    parts = mid.split('_', 1)
    if len(parts) == 2:
        try: return Composition(parts[1])
        except: pass
    return None

results = []
stats = dict(total=0,direct=0,stable=0,bg=0,in_mp=0,clean=0)
with open(PRED_CSV) as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        stats['total'] += 1
        bg = float(row['bg_pred_eV'])
        if row['gt_pred'] != 'Direct': continue
        stats['direct'] += 1
        if 'Unstable' in row['eh_pred']: continue
        stats['stable'] += 1
        if not (0.5 <= bg <= 1.0): continue
        stats['bg'] += 1
        comp = extract_comp(row['material_id'])
        if comp is None: continue
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

print(f'=== FILTER SUMMARY ===')
print(f'Total:              {stats["total"]}')
print(f'Direct gap:         {stats["direct"]}')
print(f'+ Stable:           {stats["stable"]}')
print(f'+ BG 0.5-1.0eV:     {stats["bg"]}')
print(f'- In MP (excluded): {stats["in_mp"]}')
print(f'= CLEAN:            {stats["clean"]}')
print(f'Saved: {OUT_CSV}')
print('--- Top 30 (sorted by BG) ---')
results.sort(key=lambda r: float(r['bg_pred_eV']))
for r in results[:30]:
    print(f'  {r["material_id"][:55]:55s}  BG={float(r["bg_pred_eV"]):.3f}+-{float(r["bg_std_eV"]):.3f}  {r["reduced_formula"]}')
