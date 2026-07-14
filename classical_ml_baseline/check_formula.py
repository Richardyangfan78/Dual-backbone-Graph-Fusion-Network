import warnings; warnings.filterwarnings('ignore')
import os, csv, numpy as np
from collections import defaultdict
from pymatgen.core import Structure
DATA='/path/to/Dual-backbone-Graph-Fusion-Network/Data/Inorganic_datasets'
rows=[r for r in csv.reader(open(DATA+'/id_prop.csv')) if r]
bg={r[0]:float(r[1]) for r in rows}; gt={r[0]:int(r[2]) for r in rows}; eh={r[0]:float(r[3]) for r in rows}
F=defaultdict(list); n=0
for r in rows:
    mid=r[0]
    try:
        s=Structure.from_file(os.path.join(DATA,mid+'.cif'))
        F[s.composition.reduced_formula].append(mid); n+=1
    except Exception: pass
print('structures=%d  unique reduced formulas=%d'%(n,len(F)))
dup=[ids for ids in F.values() if len(ids)>1]
print('formulas with >=2 polymorphs: %d  (covering %d structures, %.0f%%)'%(len(dup),sum(map(len,dup)),100*sum(map(len,dup))/n))
spreads=[max(bg[i] for i in ids)-min(bg[i] for i in ids) for ids in dup]
print('within same-formula: median bandgap spread=%.3f eV, max=%.3f eV'%(np.median(spreads),np.max(spreads)))
print('same-formula groups disagreeing >0.5 eV in bandgap: %d / %d'%(sum(s>0.5 for s in spreads),len(dup)))
print('same-formula groups with conflicting gap_type: %d'%sum(len({gt[i] for i in ids})>1 for ids in dup))
print('same-formula groups with conflicting stability: %d'%sum(len({eh[i]>=0.1 for i in ids})>1 for ids in dup))
