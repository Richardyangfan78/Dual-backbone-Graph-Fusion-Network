import warnings; warnings.filterwarnings('ignore')
import os, csv, time, pickle
import numpy as np, pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty, Stoichiometry, ValenceOrbital, IonProperty,
    TMetalFraction, BandCenter)
from matminer.featurizers.structure import (
    DensityFeatures, GlobalSymmetryFeatures, StructuralComplexity)

DATA = '/path/to/Dual-backbone-Graph-Fusion-Network/Data/Inorganic_datasets'
OUT  = '/path/to/Dual-backbone-Graph-Fusion-Network/classical_ml_baseline'

t0 = time.time()
rows = [r for r in csv.reader(open(os.path.join(DATA, 'id_prop.csv'))) if r]
ids = [r[0] for r in rows]
bg = np.array([float(r[1]) for r in rows])
gt = np.array([int(r[2]) for r in rows])
eh = np.array([float(r[3]) for r in rows])
print('n samples', len(ids), flush=True)

structs, keep = [], []
for k, i in enumerate(ids):
    p = os.path.join(DATA, i + '.cif')
    try:
        structs.append(Structure.from_file(p)); keep.append(k)
    except Exception as e:
        print('load fail', i, e)
keep = np.array(keep)
bg, gt, eh = bg[keep], gt[keep], eh[keep]
ok_ids = [ids[k] for k in keep]
comps = [s.composition for s in structs]
print('loaded structures', len(structs), 'in %.1fs' % (time.time() - t0), flush=True)

comp_f = MultipleFeaturizer([
    ElementProperty.from_preset('magpie'),
    Stoichiometry(),
    ValenceOrbital(),
    IonProperty(fast=True),
    TMetalFraction(),
    BandCenter(),
])
struct_f = MultipleFeaturizer([
    DensityFeatures(),
    GlobalSymmetryFeatures(),
    StructuralComplexity(),
])
for f in (comp_f, struct_f):
    f.set_n_jobs(8)

t1 = time.time()
Xc = comp_f.featurize_many(comps, ignore_errors=True, pbar=False)
print('comp feats done %.1fs' % (time.time() - t1), flush=True)
t2 = time.time()
Xs = struct_f.featurize_many(structs, ignore_errors=True, pbar=False)
print('struct feats done %.1fs' % (time.time() - t2), flush=True)

dfc = pd.DataFrame(Xc, columns=comp_f.feature_labels())
dfs = pd.DataFrame(Xs, columns=struct_f.feature_labels())
df = pd.concat([dfc, dfs], axis=1)
df = df.apply(pd.to_numeric, errors='coerce')      # drop string cols (e.g. crystal_system)
nuniq = df.nunique(dropna=True)
df = df.loc[:, nuniq > 1]                           # drop constant / all-NaN cols
# dedupe any duplicate column names
df = df.loc[:, ~df.columns.duplicated()]
print('feature matrix', df.shape, '| NaN cells', int(df.isna().sum().sum()), flush=True)

out = {'ids': ok_ids, 'X': df.values.astype('float64'), 'cols': list(df.columns),
       'bg': bg, 'gt': gt, 'eh': eh}
with open(os.path.join(OUT, 'features.pkl'), 'wb') as fh:
    pickle.dump(out, fh)
df.to_csv(os.path.join(OUT, 'features.csv'), index=False)
print('SAVED features.pkl (%d x %d)  total %.1fs' % (df.shape[0], df.shape[1], time.time() - t0), flush=True)
