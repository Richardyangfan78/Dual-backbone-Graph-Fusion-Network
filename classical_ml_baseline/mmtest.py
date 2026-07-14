import warnings; warnings.filterwarnings('ignore')
import numpy, pandas, sklearn, matminer
print('numpy',numpy.__version__,'pandas',pandas.__version__,'sklearn',sklearn.__version__,'matminer',matminer.__version__)
from pymatgen.core import Structure
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import DensityFeatures
import glob
f=sorted(glob.glob('/path/to/Dual-backbone-Graph-Fusion-Network/Data/Inorganic_datasets/*.cif'))[0]
s=Structure.from_file(f)
ep=ElementProperty.from_preset('magpie'); v=ep.featurize(s.composition)
print('magpie nfeat',len(v))
df=DensityFeatures(); print('density',[round(x,3) for x in df.featurize(s)])
print('OK',s.composition.reduced_formula)
