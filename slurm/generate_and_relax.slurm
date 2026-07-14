#!/bin/bash
#SBATCH --job-name=gen_relax
#SBATCH --account=a131
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gen_relax_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/gen_relax_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
source ${PROJECT}/venv2/bin/activate
export PYTHONPATH=${PROJECT}/multitask:${PYTHONPATH}
export PYTHONUNBUFFERED=1
export HF_HOME=${PROJECT}/.cache
export MACE_CACHE=${PROJECT}/.cache

echo "=== Step 1: Generate missing structures === $(date)"
python ${PROJECT}/scripts/generate_missing.py

echo ""
echo "=== Step 2: Count generated ==="
ls ${PROJECT}/Data/predict_new_generated/*.cif 2>/dev/null | wc -l

echo ""
echo "=== Step 3: Relax with MACE-MP-0 === $(date)"
cd ${PROJECT}/multitask
python ${PROJECT}/scripts/relax_generated.py \
    ${PROJECT}/Data/predict_new_generated \
    ${PROJECT}/Data/predict_new_relaxed \
    --device cuda --fmax 0.05 --max-steps 300

echo ""
echo "=== Step 4: Add to predict_final === $(date)"
python - << PYEOF
import os, shutil
from pymatgen.core import Composition, Element
from itertools import product as iproduct

SRC = "${PROJECT}/Data/predict_new_relaxed"
DST = "${PROJECT}/Data/predict_final"
CH = {"S","Se","Te"}
X  = {"F","Cl","Br","I"}

def get_all_pos_ox(el_str):
    try:
        states = [int(o) for o in Element(el_str).oxidation_states if o > 0]
        return states if states else [1,2,3,4]
    except: return [1,2,3,4]

def is_balanced(formula):
    try:
        comp = Composition(formula)
        anion_q = sum(({"S":-2,"Se":-2,"Te":-2,"F":-1,"Cl":-1,"Br":-1,"I":-1}.get(str(e),0)*int(comp[e])) for e in comp.elements)
        needed = -anion_q
        m_els = [(str(e),int(comp[e])) for e in comp.elements if str(e) not in CH and str(e) not in X]
        if not m_els: return False
        ox_opts = [[(ox,cnt) for ox in get_all_pos_ox(el)] for el,cnt in m_els]
        for combo in iproduct(*ox_opts):
            if abs(sum(ox*cnt for ox,cnt in combo)-needed)<0.01: return True
        return False
    except: return True

added = 0
skipped_bal = 0
skipped_exist = 0
if os.path.exists(SRC):
    for fname in os.listdir(SRC):
        if not fname.endswith(".cif"): continue
        formula = fname.replace(".cif","")
        dst_path = os.path.join(DST, fname)
        if os.path.exists(dst_path):
            skipped_exist += 1; continue
        if not is_balanced(formula):
            skipped_bal += 1; continue
        shutil.copy2(os.path.join(SRC, fname), dst_path)
        added += 1
print("Added to predict_final: " + str(added))
print("Skipped (unbalanced): " + str(skipped_bal))
print("Skipped (existing): " + str(skipped_exist))
print("predict_final total: " + str(len([f for f in os.listdir(DST) if f.endswith(".cif")])))
PYEOF

echo "=== Done === $(date)"
