#!/bin/bash
#SBATCH --job-name=build_mp_forms
#SBATCH --account=a131
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/build_mp_forms_%j.out

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
OUT=/path/to/Dual-backbone-Graph-Fusion-Network/Data/mp_all_formulas.txt

echo 'Step1: grep chalco training CIFs...'
grep -rh '_chemical_formula_structural' ${PROJECT}/Data/cifs_chalcohalide/ | awk '{print $2}' | tr -d "'\"" | sort -u > ${OUT}
echo "Chalco: $(wc -l < ${OUT}) formulas"

echo 'Step2: grep mp_bandgap_v2 CIFs...'
grep -rh '_chemical_formula_structural' ${PROJECT}/Data/mp_bandgap_v2/cifs/ | awk '{print $2}' | tr -d "'\"" >> ${OUT}
sort -u ${OUT} -o ${OUT}
echo "Total unique: $(wc -l < ${OUT}) formulas"
echo 'Done.'
