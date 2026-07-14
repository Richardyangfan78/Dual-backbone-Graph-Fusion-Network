#!/bin/bash
#SBATCH --job-name=filter_v2
#SBATCH --account=a131
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/Dual-backbone-Graph-Fusion-Network/logs/filter_v2_%j.out
#SBATCH --error=/path/to/Dual-backbone-Graph-Fusion-Network/logs/filter_v2_%j.err

PROJECT=/path/to/Dual-backbone-Graph-Fusion-Network
echo '[Step1] Extracting MP formulas via grep...'
grep -h '_chemical_formula_structural'     ${PROJECT}/Data/cifs_chalcohalide/*.cif     ${PROJECT}/Data/mp_bandgap_v2/cifs/*.cif     | awk '{print $2}' | tr -d "'\"" | sort -u > /tmp/mp_raw_formulas.txt
echo "Raw formula lines: $(wc -l < /tmp/mp_raw_formulas.txt)"

source ${PROJECT}/venv/bin/activate
python3 ${PROJECT}/scripts/filter_clean_v2.py
