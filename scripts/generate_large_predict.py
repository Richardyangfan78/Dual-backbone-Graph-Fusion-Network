"""
generate_large_predict.py
--------------------------
基于全部2529个训练结构做系统性chalcogen/halogen元素替换，
生成更大规模的预测集（包含oxy-chalcohalide等宽泛范围）。

策略：
  - 对每个训练结构，识别其中含有的所有chalcogen和halogen
  - 穷举所有 (new_Ch, new_Hal) 组合（3×4=12种，减去原始组合）
  - 若结构含多种chalcogen，统一替换为同一种新chalcogen
  - 若结构含多种halogen，统一替换为同一种新halogen
  - 生成新结构 + 体积缩放，跳过已在训练集/已有预测集中的组成
  - 输出到 Data/predict_large/

用法：
  python scripts/generate_large_predict.py [--dry-run]
"""
import os, sys, argparse, itertools
from collections import defaultdict
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

PROJECT  = '/path/to/Dual-backbone-Graph-Fusion-Network'
TRAIN_DIR = os.path.join(PROJECT, 'Data/cifs_chalcohalide')
OUT_DIR   = os.path.join(PROJECT, 'Data/predict_large')

CHALCOGENS = ['S', 'Se', 'Te']
HALOGENS   = ['F', 'Cl', 'Br', 'I']

# 离子半径 (pm)，与现有脚本保持一致
RADII = {
    'H':31, 'Li':76, 'Na':102, 'K':138, 'Rb':152, 'Cs':167,
    'Mg':72, 'Ca':100, 'Sr':118, 'Ba':135,
    'Cu':77, 'Ag':115, 'Au':137,
    'Zn':74, 'Cd':95, 'Hg':102,
    'Al':53, 'Ga':62, 'In':80, 'Tl':150,
    'Si':40, 'Ge':73, 'Sn':118, 'Pb':119,
    'As':58, 'Sb':76, 'Bi':103,
    'P':44, 'N':146,
    'S':184, 'Se':198, 'Te':221,
    'O':140,
    'F':133, 'Cl':181, 'Br':196, 'I':220,
    'Mn':83, 'Fe':75, 'Co':75, 'Ni':69,
    'Nb':72, 'Ta':72, 'Mo':69, 'W':66,
    'Re':63, 'Ru':68, 'Os':63, 'Rh':67, 'Ir':68,
    'Pd':86, 'Pt':80,
    'La':103, 'Ce':101, 'Pr':99, 'Nd':98, 'Sm':96,
    'Gd':94, 'Tb':92, 'Dy':91, 'Ho':90, 'Er':89,
    'Tm':88, 'Yb':87, 'Lu':86,
    'Y':90, 'Zr':72, 'Hf':71,
    'U':103, 'Th':105,
    'B':27, 'C':77, 'Cr':73, 'V':79, 'Ti':86, 'Sc':74,
    'Tc':69, 'Eu':117,
}


def get_ch_hal(elems):
    """返回结构中出现的chalcogen集合和halogen集合。"""
    chs  = elems & set(CHALCOGENS)
    hals = elems & set(HALOGENS)
    return chs, hals


def reduced_formula(comp):
    """pymatgen Composition的约简化学式，用于去重。"""
    return comp.reduced_formula


def vol_scale_factor(old_chs, old_hals, new_ch, new_hal):
    """
    基于离子半径估算体积缩放因子。
    对每种被替换的元素类型做一次 r_new/r_old 乘积（与现有脚本一致）。
    """
    factor = 1.0
    # chalcogen 替换
    for old in old_chs:
        if old != new_ch:
            if old in RADII and new_ch in RADII:
                factor *= (RADII[new_ch] / RADII[old])
    # halogen 替换
    for old in old_hals:
        if old != new_hal:
            if old in RADII and new_hal in RADII:
                factor *= (RADII[new_hal] / RADII[old])
    return factor


def load_known_formulas():
    """加载训练集和现有预测集中已知的约简化学式，用于去重。"""
    known = set()
    dirs = [
        TRAIN_DIR,
        os.path.join(PROJECT, 'Data/predict'),
        os.path.join(PROJECT, 'Data/predict_new'),
        os.path.join(PROJECT, 'Data/predict_type2'),
        os.path.join(PROJECT, 'Data/predict_type3'),
    ]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith('.cif'):
                continue
            try:
                s = Structure.from_file(os.path.join(d, f))
                known.add(reduced_formula(s.composition))
            except:
                pass
    print(f'已知组成（训练集+现有预测集）: {len(known)} 个')
    return known


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='只统计，不写文件')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('正在加载已知组成...')
    known_formulas = load_known_formulas()

    train_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.cif')]
    print(f'训练结构总数: {len(train_files)}')

    stats = defaultdict(int)
    generated_formulas = set()   # 本次运行中已生成的，防重复

    for i, fname in enumerate(train_files):
        if i % 200 == 0:
            print(f'  处理进度: {i}/{len(train_files)}, 已生成: {stats["generated"]}')

        try:
            s = Structure.from_file(os.path.join(TRAIN_DIR, fname))
        except Exception as e:
            stats['load_fail'] += 1
            continue

        elems = {str(e) for e in s.composition.elements}
        old_chs, old_hals = get_ch_hal(elems)

        # 至少要有一个可替换位
        if not old_chs and not old_hals:
            stats['no_ch_hal'] += 1
            continue

        # 若无chalcogen，则不替换chalcogen（只替换halogen）
        ch_targets = CHALCOGENS if old_chs else [None]
        hal_targets = HALOGENS  if old_hals else [None]

        for new_ch, new_hal in itertools.product(ch_targets, hal_targets):
            # 若替换后与原始完全相同则跳过
            ch_same  = (new_ch  is None) or (old_chs  == {new_ch})
            hal_same = (new_hal is None) or (old_hals == {new_hal})
            if ch_same and hal_same:
                stats['same_as_original'] += 1
                continue

            # 构建替换映射
            replace_map = {}
            if new_ch is not None:
                for old in old_chs:
                    if old != new_ch:
                        replace_map[old] = new_ch
            if new_hal is not None:
                for old in old_hals:
                    if old != new_hal:
                        replace_map[old] = new_hal

            if not replace_map:
                stats['same_as_original'] += 1
                continue

            try:
                new_s = s.copy()
                new_s.replace_species(replace_map)
                formula = reduced_formula(new_s.composition)

                # 去重
                if formula in known_formulas or formula in generated_formulas:
                    stats['skipped_duplicate'] += 1
                    continue

                # 体积缩放
                factor = vol_scale_factor(
                    old_chs  if old_chs  else set(),
                    old_hals if old_hals else set(),
                    new_ch  if new_ch  is not None else list(old_chs)[0] if old_chs else '',
                    new_hal if new_hal is not None else list(old_hals)[0] if old_hals else '',
                )
                if factor != 1.0:
                    new_s.scale_lattice(new_s.volume * factor)

                # 文件名：原始mp-id + 新组成
                base = fname.replace('.cif', '')
                safe_formula = formula.replace(' ', '')
                out_fname = f'{base}_{safe_formula}.cif'
                out_path = os.path.join(OUT_DIR, out_fname)

                if not args.dry_run:
                    CifWriter(new_s).write_file(out_path)

                generated_formulas.add(formula)
                stats['generated'] += 1

            except Exception as e:
                stats['gen_fail'] += 1

    print()
    print('=== 生成完成 ===')
    print(f'  成功生成:       {stats["generated"]}')
    print(f'  跳过（重复）:    {stats["skipped_duplicate"]}')
    print(f'  跳过（无变化）:  {stats["same_as_original"]}')
    print(f'  加载失败:        {stats["load_fail"]}')
    print(f'  生成失败:        {stats["gen_fail"]}')
    print(f'  输出目录: {OUT_DIR}')
    if args.dry_run:
        print('  (dry-run模式，未写入文件)')

if __name__ == '__main__':
    main()
