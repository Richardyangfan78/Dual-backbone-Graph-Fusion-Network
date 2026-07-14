# CGCNN vs ALIGNN vs MODNet 对比 Benchmark

在相同数据和相同划分（80/10/10，seed=123）上训练三个模型，并在以下任务上对比：

1. **Bandgap 回归**：带隙值预测，指标 MAE (eV)
2. **Gap type 分类**：带隙类型（Direct / Indirect / Metal），指标 Accuracy、F1
3. **Stability 回归**：热力学稳定性（如 e_hull），指标 MAE

## 数据准备

- **CGCNN**：直接使用 `Data/bandgap_regression`、`Data/gap_type_classification`、`Data/stability_regression`（id_prop.csv + CIF）
- **ALIGNN**：需先跑一次数据准备（PBS 里已包含），会生成 `Data_alignn/`（id_prop 第一列为 `id.cif`，CIF 以符号链接形式存在）
- **MODNet**：直接读同一批 `Data/` 下 CIF + id_prop，脚本内用 pymatgen 加载并做 80/10/10 划分

## 训练顺序

1. **CGCNN**（已有）  
   - `pbs/train_bandgap.pbs`  
   - `pbs/train_gaptype.pbs`  
   - `pbs/train_stability.pbs`  

2. **ALIGNN**  
   - `pbs/train_alignn_bandgap.pbs`  
   - `pbs/train_alignn_gaptype.pbs`  
   - `pbs/train_alignn_stability.pbs`  

3. **MODNet**（无需 GPU）  
   - `pbs/train_modnet_bandgap.pbs`  
   - `pbs/train_modnet_gaptype.pbs`  
   - `pbs/train_modnet_stability.pbs`  

## 依赖

- 当前环境需已安装：**CGCNN**（本项目 cgcnn）、**ALIGNN**（alignn 子目录，含 jarvis-core 等）、**MODNet**（modnet 子目录，含 tensorflow、matminer 等）。
- 若未安装 ALIGNN/MODNet，可在 venv 中：  
  `pip install -e alignn`、`pip install -e modnet`（在 Chalcohalide_GNN 下）。

## 汇总结果

训练完成后在项目根目录执行：

```bash
python scripts/run_benchmark.py
```

会读取各模型输出（CGCNN 的 log、ALIGNN 的 `alignn_output/*/prediction_results_test_set.csv`、MODNet 的 `modnet_output/*/metrics.json` 与 test_predictions），生成 `benchmark_results.csv` 并在终端打印汇总表。
