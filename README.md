# Material Feature Fusion

本项目用于研究不同原子描述符在机器学习力场中的作用，并据此设计可复现的特征融合方案。当前代码处于第一阶段：建立数据接口、描述符生成流程和 SchNetPack SchNet 基线。

## 研究主线

论文不以复杂网络结构为主要创新，而以“特征分析 -> 融合设计 -> 统一模型验证”为主线：

1. 分别评估元素、局部结构和静电相关描述符在能量、力和应力预测中的表现。
2. 分析各描述符的互补性、冗余性、尺度敏感性、噪声鲁棒性和数据效率。
3. 根据分析结果选择并融合描述符，而不是预先假定某一种融合方式有效。
4. 使用同一个 SchNet 基础模型进行基线、消融和融合对比。
5. 将门控权重、分支消融和梯度/置换分析作为解释手段，用来说明融合特征在什么结构环境下发挥作用。

描述符作为外部生成的辅助特征输入，不要求 ACSF、SOAP 或局部 Coulomb 特征参与能量对坐标的自动微分。论文中需要明确：该设置属于固定辅助表征下的能量与力联合预测框架，不宣称描述符分支本身提供严格的保守力导数。

## 描述符设计

当前预留四类特征：

- 元素特征：SchNet 的可学习 `Embedding(Z)`。
- 结构特征一：DScribe ACSF，强调径向和角向局部统计。
- 结构特征二：DScribe SOAP，描述连续局部原子密度和环境相似性。
- 电荷相关特征：固定宽度的局部 Coulomb interaction descriptor，使用邻域中的 `Zi*Zj/r`，而不是依赖体系原子数的全局 Coulomb Matrix。

标准 Coulomb Matrix 的维度依赖原子数，且不适合直接处理周期性材料。因此本项目使用按距离排序、截断或零填充的局部版本。它更准确的名称是静电相互作用描述符，而不是直接的真实电荷标签。

DScribe 生成的特征会写入新的 ASE SQLite 数据库，原始数据库保持不变。特征数据库同时保存配置元数据，保证特征维度和超参数可追溯。

## 数据格式

输入是 ASE SQLite Database。每行对应一个 `ase.Atoms` 构型，能量、力和应力存储在 `row.data` 中：

```text
row.data["energy"]  -> scalar, eV
row.data["forces"]  -> (natoms, 3), eV/Angstrom
row.data["stress"]  -> optional stress tensor
```

完整约定见 [ASE_DB_FORMAT.md](ASE_DB_FORMAT.md)。训练前应检查：

- 力的形状是否为 `(natoms, 3)`；
- 周期性结构是否正确保存 `cell` 和 `pbc`；
- 是否存在重复构型；
- 数据划分是否按材料/轨迹分组，避免同一构型或相邻轨迹泄漏到验证集和测试集。

## 目录结构

```text
material_feature_fusion/
  data.py          # row.data 读取、校验和数据库摘要
  descriptors.py   # DScribe 和局部 Coulomb 描述符
  fusion.py        # 描述符投影、拼接和门控分析模块
  keys.py          # 统一数据键
scripts/
  inspect_db.py
  prepare_descriptors.py
  train_schnet.py
requirements.txt
```

## Conda 环境

建议使用独立环境：

```bash
conda create -n material-feature-fusion python=3.11
conda activate material-feature-fusion
python -m pip install -r requirements.txt
```

实际训练前请根据 CUDA 版本安装匹配的 PyTorch。若使用 CPU 或 Apple Silicon，可以直接使用 conda/pip 中可用的 PyTorch 构建版本。

## 基础用法

检查数据库：

```bash
python scripts/inspect_db.py data/example.db
```

生成描述符数据库：

```bash
python scripts/prepare_descriptors.py \
  data/raw/example.db \
  data/processed/example_descriptors.db
```

训练 SchNet 基线：

```bash
python scripts/train_schnet.py \
  data/processed/example_descriptors.db \
  --output-dir training_runs/example \
  --max-epochs 100
```

当前训练入口首先验证 SchNetPack 的标准能量/力路径。描述符键已经支持加载，但描述符注入 SchNet 原子表示的完整模块将在下一阶段加入，以便先固定数据字段、训练分割和基线评价协议。

## 后续实验协议

计划采用以下顺序：

1. SchNet + 元素 Embedding 基线。
2. SchNet + ACSF、SOAP、局部 Coulomb 单特征实验。
3. 两两组合和全特征组合。
4. 拼接、投影后拼接、归一化融合和门控融合消融。
5. 随机测试、低数据量测试和结构/化学组成 OOD 测试。
6. 以力 MAE 为主指标，同时报告能量、应力、参数量和推理成本。

所有主要结果至少使用三个随机种子，并保存配置、数据划分、特征参数和原始预测结果。

## 基础检查

```bash
python -m compileall material_feature_fusion scripts tests
pytest -q
ruff check material_feature_fusion scripts tests
```
