# `painn.py` 命令行参数说明

`painn.py` 是本项目的统一命令行入口。建议在 Conda 环境
`material-feature-fusion` 中执行：

```bash
conda activate material-feature-fusion
python painn.py --help
```

## 命令总览

```text
python painn.py inspect DATABASE [OPTIONS]
python painn.py prepare INPUT_DB OUTPUT_DB [OPTIONS]
python painn.py train DATABASE [OPTIONS]
```

为了兼容旧用法，下面的命令等价于 `train` 子命令：

```bash
python painn.py DATABASE --max-epochs 5
```

## `inspect`

检查 `row.data` 中的能量、力和可选应力，并输出数据库摘要。执行时会验证
数据库中的每一行；`--limit` 只限制检查和摘要的行数。

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `DATABASE` | `Path` | 必填 | ASE SQLite 数据库路径 |
| `--limit` | `int` | `None` | 最多检查多少条记录；不填写则检查全部记录 |

示例：

```bash
python painn.py inspect ethanol.db --limit 100
```

## `prepare`

从输入数据库生成一个新的特征数据库。输入数据库不会被修改。每个选中的
特征会以逐原子数组写入 `row.data`，同时在数据库 metadata 中写入特征参数、
元素种类、单位和源数据库信息。

### 数据库参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `INPUT_DB` | `Path` | 必填 | 原始 ASE SQLite 数据库 |
| `OUTPUT_DB` | `Path` | 必填 | 输出的特征数据库；必须不同于输入路径 |
| `--overwrite` | flag | 关闭 | 输出文件已存在时覆盖它；不指定则报错 |

### 特征参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--features` | 一个或多个枚举值 | `acsf soap local_coulomb` | 要生成的特征，可选 `acsf`、`soap`、`local_coulomb` |
| `--cutoff` | `float` | `5.0` | 邻域截断半径，单位 Angstrom |
| `--soap-n-max` | `int` | `6` | SOAP 径向基函数截断数 |
| `--soap-l-max` | `int` | `4` | SOAP 角向阶数截断 |
| `--soap-sigma` | `float` | `0.5` | SOAP 高斯展宽参数 |
| `--local-coulomb-neighbors` | `int` | `16` | 局部 Coulomb 特征保留的近邻数；不足时补零 |
| `--acsf-g2 ETA RS` | 两个 `float`，可重复 | 内置默认值 | ACSF 径向 G2 参数；每次出现增加一组 `(eta, Rs)` |
| `--acsf-g4 ETA ZETA LAMBDA` | 三个 `float`，可重复 | 内置默认值 | ACSF 角向 G4 参数；每次出现增加一组 `(eta, zeta, lambda)` |

示例：

```bash
python painn.py prepare ethanol.db data/processed/ethanol_acsf_soap.db \
  --features acsf soap \
  --cutoff 5.0 \
  --soap-n-max 6 \
  --soap-l-max 4 \
  --overwrite
```

## `train`

训练命令支持 PaiNN 和项目自有的可替换特征 SchNet。外部 ACSF、SOAP 和局部
Coulomb 特征都是由 NumPy/DScribe 生成的固定输入，不对坐标求导；因此这些
特征分支不会直接贡献严格的保守力导数。

### 数据来源和特征替换

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `DATABASE` | `Path` | 必填 | ASE SQLite 数据库 |
| `--architecture` | `painn` 或 `schnet` | `painn` | 选择 PaiNN 或可替换特征 SchNet 主干 |
| `--feature-mode` | 枚举值 | `atomic_numbers` | 初始逐原子特征来源：`atomic_numbers` 使用原子序数 Embedding；`dataset` 从 `row.data` 读取；`realtime` 在模型输入阶段实时生成 |
| `--features` | 一个或多个枚举值 | `acsf` | `dataset`/`realtime` 使用的特征分支 |
| `--descriptor-key` | `str` | `None` | 单特征兼容别名；指定后优先使用该名称，并覆盖 `--features` |
| `--fusion` | `concat` 或 `gated_sum` | `concat` | 多特征处理方式：投影后拼接，或投影后门控加权求和 |

`atomic_numbers` 模式不使用 `--features`，仍保留参数只是为了让不同运行的
JSON 结构一致。单特征模式直接映射到原子表示；多特征模式先分别进行
`LayerNorm + Linear` 投影，再执行所选融合。

### 特征超参数

训练命令同样支持以下参数，它们只在 `dataset` 生成实时特征或准备缓存特征
时生效：

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--cutoff` | `float` | `5.0` | 特征和邻居列表的截断半径，单位 Angstrom |
| `--soap-n-max` | `int` | `6` | SOAP 径向截断 |
| `--soap-l-max` | `int` | `4` | SOAP 角向截断 |
| `--soap-sigma` | `float` | `0.5` | SOAP 展宽参数 |
| `--local-coulomb-neighbors` | `int` | `16` | 局部 Coulomb 固定宽度 |
| `--acsf-g2 ETA RS` | 两个 `float`，可重复 | 内置默认值 | ACSF G2 参数 |
| `--acsf-g4 ETA ZETA LAMBDA` | 三个 `float`，可重复 | 内置默认值 | ACSF G4 参数 |

### 数据划分和批处理

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--num-train` | `float` | `0.8` | 训练集比例；大于 1 的整数值表示绝对条数 |
| `--num-val` | `float` | `0.1` | 验证集比例或绝对条数 |
| `--num-test` | `float` | `0.1` | 测试集比例或绝对条数 |
| `--batch-size` | `int` | `16` | 每个 batch 的构型数量 |
| `--num-workers` | `int` | `0` | DataLoader 工作进程数；实时 DScribe 模式建议先使用 `0` |
| `--max-rows` | `int` | `None` | 只复制输入数据库前 N 条构型后训练，适合冒烟测试；不填写则使用全库 |

例如，20 条数据按 12/4/4 划分：

```bash
--max-rows 20 --num-train 12 --num-val 4 --num-test 4
```

### 模型参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--n-atom-basis` | `int` | `64` | 原子隐表示维度 |
| `--n-interactions` | `int` | `6` | 交互模块数量 |
| `--n-rbf` | `int` | `20` | 距离径向基函数数量 |
| `--cutoff` | `float` | `5.0` | SchNet/PaiNN 邻居列表截断半径 |

### 优化和训练参数

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--max-epochs` | `int` | `100` | 最大训练 epoch 数 |
| `--lr` | `float` | `1e-4` | AdamW 学习率 |
| `--energy-weight` | `float` | `0.01` | 能量均方误差损失权重 |
| `--forces-weight` | `float` | `0.99` | 力均方误差损失权重 |
| `--stress-weight` | `float` | `0.0` | 应力损失权重；大于 0 时必须有 `row.data['stress']` |
| `--seed` | `int` | `2026` | Python、NumPy、PyTorch 随机种子 |
| `--device` | `cpu`、`cuda` 或 `mps` | 自动选择 | PyTorch Lightning 加速设备 |
| `--run-test` | flag | 关闭 | 训练后运行测试集；开启后会关闭 Lightning inference mode，使力响应层可以对坐标求导 |

### 输出目录和文件

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--output-dir` | `Path` | 自动生成 | 显式指定训练目录；不指定时自动生成带实验信息的目录 |

默认目录格式为：

```text
training_runs/YYYYMMDD_HHMMSS_microseconds_DATASET_ARCH_FEATURES_MODE_FUSION/
```

例如：

```text
training_runs/20260717_153012_123456_ethanol_schnet_acsf_realtime_concat/
```

默认目录会包含：

| 文件 | 内容 |
| --- | --- |
| `hyperparameters.json` | 本次 CLI 参数、派生特征列表、时间、输出路径、split 和 lock 路径 |
| `best_model` | SchNetPack 保存的验证集 `val_loss` 最优推理模型，已包含后处理 |
| `split.npz` | SchNetPack 保存的数据划分 |
| `splitting.lock` | 本次数据划分使用的进程锁，不再生成到启动命令所在目录 |
| `training_subset.db` | 使用 `--max-rows` 时生成的训练子集 |
| `schnetpack_input.db` | `realtime` 模式下生成的 SchNetPack 兼容副本 |
| `lightning_logs/` 和 `.ckpt` | Lightning 训练日志及恢复训练用的原始 checkpoint |

程序会在数据模块 setup、训练和可选测试阶段临时将工作目录切换到本次训练
目录。因此 SchNetPack 2.1.1 硬编码的相对路径 `splitting.lock` 会直接生成在
训练目录中，不需要每次手动删除启动目录下的 lock 文件。

## 推荐用法

### 原始 SchNet

```bash
python painn.py train ethanol.db \
  --architecture schnet \
  --feature-mode atomic_numbers \
  --max-epochs 100
```

### ACSF 替换特征

```bash
python painn.py train ethanol.db \
  --architecture schnet \
  --feature-mode realtime \
  --features acsf \
  --max-epochs 100
```

### 小批量功能检查

```bash
python painn.py train ethanol.db \
  --architecture schnet \
  --feature-mode realtime \
  --features acsf \
  --max-rows 20 \
  --num-train 12 --num-val 4 --num-test 4 \
  --batch-size 4 \
  --n-atom-basis 16 \
  --n-interactions 2 \
  --n-rbf 8 \
  --max-epochs 5 \
  --device cpu
```
