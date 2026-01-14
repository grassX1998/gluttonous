# Gluttonous - 量化交易机器学习项目

## 项目概述

这是一个基于机器学习的A股量化交易策略项目，使用 LSTM/Transformer 模型预测股票涨跌，并进行回测验证。

## 硬件环境

- **GPU**: 本地显卡用于加速训练
- **内存**: 32GB
- **存储**: 本地 SSD + NAS 网络存储
- **数据源**: `\\DXP8800PRO-A577\data\stock\gm`, 只能读，千万不能改动这里面的东西

## 项目结构

```
gluttonous/
├── pipeline/                    # 核心流水线模块
│   ├── data_cleaning/          # 数据清洗
│   │   ├── clean.py           # 数据清洗脚本
│   │   └── features.py        # 特征工程
│   ├── data_validation/        # 数据验证
│   │   └── validate.py        # 数据校验脚本
│   ├── training/              # 模型训练
│   │   └── train.py          # GPU加速训练脚本
│   ├── backtest/              # 策略回测
│   │   └── backtest.py       # 回测脚本
│   └── shared/                # 共享模块
│       ├── config.py         # 配置文件
│       └── utils.py          # 工具函数
├── .pipeline_data/            # 流水线数据目录（本地SSD）
│   ├── cleaned/              # 清洗后的数据
│   ├── features/             # 特征数据
│   ├── train/                # 训练数据
│   ├── checkpoints/          # 模型检查点
│   └── backtest_results/     # 回测结果
├── ml/                        # 旧版ML代码（已迁移）
├── archive/                   # 归档代码
└── docs/                      # 文档
```

## 数据目录结构

项目使用的行情数据存储在 NAS 上：`\\DXP8800PRO-A577\data\stock`

```
\\DXP8800PRO-A577\data\stock\
├── gm/                          # 掘金量化数据
│   ├── cfg/
│   │   └── trading_days.toml    # 交易日历配置
│   ├── meta/
│   │   ├── instruments.parquet  # 证券基础信息
│   │   └── index/               # 指数成分股
│   │       └── {date}/          # 按日期存储
│   │           └── {index_code}.toml  # 指数成分股列表
│   ├── mkline/                  # 分钟K线数据
│   │   └── {symbol}/            # 按证券代码存储
│   │       └── {date}.parquet   # 每日分钟K线
│   └── tick_l1/                 # Level1 Tick 数据
│       └── {symbol}/
│           └── {date}.parquet
└── cal/                         # 计算结果缓存
    └── sor1/                    # 策略1的缓存数据
```

## 数据格式

### 交易日历 (trading_days.toml)

```toml
trading_days = ["2024-01-02", "2024-01-03", ...]
```

### 指数成分股 ({index_code}.toml)

```toml
sec_ids = ["SHSE.600000", "SHSE.600004", ...]
```

### 分钟K线 (mkline/{symbol}/{date}.parquet)

| 字段 | 类型 | 说明 |
|------|------|------|
| symbol | str | 证券代码 (如 SHSE.600000) |
| open | f64 | 开盘价 |
| high | f64 | 最高价 |
| low | f64 | 最低价 |
| close | f64 | 收盘价 |
| volume | i64 | 成交量 |
| turnover | f64 | 成交额 |
| date | date | 日期 |
| time | time | 时间 |

## 证券代码格式

- **上交所**: `SHSE.{code}` (如 SHSE.600000, SHSE.000300)
- **深交所**: `SZSE.{code}` (如 SZSE.000001, SZSE.399975)

### 常用指数代码

| 代码 | 名称 |
|------|------|
| SZSE.000852 | 中证1000指数 |
| SZSE.000905 | 中证500指数 |

> 注意：指数数据存储在 `SZSE.` 前缀下

## 数据时间范围

- 分钟K线: 2024-06 至 2026-01
- 交易日历: 2024-01 至 2025-12

## Pipeline 工作流程

### 0. 清除缓存 (重要!)

**当修改了数据清洗或特征工程的逻辑后，必须先清除缓存再重新运行，防止数据污染：**

```powershell
# 清除所有pipeline缓存
Remove-Item -Recurse -Force ".pipeline_data\cleaned\*"
Remove-Item -Recurse -Force ".pipeline_data\features\*"
Remove-Item -Recurse -Force ".pipeline_data\train\*"
```

缓存目录说明：
| 目录 | 内容 | 何时需要清除 |
|------|------|-------------|
| `.pipeline_data/cleaned/` | 清洗后的日K线数据 | 修改 `clean.py` 后 |
| `.pipeline_data/features/` | 特征工程输出 | 修改 `features.py` 后 |
| `.pipeline_data/train/` | 训练数据集 | 修改特征或标签逻辑后 |
| `.pipeline_data/checkpoints/` | 模型检查点 | 重新训练时可保留 |

### 1. 数据清洗 (Data Cleaning)
```powershell
# 默认使用 CSI500 + CSI1000 成分股（约1500只）
python -m pipeline.data_cleaning.clean

# 或指定参数
python -m pipeline.data_cleaning.clean --start_date 2024-06-18 --end_date 2026-01-13

# 使用所有股票（不推荐，数据量太大）
python -m pipeline.data_cleaning.clean --all_stocks
```

### 2. 数据校验 (Data Validation)
```powershell
python -m pipeline.data_validation.validate
```

### 3. 特征工程 (Feature Engineering)
```powershell
python -m pipeline.data_cleaning.features
```

### 4. 模型训练 (Training)
```powershell
python -m pipeline.training.train --batch_size 512 --epochs 100
```

### 5. 策略回测 (Backtest)
```powershell
python -m pipeline.backtest.backtest --start_date 2025-01-01 --end_date 2025-12-31
```

## 技术栈

- **数据处理**: Polars (高性能DataFrame)
- **深度学习**: PyTorch + CUDA
- **特征工程**: 技术指标 (MA, RSI, MACD, Bollinger等)
- **模型**: LSTM with Attention / Transformer

## 性能优化要点

1. **GPU加速**: 使用 CUDA 和混合精度训练 (AMP)
2. **内存优化**: Polars 流式处理，numpy 内存映射
3. **I/O优化**: 数据预加载，多进程 DataLoader
4. **存储优化**: Parquet + ZSTD 压缩

---

## ⚠️ 时间序列训练规范 (重要)

### 前瞻偏差问题

在量化交易的机器学习中，**绝对禁止**使用随机划分训练/测试集的方式。这会导致：
- 模型在训练时"看到"未来数据（前瞻偏差 Lookahead Bias）
- 回测结果严重过拟合，无法反映真实交易表现
- 实盘亏损风险极高

### 正确的训练方式：滚动时间窗口验证 (Walk-Forward Validation)

```
时间轴: ----[训练窗口]---[验证]---[测试/交易]---->

Day 1-100: 训练数据
Day 101:   用训练好的模型预测，记录结果
Day 102:   将 Day 101 数据加入训练集，重新训练
Day 103:   预测...
...依此类推
```

### 实现方式

#### 方式一：每日增量训练
```python
for date in trading_days:
    # 1. 用截止到昨天的数据训练模型
    train_data = all_data[all_data['date'] < date]
    model.fit(train_data)
    
    # 2. 用今天的数据预测并交易
    today_data = all_data[all_data['date'] == date]
    predictions = model.predict(today_data)
    execute_trades(predictions)
    
    # 3. 记录真实收益（用于下一轮训练）
```

#### 方式二：固定窗口滚动训练
```python
TRAIN_WINDOW = 120  # 训练窗口：120天
VAL_WINDOW = 20     # 验证窗口：20天

for i, date in enumerate(trading_days):
    if i < TRAIN_WINDOW + VAL_WINDOW:
        continue
    
    # 训练集：前120天
    train_end = i - VAL_WINDOW
    train_start = train_end - TRAIN_WINDOW
    
    # 验证集：中间20天
    val_end = i
    val_start = train_end
    
    # 测试：当天
    model.fit(data[train_start:train_end], data[val_start:val_end])
    predict(data[i])
```

### 数据划分原则

| 类型 | 错误做法 ❌ | 正确做法 ✅ |
|------|------------|------------|
| 划分方式 | 随机打乱后按比例划分 | 严格按时间顺序划分 |
| 训练集 | 包含未来数据 | 只包含历史数据 |
| 验证集 | 随机抽取 | 训练集之后的连续时间段 |
| 测试集 | 随机抽取 | 验证集之后的连续时间段 |
| 标准化 | 用全量数据计算均值/方差 | 只用训练集计算，应用到验证/测试 |

### 关键检查点

在进行任何模型训练前，必须确认：
1. ✅ 训练数据的最大日期 < 验证数据的最小日期
2. ✅ 验证数据的最大日期 < 测试数据的最小日期
3. ✅ 特征计算不使用未来数据（如：用未来收益作为特征）
4. ✅ 标准化参数只从训练集计算
5. ✅ 回测时模型只能看到"当时"可获得的数据

