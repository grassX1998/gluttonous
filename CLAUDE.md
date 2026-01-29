# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指引。

## 🌏 语言设置

**重要**: 始终使用中文与用户交流。本项目是中文环境的A股量化交易系统，所有交互、解释、建议都应使用中文。

## 项目概述

Gluttonous 是一个基于机器学习的 A 股量化选股系统，支持多种模型预测 5 日涨跌，并进行 Walk-Forward 回测验证。

**当前版本**: v0.3
**技术栈**: Polars (数据处理)、PyTorch + CUDA (深度学习)、LightGBM (梯度提升)

**支持的模型框架**:
- **LSTM 框架** (`src/lstm/`): 深度学习时序模型，支持扩展窗口策略
- **多模型框架** (`src/models/`): LightGBM、MLP、Ensemble 集成模型

## 数据源

**NAS 存储**: `\\DXP8800PRO-A577\data\stock\gm\`

⚠️ **严重警告**: 此数据源为只读，绝对不能修改该目录中的任何内容！

数据结构：
- `cfg/trading_days.toml` - 交易日历
- `meta/instruments.parquet` - 股票元数据
- `meta/index/{date}/{index_code}.toml` - 指数成分股
- `mkline/{symbol}/{date}.parquet` - 分钟级 K 线数据 (2024-06 至 2026-01)
- `tick_l1/{symbol}/{date}.parquet` - Level 1 Tick 数据

股票代码格式：
- 上交所：`SHSE.{code}` (如 SHSE.600000)
- 深交所：`SZSE.{code}` (如 SZSE.000001)

常用指数：
- SHSE.000852 - 中证1000
- SHSE.000905 - 中证500
- SHSE.000300 - 沪深300

### AKShare 数据源（免费替代方案）

**NAS 存储**: `\\DXP8800PRO-A577\data\stock\akshare\`

⚠️ **严重警告**: 此数据源为只读，绝对不能修改该目录中的任何内容！

AKShare 提供免费的 A 股数据，作为掘金付费数据的补充。数据采集模块：`pipeline/data_collection/akshare_api.py`

目录结构：
```
akshare/
├── dividend/                    # 分红历史数据（按标的）
│   └── {code}.parquet          # 如 600000.parquet
├── finance/                     # 财务数据
│   ├── abstract/               # 财务摘要
│   ├── balance/                # 资产负债表
│   ├── cashflow/               # 现金流量表
│   └── income/                 # 利润表
├── hist/                        # 历史行情
│   └── daily/                  # 日K线数据（按日期存储）
│       ├── {YYYYMMDD}.parquet  # 如 20260117.parquet（所有股票当日数据）
│       └── by_symbol/          # 按标的存储（兼容旧接口）
│           └── {code}.parquet
├── industry/                    # 行业分类
│   ├── sw_level1.parquet       # 申万一级行业
│   ├── sw_level2.parquet       # 申万二级行业
│   ├── sw_level3.parquet       # 申万三级行业
│   └── sw_constituents/        # 行业成分股
└── reports/                     # 研报数据（预留）
```

常用命令：
```bash
# 测试 AKShare 连接
python -m pipeline.data_collection.akshare_api --test

# 采集申万行业分类
python -m pipeline.data_collection.akshare_api --industry

# 采集所有行业成分股
python -m pipeline.data_collection.akshare_api --constituents

# 采集指定股票财务数据
python -m pipeline.data_collection.akshare_api --finance --symbol 600000

# ⭐⭐⭐ 批量采集所有股票完整历史日K线（推荐，快100倍）
# 包含已退市股票，按股票存储到 hist/daily/by_symbol/
python -m pipeline.data_collection.akshare_api --hist-batch                    # 完整历史（1991年至今）
python -m pipeline.data_collection.akshare_api --hist-batch --start 2015-01-01 # 从指定日期开始

# 将按股票存储的数据转换为按日期存储
python -m pipeline.data_collection.akshare_api --convert-daily

# 采集单个日期所有股票（较慢，适合增量更新）
python -m pipeline.data_collection.akshare_api --hist-all --date 2026-01-17

# 采集单只股票历史日K线
python -m pipeline.data_collection.akshare_api --hist --symbol 600000 --start 2024-01-01 --end 2026-01-17

# 采集分红历史
python -m pipeline.data_collection.akshare_api --dividend --symbol 600000
```

配置常量（`pipeline/shared/config.py`）：
- `AKSHARE_DATA_ROOT` - AKShare 数据根目录
- `AKSHARE_INDUSTRY_DIR` - 行业分类目录
- `AKSHARE_FINANCE_DIR` - 财务数据目录
- `AKSHARE_DIVIDEND_DIR` - 分红数据目录
- `AKSHARE_HIST_DIR` - 历史行情目录

## 常用命令

### 数据采集（每日任务）

```bash
# 手动采集指定日期
python -m pipeline.data_collection.collector --date 2026-01-17

# 手动采集日期范围
python -m pipeline.data_collection.collector --start 2026-01-01 --end 2026-01-17

# 查看采集状态
python -m pipeline.data_collection.daily_task --status
```

**后台启动每日任务**（推荐）:

```batch
# 启动脚本（自动清理旧进程，后台运行）
scripts\start_daily_task.bat

# 测试模式（立即执行一次采集）
scripts\start_daily_task.bat --test
```

脚本功能：
- 自动检测并终止已存在的每日任务进程
- 删除旧锁文件
- 在后台启动新的调度器（17:00 后自动采集）
- 显示运行状态和常用命令

**开机自启动**（推荐，防止系统重启后任务中断）:

```batch
# 安装开机自启动
scripts\install_startup.bat

# 卸载：删除快捷方式
del "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\GluttonousDailyTask.lnk"
```

**手动管理**:

```powershell
# 查看状态文件
type DATA_COLLECTION_STATUS.json

# 查看今日日志
Get-Content logs\2026-01-22\data_collection.log -Tail 30

# 停止任务（PID 从状态或脚本输出获取）
taskkill /PID <pid> /F
```

配置文件：`pipeline/shared/config.py`
- `JUEJIN_CONFIG`: 掘金 API 配置
- `DAILY_TASK_CONFIG`: 每日任务配置（运行时间、重试次数等）

日志输出：`logs/{date}/data_collection.log`
状态文件：`DATA_COLLECTION_STATUS.json`（项目根目录）

### LSTM 实验框架（推荐使用）

```bash
# 测试框架
python src/lstm/scripts/test_framework.py

# 运行扩展窗口策略实验
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 计算回测指标并更新文档
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics \
    --update_claude_md
```

详细使用指南：`src/lstm/README.md`
迁移文档：`docs/LSTM_FRAMEWORK_MIGRATION.md`
快速入门：`docs/QUICKSTART_LSTM.md`

### 多模型实验框架

```bash
# 运行 LightGBM 模型
python src/models/scripts/run_lightgbm.py \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 运行 MLP 模型
python src/models/scripts/run_mlp.py \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 运行 Ensemble 集成模型
python src/models/scripts/run_ensemble.py \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 多模型对比（运行所有模型并生成对比报告）
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp,ensemble \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 生成可视化报告
python src/models/scripts/generate_report.py \
    --result src/models/data/results/lightgbm_xxx.json
```

模型配置：`src/models/config.py`
结果输出：`src/models/data/results/`

### 数据流水线

```bash
# 1. 数据清洗（将分钟数据转换为日K线）
# ⚠️ 默认使用中证1000指数成分股，并加载历史所有成分股（避免幸存者偏差）
python -m pipeline.data_cleaning.clean

# 可选参数：
# --index SHSE.000852              # 指定指数代码（默认中证1000）
# --start_date 2024-06-18          # 开始日期
# --end_date 2026-01-13            # 结束日期
# --use_historical                 # 加载历史成分股（默认启用）
# --all_stocks                     # 使用所有股票（不推荐，数据量巨大）

# 2. 特征工程（计算技术指标）
# 会自动添加 in_index 字段标记每天的成分股状态
python -m pipeline.data_cleaning.features

# 3. 数据校验
python -m pipeline.data_validation.validate

# 4. 测试动态成分股功能
python scripts/test_dynamic_constituents.py
```

**动态成分股功能说明**：

为避免**幸存者偏差**和**前瞻偏差**，系统实现了按日期动态读取指数成分股：

1. **数据清洗阶段**：
   - 读取历史上所有出现过的指数成分股（不只是最新成分股）
   - 保存历史成分股数据到 `.pipeline_data/index_constituents.parquet`

2. **特征工程阶段**：
   - 为每条记录添加 `in_index` 字段（0或1）
   - 标记该股票在该日期是否属于指数

3. **训练和回测阶段**：
   - 只使用当日 `in_index == 1` 的股票进行预测和选股
   - 确保每个时间点使用的是当时实际的成分股

### 缓存管理

⚠️ **重要**: 修改数据清洗或特征工程逻辑后，必须先清除缓存以防止数据污染：

```powershell
# 清除所有流水线缓存
Remove-Item -Recurse -Force ".pipeline_data\cleaned\*"
Remove-Item -Recurse -Force ".pipeline_data\features\*"
Remove-Item -Recurse -Force ".pipeline_data\train\*"
```

缓存目录说明：
- `.pipeline_data/cleaned/` - 清洗后的日K线数据（修改 `clean.py` 后清除）
- `.pipeline_data/features/` - 特征工程输出（修改 `features.py` 后清除）
- `.pipeline_data/train/` - 训练数据集（修改特征或标签逻辑后清除）
- `.pipeline_data/checkpoints/` - 模型检查点（重新训练时可保留）
- `.pipeline_data/backtest_results/` - 回测结果

## 架构

### 目录结构

```
src/
├── lstm/                        # LSTM 训练框架
│   ├── config.py               # LSTM 专用配置
│   ├── models/                 # 模型定义
│   │   └── lstm_model.py      # SimpleLSTMModel & LSTMModel
│   ├── experiments/            # 多策略实验框架
│   │   ├── base_executor.py   # 策略执行器基类
│   │   ├── experiment_manager.py  # 实验管理器
│   │   ├── executors/         # 策略执行器
│   │   │   └── expanding_window.py  # 扩展窗口策略
│   │   └── metrics/           # 结果记录
│   │       └── result_recorder.py   # 指标计算
│   ├── scripts/               # 运行脚本
│   │   ├── run_experiments.py # 主实验脚本
│   │   └── test_framework.py  # 测试脚本
│   └── data/                  # 数据目录（gitignore）
│       ├── checkpoints/       # 模型检查点
│       └── results/           # 实验结果
│
└── models/                      # 多模型训练框架
    ├── config.py               # 多模型配置（LightGBM/MLP/Ensemble）
    ├── base/                   # 基类定义
    │   ├── base_model.py      # 模型基类
    │   └── base_executor.py   # 执行器基类
    ├── lightgbm/              # LightGBM 模型
    │   ├── model.py           # 模型实现
    │   └── executor.py        # 执行器
    ├── mlp/                   # MLP 多层感知器
    │   ├── model.py           # 模型实现
    │   └── executor.py        # 执行器
    ├── ensemble/              # 集成模型
    │   ├── voting.py          # 投票策略
    │   └── executor.py        # 执行器
    ├── experiments/           # 实验管理
    │   └── experiment_manager.py
    ├── scripts/               # 运行脚本
    │   ├── run_lightgbm.py   # 运行 LightGBM
    │   ├── run_mlp.py        # 运行 MLP
    │   ├── run_ensemble.py   # 运行集成模型
    │   ├── compare_models.py # 多模型对比
    │   └── generate_report.py # 生成报告
    └── data/                  # 数据目录（gitignore）
        ├── checkpoints/       # 模型检查点
        └── results/           # 实验结果

pipeline/
├── data_collection/     # 数据采集模块
│   ├── juejin_api.py   # 掘金 API 封装
│   ├── collector.py    # 数据采集器
│   ├── daily_task.py   # 每日任务调度器
│   └── notifier.py     # 通知模块（邮件等）
├── data_cleaning/       # 数据清洗 & 特征工程
│   ├── clean.py        # 将分钟数据转换为日K线
│   └── features.py     # 技术指标计算
├── data_validation/     # 数据校验
│   └── validate.py
└── shared/             # 共享模块
    ├── config.py          # 基础配置
    ├── logging_config.py  # 统一日志配置
    └── utils.py           # 工具函数

scripts/                 # 脚本目录
├── start_daily_task.bat  # 每日任务启动脚本
├── archive_strategy.py   # 策略归档脚本
└── temp/                 # ⚠️ 临时脚本（不纳入版本控制）

logs/                    # 工作流日志（按日期组织）
└── {date}/
    ├── data_collection.log
    ├── data_cleaning.log
    └── ...
```

### 临时脚本规范

⚠️ **重要规则**: 所有临时脚本、测试脚本、一次性脚本必须放在 `scripts/temp/` 目录下。

- `scripts/` - 仅存放正式的、长期使用的脚本
- `scripts/temp/` - 临时脚本目录，已配置 `.gitignore` 不纳入版本控制

**示例**:
```bash
# 正式脚本 → scripts/
scripts/start_daily_task.bat    # 每日任务启动
scripts/archive_strategy.py     # 策略归档

# 临时脚本 → scripts/temp/
scripts/temp/test_xxx.py        # 测试脚本
scripts/temp/exp_xxx.py         # 实验脚本
scripts/temp/debug_xxx.py       # 调试脚本
```

### 数据流

1. **原始数据** (NAS) → `clean.py` → **日K线** (`.pipeline_data/cleaned/`)
2. **日K线** → `features.py` → **特征 + 标签** (`.pipeline_data/features/`)
3. **特征** → **模型训练** → **检查点**
   - LSTM 框架 → `src/lstm/data/checkpoints/`
   - 多模型框架 → `src/models/data/checkpoints/`
4. **模型** → **回测预测** → **实验结果**
   - LSTM 框架 → `src/lstm/data/results/`
   - 多模型框架 → `src/models/data/results/`

### Walk-Forward 验证

项目实现了正确的时间序列验证以避免前瞻偏差：

**核心原则**: 模型只能看到历史数据。训练集/验证集/测试集必须严格按时间顺序分离。

**实现方式** (LSTM 实验框架):
- 支持多种训练策略（扩展窗口、K折验证、多尺度集成等）
- 扩展窗口策略：累积历史数据，样本权重衰减
- 可配置训练/验证窗口（默认：60天最小训练集）
- 确保无时间泄露：`train_dates.max() < val_dates.min()`

### 模型配置

#### LSTM 框架 (`src/lstm/config.py`)

- 模型：LSTM (hidden=128, layers=2, dropout=0.3)
- 策略：每日持仓前10只，持有5天
- 概率阈值：0.60
- 批次大小：1024 (GPU)
- 训练：10个epoch，早停耐心值=3

#### 多模型框架 (`src/models/config.py`)

**LightGBM 配置**:
- num_leaves: 24, max_depth: 5
- learning_rate: 0.05, n_estimators: 150
- 正则化: L1=0.15, L2=0.15
- 概率阈值: 0.55

**MLP 配置**:
- 网络结构: [64, 32]，dropout=0.6
- 训练: 15 epochs, batch_size=1024
- 学习率: 0.001, weight_decay=0.02
- 概率阈值: 0.60

**Ensemble 配置**:
- 子模型: LightGBM + MLP
- 投票方式: soft voting
- 权重: LightGBM:MLP = 2:8
- 概率阈值: 0.55

### 特征工程

特征定义在 `pipeline/data_cleaning/features.py`:
- 收益率：1/5/10/20日收益率
- 移动平均：5/10/20/60日均线比率
- 波动率：5/10/20日波动率
- 技术指标：RSI、MACD、布林带
- 成交量特征：成交量比率、换手率
- 策略特征：突破信号、止损触发
- 市场特征：市场收益、相对强度
- 涨跌停标记

标签：二分类（5日后价格是否上涨？）
- `label = 1` 如果 `close[T+5] > close[T]`
- `label = 0` 否则

## 硬件环境

- **GPU**: 本地 CUDA 显卡用于训练加速
- **内存**: 32GB RAM
- **存储**: 本地 SSD 用于 `.pipeline_data/` + NAS 用于原始数据
- **Polars 流式处理**: 已启用，避免内存溢出

## 日志规范

⚠️ **重要规则**: 所有日志必须输出到 `logs/` 目录，严禁使用 `logging.basicConfig()` 或直接输出到控制台/文件。

**统一日志配置**: `pipeline/shared/logging_config.py`

```python
# 正确用法
from pipeline.shared.logging_config import get_collection_logger, get_akshare_logger

# 数据采集日志
logger = get_collection_logger("2026-01-18")  # 按日期创建日志

# AKShare 日志
logger = get_akshare_logger()  # 通用 AKShare 日志
```

**日志目录结构**:
```
logs/
└── {YYYY-MM-DD}/
    ├── data_collection.log    # 掘金数据采集
    ├── akshare.log           # AKShare 数据采集
    ├── data_cleaning.log     # 数据清洗
    └── ...
```

**开发新模块时**:
1. 在 `logging_config.py` 中添加对应的 `get_xxx_logger()` 函数
2. 使用该函数获取 logger，不要自行配置
3. 日志文件按日期自动归档到对应目录

## 时间序列训练规则

⚠️ **关键**: 量化交易机器学习绝不能使用随机划分训练/测试集。这会导致前瞻偏差，虚高回测结果。

**训练前必须检查**:
1. ✅ `max(train_dates) < min(val_dates)`
2. ✅ `max(val_dates) < min(test_dates)`
3. ✅ 特征不使用未来数据
4. ✅ 标准化参数仅从训练集计算
5. ✅ 回测只使用当时可获得的数据

## 策略配置

当前设置（`src/lstm/config.py` - TRADING_CONFIG）:
```python
TOP_N = 10              # 每日持仓数
PROB_THRESHOLD = 0.60   # 概率阈值
HOLDING_DAYS = 5        # 持有天数
COMMISSION = 0.001      # 0.1% 手续费
SLIPPAGE = 0.001        # 0.1% 滑点
```

最近表现（v0.3，扩展窗口策略，2025-04 至 2026-01）:
- 总收益率：+74.84%
- 夏普比率：1.566
- 最大回撤：47.04%

## 补充说明

- **默认股票池**：中证1000指数成分股（SHSE.000852）
  - 使用动态成分股：每个时间点使用当时实际的成分股
  - 历史上约1000只股票，具体数量随时间变化
  - 避免幸存者偏差：清洗历史上所有出现过的成分股
- 结果保存到 `.pipeline_data/backtest_results/`，JSON 格式
- 训练使用采样（50%数据）以加快迭代速度同时保持稳定性
- **重要**：回测时只使用当日 `in_index == 1` 的股票进行选股

## 工作流程

### 完整工作流程

1. **数据准备**：
   ```bash
   python -m pipeline.data_cleaning.clean       # 数据清洗
   python -m pipeline.data_cleaning.features    # 特征工程
   python -m pipeline.data_validation.validate  # 数据校验
   ```

2. **模型训练与回测**：

   **LSTM 框架**:
   ```bash
   # 测试框架
   python src/lstm/scripts/test_framework.py

   # 运行完整实验（含回测和指标计算）
   python src/lstm/scripts/run_experiments.py \
       --strategies expanding_window \
       --calculate_metrics \
       --update_claude_md
   ```

   **多模型框架**:
   ```bash
   # 运行单个模型
   python src/models/scripts/run_lightgbm.py --start_date 2025-04-01 --end_date 2026-01-15
   python src/models/scripts/run_mlp.py --start_date 2025-04-01 --end_date 2026-01-15
   python src/models/scripts/run_ensemble.py --start_date 2025-04-01 --end_date 2026-01-15

   # 多模型对比
   python src/models/scripts/compare_models.py --models lightgbm,mlp,ensemble
   ```

3. **查看结果**：
   - LSTM 结果：`src/lstm/data/results/experiments/`
   - 多模型结果：`src/models/data/results/`
   - 模型检查点：`src/lstm/data/checkpoints/` 或 `src/models/data/checkpoints/`

### 使用 Skills（自动化工作流）

项目定义了 8 个核心技能，可通过以下命令调用：

- `/collect` - 数据采集
- `/clean` - 数据清洗
- `/validate` - 数据校验
- `/train` - 模型训练（支持 LSTM/LightGBM/MLP/Ensemble）
- `/backtest` - 策略回测
- `/compare` - 多模型对比
- `/archive` - 策略归档
- `/review` - 代码审查

详细说明见：[.claude/skills/README.md](.claude/skills/README.md)


## 实验结果对比

**更新时间**: 2026-01-17T20:57:31.136808

### LSTM 框架

| 策略 | 实验ID | 总收益率 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易次数 |
|------|--------|---------|---------|--------|---------|------|---------|
| v03_daily | 05030_baaf99 | +39.14% | +196.39% | 2.744 | 30.56% | 61.84% | 693 |

### 多模型框架

| 模型 | 实验ID | 总收益率 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易次数 |
|------|--------|---------|---------|--------|---------|------|---------|
| LightGBM | - | - | - | - | - | - | - |
| MLP | - | - | - | - | - | - | - |
| Ensemble | - | - | - | - | - | - | - |

> 注：多模型框架结果将在首次运行后自动更新
