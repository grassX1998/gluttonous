# LSTM 量化模型训练框架

## 📁 目录结构

```
src/lstm/
├── __init__.py
├── config.py                    # 统一配置
├── README.md                    # 本文件
├── models/                      # 模型定义
│   ├── __init__.py
│   └── lstm_model.py           # SimpleLSTMModel & LSTMModel
├── experiments/                 # 实验框架
│   ├── __init__.py
│   ├── base_executor.py        # 策略执行器基类
│   ├── experiment_manager.py   # 实验管理器
│   ├── executors/              # 策略执行器
│   │   ├── __init__.py
│   │   └── expanding_window.py # 扩展窗口策略
│   └── metrics/                # 指标和记录
│       ├── __init__.py
│       └── result_recorder.py  # 结果记录器
├── scripts/                     # 运行脚本
│   ├── run_experiments.py      # 主实验运行脚本
│   └── test_framework.py       # 测试脚本（待创建）
└── data/                        # 数据目录（gitignore）
    ├── features/               # 特征数据缓存
    ├── checkpoints/            # 模型检查点
    └── results/                # 实验结果
```

## 🚀 快速开始

### 1. 环境准备

确保已安装依赖：
```bash
pip install torch polars numpy pandas
```

### 2. 数据准备

框架需要从 `pipeline` 模块读取原始特征数据，确保以下数据已准备：
- `.pipeline_data/features_monthly/` - 月度特征数据
- `.pipeline_data/daily/` - 日线价格数据

### 3. 运行第一个实验

```bash
# 在项目根目录运行
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15 \
    --calculate_metrics \
    --update_claude_md
```

### 4. 查看结果

实验结果保存在 `src/lstm/data/results/` 目录：
- JSON 格式的详细结果
- 包含预测、重训练记录、性能历史

## 📖 配置说明

### 策略配置

所有策略配置在 `src/lstm/config.py` 中定义：

```python
from src.lstm.config import (
    ExpandingWindowConfig,      # 扩展窗口策略
    RollingKFoldConfig,         # K折验证策略
    MultiScaleEnsembleConfig,   # 多尺度集成
    AdaptiveRetrainConfig,      # 自适应重训练
    IncrementalLearningConfig,  # 增量学习
    NoValBayesianConfig,        # 无验证集+贝叶斯优化
)
```

### 模型配置

```python
# 在 config.py 中
MODEL_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 1024,
    'epochs': 10,
    'learning_rate': 0.001,
    'early_stop_patience': 3,
}
```

### 交易配置

```python
TRADING_CONFIG = {
    'top_n': 10,              # 每日持仓数
    'prob_threshold': 0.60,   # 概率阈值
    'holding_days': 5,        # 持有天数
    'commission': 0.001,      # 手续费
    'slippage': 0.001,        # 滑点
}
```

## 🔧 使用方法

### 方式1：命令行运行

```bash
# 运行单个策略
python src/lstm/scripts/run_experiments.py --strategies expanding_window

# 运行多个策略
python src/lstm/scripts/run_experiments.py --strategies expanding_window rolling_kfold

# 指定日期范围
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-10-01 \
    --end_date 2025-12-31

# 计算回测指标并更新 CLAUDE.md
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics \
    --update_claude_md
```

### 方式2：Python 代码

```python
from src.lstm.experiments import ExperimentManager
from src.lstm.config import ExpandingWindowConfig

# 创建管理器
manager = ExperimentManager(strategies=["expanding_window"])

# 运行实验
manager.run_all_experiments("2025-04-01", "2026-01-15")

# 打印摘要
manager.print_summary()
```

### 方式3：自定义策略

```python
from src.lstm.experiments import BaseStrategyExecutor
from src.lstm.config import ExpandingWindowConfig

# 创建执行器
config = ExpandingWindowConfig(
    min_train_days=90,
    max_train_days=600,
    use_sample_weight=True
)

from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
executor = ExpandingWindowExecutor(config)

# 运行回测
result = executor.run("2025-04-01", "2026-01-15")

# 保存模型
executor.save_model(Path("my_model.pt"))
```

## 📊 已实现的策略

### 方案1: 扩展窗口策略 (expanding_window)

**特点：**
- 训练集持续增长（累积历史数据）
- 使用指数衰减权重（近期数据权重更高）
- 适合市场存在长期趋势的场景

**配置：**
```python
ExpandingWindowConfig(
    min_train_days=60,
    max_train_days=500,
    use_sample_weight=True,
    weight_decay_days=30,
    weight_decay_rate=0.98
)
```

## 🔮 待实现的策略

- `rolling_kfold` - K折验证策略
- `multiscale_ensemble` - 多尺度集成
- `adaptive_retrain` - 自适应重训练
- `incremental_learning` - 增量学习
- `no_val_bayesian` - 无验证集+贝叶斯优化

## 🐛 故障排查

### 问题1：模块导入错误

```
ModuleNotFoundError: No module named 'src'
```

**解决方案**：在项目根目录运行脚本
```bash
cd /path/to/gluttonous
python src/lstm/scripts/run_experiments.py ...
```

### 问题2：数据文件不存在

```
FileNotFoundError: .pipeline_data/features_monthly
```

**解决方案**：先运行数据准备流程
```bash
python -m pipeline.data_cleaning.clean
python -m pipeline.data_cleaning.features
```

### 问题3：CUDA 内存不足

**解决方案**：降低批次大小
```python
# 在 config.py 中修改
MODEL_CONFIG = {
    'batch_size': 512,  # 降低批次大小
    ...
}
```

## 📚 参考文档

- 完整使用指南：`docs/EXPERIMENT_FRAMEWORK.md`
- 重构总结：`docs/REFACTORING_SUMMARY.md`
- 项目说明：`CLAUDE.md`

## 🤝 如何贡献

1. 实现新策略：继承 `BaseStrategyExecutor` 并实现抽象方法
2. 添加新模型：在 `models/` 目录添加新模型定义
3. 改进指标：在 `metrics/` 目录添加新的评估指标

---

**版本**: v1.0.0
**更新时间**: 2026-01-16
