# 多策略实验框架使用指南

## 概述

本项目实现了一个灵活的多策略实验框架，支持同时运行和对比多种训练策略。

**框架特点：**
- ✅ 统一的配置管理（`pipeline/shared/config.py`）
- ✅ 可扩展的策略执行器架构
- ✅ 自动化的结果记录和对比
- ✅ 支持更新 CLAUDE.md 实验记录

---

## 快速开始

### 1. 验证环境

首先验证重构是否成功：

```bash
# 测试基础导入
python -c "from pipeline.shared.config import ALL_STRATEGY_CONFIGS; print('OK')"

# 测试实验框架
python scripts/test_experiment_framework.py
```

### 2. 运行第一个实验

运行扩展窗口策略（目前唯一完整实现的策略）：

```bash
python scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15 \
    --calculate_metrics \
    --update_claude_md
```

**参数说明：**
- `--strategies`: 要运行的策略列表（可多个）
- `--start_date`: 回测开始日期
- `--end_date`: 回测结束日期
- `--calculate_metrics`: 计算回测指标（收益率、夏普比率等）
- `--update_claude_md`: 自动更新 CLAUDE.md 中的实验结果表格

---

## 架构说明

### 目录结构

```
pipeline/
├── shared/
│   └── config.py                  # 统一配置（包含6个策略配置类）
├── experiments/
│   ├── base_executor.py           # 策略执行器基类
│   ├── experiment_manager.py      # 实验管理器
│   ├── executors/                 # 策略执行器
│   │   ├── expanding_window.py   # ✅ 扩展窗口策略（已实现）
│   │   ├── rolling_kfold.py      # ⏳ K折验证策略（待实现）
│   │   ├── multiscale_ensemble.py # ⏳ 多尺度集成（待实现）
│   │   ├── adaptive_retrain.py   # ⏳ 自适应重训练（待实现）
│   │   ├── incremental_learning.py # ⏳ 增量学习（待实现）
│   │   └── no_val_bayesian.py    # ⏳ 无验证集+贝叶斯优化（待实现）
│   └── metrics/
│       ├── result_recorder.py     # 结果记录器
│       └── performance_tracker.py # 性能追踪器（待实现）
scripts/
├── run_experiments.py             # 主实验运行脚本
├── test_experiment_framework.py   # 框架测试脚本
└── validate_refactoring.py        # 重构验证脚本
```

### 策略配置

所有策略配置定义在 `pipeline/shared/config.py` 中：

```python
from pipeline.shared.config import (
    ExpandingWindowConfig,      # 方案1: 扩展窗口
    RollingKFoldConfig,         # 方案2: K折验证
    MultiScaleEnsembleConfig,   # 方案3: 多尺度集成
    AdaptiveRetrainConfig,      # 方案4: 自适应重训练
    IncrementalLearningConfig,  # 方案5: 增量学习
    NoValBayesianConfig,        # 方案6: 无验证集
)
```

**示例：创建配置**

```python
# 使用默认配置
config = ExpandingWindowConfig()

# 自定义配置
config = ExpandingWindowConfig(
    min_train_days=90,
    max_train_days=600,
    use_sample_weight=True,
    weight_decay_days=20
)
```

---

## 已实现的策略

### 方案1: 扩展窗口策略 (expanding_window)

**特点：**
- 训练集持续增长（累积历史数据）
- 使用指数衰减权重（近期数据权重更高）
- 适合市场存在长期趋势的场景

**配置参数：**
```python
ExpandingWindowConfig(
    min_train_days=60,        # 最小训练天数
    max_train_days=500,       # 最大训练天数
    val_days=1,               # 验证集天数
    use_sample_weight=True,   # 是否使用样本权重
    weight_decay_days=30,     # 权重衰减周期（天）
    weight_decay_rate=0.98,   # 衰减率
    retrain_interval=1        # 重训练间隔（天）
)
```

**运行示例：**
```bash
python scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15
```

---

## 待实现的策略

### 方案2: 固定滚动窗口 + K折验证 (rolling_kfold)

**特点：**
- 固定60天训练窗口
- 时间序列K折交叉验证
- 充分利用有限数据

### 方案3: 多尺度集成 (multiscale_ensemble)

**特点：**
- 短期（20天）、中期（60天）、长期（120天）模型
- 加权集成预测
- 捕捉不同时间尺度的模式

### 方案4: 自适应重训练 (adaptive_retrain)

**特点：**
- 监控验证集性能
- 性能下降时才重训练
- 节省计算资源

### 方案5: 增量学习 (incremental_learning)

**特点：**
- 在线学习，每日微调
- 快速适应市场变化
- 避免灾难性遗忘

### 方案6: 无验证集 + 贝叶斯优化 (no_val_bayesian)

**特点：**
- 无显式验证集
- 使用贝叶斯优化搜索超参数
- 更强的正则化

---

## 如何添加新策略

### Step 1: 定义配置类

在 `pipeline/shared/config.py` 中添加：

```python
@dataclass
class MyNewStrategyConfig(TrainStrategyConfig):
    """新策略配置"""
    strategy_name: str = "my_new_strategy"
    description: str = "新策略的描述"
    # 添加策略特定参数
    param1: int = 60
    param2: float = 0.5
```

### Step 2: 实现执行器

创建 `pipeline/experiments/executors/my_new_strategy.py`：

```python
from pipeline.experiments.base_executor import BaseStrategyExecutor
from pipeline.shared.config import MyNewStrategyConfig

class MyNewStrategyExecutor(BaseStrategyExecutor):
    def __init__(self, config: MyNewStrategyConfig):
        super().__init__(config)

    def prepare_data(self, current_date: str):
        # 实现数据准备逻辑
        ...
        return X_train, y_train, X_val, y_val

    def should_retrain(self, current_date: str) -> bool:
        # 实现重训练判断逻辑
        ...
        return True/False
```

### Step 3: 注册策略

在 `pipeline/shared/config.py` 的 `ALL_STRATEGY_CONFIGS` 中注册：

```python
ALL_STRATEGY_CONFIGS = {
    ...
    "my_new_strategy": MyNewStrategyConfig,
}
```

在 `pipeline/experiments/experiment_manager.py` 的 `_get_executor` 方法中添加映射：

```python
from pipeline.experiments.executors.my_new_strategy import MyNewStrategyExecutor

executor_map = {
    ...
    "my_new_strategy": MyNewStrategyExecutor,
}
```

### Step 4: 测试和运行

```bash
python scripts/run_experiments.py --strategies my_new_strategy
```

---

## 结果分析

### 查看实验结果

实验结果保存在 `.pipeline_data/backtest_results/experiments/` 目录：

```
.pipeline_data/backtest_results/experiments/
├── expanding_window_20260116_203000.json    # 单次实验结果
├── comparison_20260116_203500.json          # 对比结果
└── ...
```

### 结果文件格式

```json
{
  "strategy": "expanding_window",
  "start_date": "2025-04-01",
  "end_date": "2026-01-15",
  "predictions": [
    {"date": "2025-04-01", "symbol": "SHSE.600000", "prob": 0.65},
    ...
  ],
  "retrain_dates": ["2025-04-01", "2025-04-08", ...],
  "performance_history": [
    {"date": "2025-04-01", "val_acc": 0.58, "train_size": 15000, "val_size": 500},
    ...
  ],
  "metrics": {
    "total_return": 0.75,
    "annual_return": 0.85,
    "sharpe_ratio": 1.82,
    "max_drawdown": 0.38,
    "win_rate": 0.58,
    "n_trades": 1520
  }
}
```

### 更新 CLAUDE.md

运行时添加 `--update_claude_md` 参数会自动更新项目根目录的 CLAUDE.md 文件，添加实验结果对比表格：

```markdown
## 实验结果对比

**更新时间**: 2026-01-16 20:35:00

| 策略 | 总收益率 | 年化收益 | Sharpe | 最大回撤 | 胜率 | 交易次数 |
|------|---------|---------|--------|---------|------|---------|
| expanding_window | +75.00% | +85.00% | 1.820 | 38.00% | 58.00% | 1520 |
```

---

## 常见问题

### Q1: 如何只测试特定日期范围？

```bash
python scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-10-01 \
    --end_date 2025-12-31
```

### Q2: 如何修改交易参数（持仓数、阈值等）？

```bash
python scripts/run_experiments.py \
    --strategies expanding_window \
    --trading_params '{"top_n": 20, "prob_threshold": 0.65, "holding_days": 3}'
```

### Q3: 如何并行运行多个策略？

```bash
python scripts/run_experiments.py \
    --strategies expanding_window rolling_kfold multiscale_ensemble
```

（注意：其他策略需要先实现）

### Q4: 实验运行很慢怎么办？

1. **减少日期范围**：先用小范围测试（如1个月）
2. **降低重训练频率**：修改策略配置的 `retrain_interval`
3. **减少训练 epochs**：修改 `base_executor.py` 中的 `model_config['epochs']`

### Q5: 如何保存和加载模型？

执行器支持模型保存和加载：

```python
# 保存模型
executor.save_model(Path("model.pt"))

# 加载模型
executor.load_model(Path("model.pt"))
```

---

## 重构变更总结

### 主要变更

1. **配置统一化**
   - 所有配置集中到 `pipeline/shared/config.py`
   - 添加6个策略配置类（dataclass）

2. **代码去重**
   - 删除 `backtest_v5.py` 中的重复定义
   - 创建 `SimpleLSTMModel` 供快速实验使用
   - 统一导入 `FEATURE_COLS`

3. **路径统一**
   - 添加 `FEATURE_DATA_MONTHLY_DIR` 和 `DAILY_DATA_DIR`
   - 所有路径从 config 导入

### 向后兼容

✅ **现有代码完全兼容**
- `backtest_v5.py` 仍可正常运行
- `pipeline/training/train.py` 未受影响
- 所有原有脚本保持不变

---

## 后续优化方向

1. **实现剩余5个策略执行器**
2. **添加性能追踪器** (`performance_tracker.py`)
3. **实现贝叶斯超参数优化**
4. **添加策略组合优化**
5. **Web 可视化界面**
6. **实时监控面板**

---

## 参考资料

- 完整计划文档：见项目根目录的重构计划文档
- 策略详细说明：`pipeline/shared/config.py` 中的配置类注释
- 基类文档：`pipeline/experiments/base_executor.py` 的类注释

---

**更新时间**: 2026-01-16
**框架版本**: v1.0
