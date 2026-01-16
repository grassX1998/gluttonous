# LSTM 框架迁移指南

## 📋 迁移概述

本次迁移将实验框架从 `pipeline/experiments/` 重新组织到 `src/lstm/` 下，形成独立的 LSTM 训练框架模块。

**迁移日期**: 2026-01-16
**框架版本**: v1.0.0

---

## 🎯 迁移目标

✅ **模块化设计**：LSTM 框架作为独立模块，易于维护和扩展
✅ **数据隔离**：`src/lstm/data/` 作为独立数据目录，已加入 gitignore
✅ **向后兼容**：原有 `pipeline/` 模块保持不变
✅ **清晰结构**：明确的目录层次和职责划分

---

## 📁 新目录结构

```
src/
└── lstm/                        # LSTM 训练框架根目录
    ├── __init__.py
    ├── README.md                # 框架使用指南
    ├── config.py                # 统一配置
    ├── models/                  # 模型定义
    │   ├── __init__.py
    │   └── lstm_model.py       # SimpleLSTMModel & LSTMModel
    ├── experiments/             # 实验框架
    │   ├── __init__.py
    │   ├── base_executor.py    # 策略执行器基类
    │   ├── experiment_manager.py   # 实验管理器
    │   ├── executors/          # 策略执行器
    │   │   ├── __init__.py
    │   │   └── expanding_window.py  # 扩展窗口策略
    │   └── metrics/            # 指标和记录
    │       ├── __init__.py
    │       └── result_recorder.py   # 结果记录器
    ├── scripts/                 # 运行脚本
    │   ├── run_experiments.py  # 主实验运行脚本
    │   └── test_framework.py   # 框架测试脚本
    └── data/                    # 数据目录（gitignore）
        ├── features/           # 特征数据缓存
        ├── checkpoints/        # 模型检查点
        └── results/            # 实验结果
```

---

## 🔄 迁移对比

### 旧结构（迁移前）

```
pipeline/
├── experiments/
│   ├── base_executor.py
│   ├── experiment_manager.py
│   ├── executors/
│   │   └── expanding_window.py
│   └── metrics/
│       └── result_recorder.py
└── shared/
    └── config.py (包含实验配置)

scripts/
└── run_experiments.py

.pipeline_data/
└── backtest_results/
    └── experiments/
```

### 新结构（迁移后）

```
src/lstm/                        # 独立模块
├── config.py                    # LSTM 专用配置
├── models/                      # 模型定义
├── experiments/                 # 实验框架
├── scripts/                     # 运行脚本
└── data/                        # 数据目录（gitignore）
    ├── checkpoints/
    └── results/

pipeline/                        # 保持不变
└── (原有结构)
```

---

## 📝 关键变更

### 1. 配置文件变更

**旧方式**：
```python
from pipeline.shared.config import (
    BACKTEST_RESULT_DIR,
    TrainStrategyConfig
)
```

**新方式**：
```python
from src.lstm.config import (
    EXPERIMENT_RESULT_DIR,
    TrainStrategyConfig,
    MODEL_CONFIG,
    TRADING_CONFIG
)
```

### 2. 模型导入变更

**旧方式**：
```python
from pipeline.training.train import SimpleLSTMModel
```

**新方式**：
```python
from src.lstm.models import SimpleLSTMModel, LSTMModel
```

### 3. 实验框架导入变更

**旧方式**：
```python
from pipeline.experiments import ExperimentManager
from pipeline.experiments.executors.expanding_window import ExpandingWindowExecutor
```

**新方式**：
```python
from src.lstm.experiments import ExperimentManager
from src.lstm.experiments.executors.expanding_window import ExpandingWindowExecutor
```

### 4. 脚本运行方式变更

**旧方式**：
```bash
python scripts/run_experiments.py --strategies expanding_window
```

**新方式**：
```bash
python src/lstm/scripts/run_experiments.py --strategies expanding_window
```

### 5. 数据目录变更

**旧路径**：
- `.pipeline_data/backtest_results/experiments/`

**新路径**：
- `src/lstm/data/results/experiments/`
- `src/lstm/data/checkpoints/`

---

## 🚀 使用新框架

### 快速测试

```bash
# 测试框架是否正常
python src/lstm/scripts/test_framework.py

# 输出：SUCCESS! LSTM 框架测试全部通过
```

### 运行实验

```bash
# 基本用法
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 完整用法（计算指标并更新文档）
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15 \
    --calculate_metrics \
    --update_claude_md
```

### Python 代码使用

```python
import sys
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入框架
from src.lstm.experiments import ExperimentManager
from src.lstm.config import ExpandingWindowConfig

# 创建并运行实验
manager = ExperimentManager(strategies=["expanding_window"])
manager.run_all_experiments("2025-04-01", "2026-01-15")
manager.print_summary()
```

---

## ⚠️ 注意事项

### 数据依赖

新框架仍然依赖 `pipeline` 模块的数据：

- **特征数据**：从 `.pipeline_data/features_monthly/` 读取
- **价格数据**：从 `.pipeline_data/daily/` 读取
- **特征列表**：从 `pipeline.data_cleaning.features` 导入 `FEATURE_COLS`

**数据准备流程保持不变**：
```bash
# 1. 数据清洗
python -m pipeline.data_cleaning.clean

# 2. 特征工程
python -m pipeline.data_cleaning.features
```

### 向后兼容

✅ **原有代码完全保留**：
- `pipeline/experiments/` 目录保持不变
- `scripts/run_experiments.py` 仍然可用
- `backtest_v5.py` 不受影响

**两套框架可以共存**：
- 新实验使用 `src/lstm/`
- 旧脚本继续使用 `pipeline/`
- 数据互不干扰

### .gitignore 更新

已添加以下内容到 `.gitignore`：

```gitignore
# LSTM training data (local cache)
src/lstm/data/
```

这确保本地的模型检查点、实验结果等数据不会被提交到 git。

---

## 📊 迁移验证

### 验证清单

- [x] 目录结构创建完成
- [x] 配置文件正确设置
- [x] 模型定义正确导出
- [x] 实验框架正常工作
- [x] 所有导入路径更新
- [x] .gitignore 已更新
- [x] 测试脚本全部通过 (4/4)
- [x] 文档已更新

### 测试结果

```bash
$ python src/lstm/scripts/test_framework.py

测试总结
============================================================
[PASS] 模块导入
[PASS] 模型创建
[PASS] 执行器创建
[PASS] 管理器创建

总计: 4/4 测试通过

SUCCESS! LSTM 框架测试全部通过
```

---

## 📚 相关文档

- **框架使用指南**: `src/lstm/README.md`
- **实验框架文档**: `docs/EXPERIMENT_FRAMEWORK.md`
- **重构总结**: `docs/REFACTORING_SUMMARY.md`
- **项目说明**: `CLAUDE.md`

---

## 🔮 后续工作

### 短期（1-2周）

- [ ] 实现剩余5个策略执行器
- [ ] 添加更多测试用例
- [ ] 优化数据加载性能

### 中期（1-2月）

- [ ] 实现模型版本管理
- [ ] 添加实验对比可视化
- [ ] 开发 Web 界面

### 长期（3-6月）

- [ ] 分布式训练支持
- [ ] 自动化超参数调优
- [ ] 实时监控面板

---

## ❓ 常见问题

### Q1: 为什么要迁移到 src/lstm/?

**A**: 模块化设计的优势：
- 独立的数据目录，不与其他模块混淆
- 清晰的职责划分
- 更容易维护和扩展
- 可以作为独立包发布

### Q2: 旧的 pipeline/experiments/ 还能用吗？

**A**: 可以。两套框架可以共存，互不影响。如果需要，可以继续使用旧框架。

### Q3: 数据存储在哪里？

**A**:
- **LSTM 框架数据**: `src/lstm/data/` (gitignore)
- **原始数据**: `.pipeline_data/` (gitignore)

### Q4: 如何切换回旧框架？

**A**: 使用旧的导入路径和脚本即可：
```python
from pipeline.experiments import ExperimentManager
```
```bash
python scripts/run_experiments.py ...
```

---

**迁移完成时间**: 2026-01-16
**测试通过率**: 100% (4/4)
**向后兼容**: ✅ 完全兼容
