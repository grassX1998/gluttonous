# LSTM 框架快速入门

## 🎯 5分钟快速开始

### 1. 验证环境

```bash
# 测试 LSTM 框架
python src/lstm/scripts/test_framework.py

# 预期输出：SUCCESS! LSTM 框架测试全部通过
```

### 2. 运行第一个实验

```bash
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15 \
    --calculate_metrics
```

### 3. 查看结果

```bash
# 实验结果保存在
ls src/lstm/data/results/experiments/

# 查看 JSON 结果
cat src/lstm/data/results/experiments/expanding_window_*.json
```

---

## 📖 常用命令

### 测试框架

```bash
# 完整测试
python src/lstm/scripts/test_framework.py

# 只测试导入
python -c "from src.lstm.config import *; from src.lstm.models import *; print('OK')"
```

### 运行实验

```bash
# 基本实验（仅生成预测）
python src/lstm/scripts/run_experiments.py --strategies expanding_window

# 完整实验（计算指标 + 更新文档）
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics \
    --update_claude_md

# 指定日期范围
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-10-01 \
    --end_date 2025-12-31

# 自定义交易参数
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --trading_params '{"top_n": 20, "prob_threshold": 0.65}'
```

---

## 📁 目录说明

```
src/lstm/
├── config.py           # 配置文件（修改这里调整参数）
├── models/             # 模型定义
├── experiments/        # 实验框架
│   ├── executors/     # 策略实现
│   └── metrics/       # 结果记录
├── scripts/           # 运行脚本
│   ├── run_experiments.py    # 主脚本
│   └── test_framework.py     # 测试脚本
└── data/              # 数据目录（gitignore）
    ├── checkpoints/   # 模型检查点
    └── results/       # 实验结果
```

---

## ⚙️ 配置修改

### 修改模型参数

编辑 `src/lstm/config.py`：

```python
MODEL_CONFIG = {
    'hidden_size': 128,      # LSTM 隐藏层大小
    'num_layers': 2,         # LSTM 层数
    'dropout': 0.3,          # Dropout 比例
    'batch_size': 1024,      # 批次大小
    'epochs': 10,            # 训练轮数
    'learning_rate': 0.001,  # 学习率
    'early_stop_patience': 3, # 早停耐心值
}
```

### 修改交易参数

```python
TRADING_CONFIG = {
    'top_n': 10,              # 每日持仓数
    'prob_threshold': 0.60,   # 概率阈值
    'holding_days': 5,        # 持有天数
    'commission': 0.001,      # 手续费
    'slippage': 0.001,        # 滑点
}
```

### 修改策略参数

```python
# 扩展窗口策略配置
config = ExpandingWindowConfig(
    min_train_days=60,        # 最小训练天数
    max_train_days=500,       # 最大训练天数
    use_sample_weight=True,   # 使用样本权重
    weight_decay_days=30,     # 权重衰减周期
    retrain_interval=1        # 重训练间隔
)
```

---

## 🐛 常见问题

### Q: 提示找不到模块

```
ModuleNotFoundError: No module named 'src'
```

**解决**：确保在项目根目录运行
```bash
cd /path/to/gluttonous
python src/lstm/scripts/run_experiments.py ...
```

### Q: 提示数据文件不存在

```
FileNotFoundError: .pipeline_data/features_monthly
```

**解决**：先准备数据
```bash
python -m pipeline.data_cleaning.clean
python -m pipeline.data_cleaning.features
```

### Q: CUDA 内存不足

**解决**：降低批次大小
```python
# 在 src/lstm/config.py 修改
MODEL_CONFIG = {
    'batch_size': 512,  # 从 1024 降到 512
    ...
}
```

---

## 📚 进一步阅读

- **完整使用指南**: `src/lstm/README.md`
- **迁移文档**: `docs/LSTM_FRAMEWORK_MIGRATION.md`
- **实验框架**: `docs/EXPERIMENT_FRAMEWORK.md`

---

**版本**: v1.0.0
**更新**: 2026-01-16
