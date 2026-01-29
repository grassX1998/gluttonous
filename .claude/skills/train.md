# 模型训练 (Training)

支持 LSTM 框架和多模型框架（LightGBM/MLP/Ensemble）进行模型训练。

## 职责

基于特征工程的输出，使用不同框架训练机器学习模型，进行股票涨跌预测。

## 框架选择

| 框架 | 模型 | 特点 | 适用场景 |
|------|------|------|----------|
| LSTM | LSTM 深度学习 | 时序建模能力强 | 时序特征丰富的数据 |
| 多模型 | LightGBM | 训练快，可解释性强 | 快速实验、特征重要性分析 |
| 多模型 | MLP | 非线性拟合能力强 | 高维特征数据 |
| 多模型 | Ensemble | 集成多模型优势 | 追求稳定表现 |

## 输入

- **特征数据**: `.pipeline_data/features_monthly/*.parquet`
- **价格数据**: `.pipeline_data/daily/*.parquet`

## 输出

### LSTM 框架
- **模型检查点**: `src/lstm/data/checkpoints/*.pt`
- **实验结果**: `src/lstm/data/results/experiments/*.json`
- **训练日志**: `logs/{date}/lstm_training.log`

### 多模型框架
- **模型检查点**: `src/models/data/checkpoints/*.pkl` (LightGBM) / `*.pt` (MLP)
- **实验结果**: `src/models/data/results/*.json`
- **训练日志**: `logs/{date}/models_training.log`

### 通用
- **实验注册表**: `logs/experiments_registry.json`

### 日志输出结构

```
logs/{date}/
├── lstm_training.log          # 训练过程日志
└── experiments/
    └── {experiment_id}/
        ├── config.json        # 配置快照
        └── metrics.json       # 回测指标
```

## 模型配置

### LSTM 配置 (`src/lstm/config.py`)

```python
MODEL_CONFIG = {
    "hidden_size": 128,        # LSTM 隐藏层大小
    "num_layers": 2,           # LSTM 层数
    "dropout": 0.3,            # Dropout率
    "batch_size": 1024,        # 批次大小(GPU)
    "epochs": 10,              # 训练轮数
    "learning_rate": 0.001,    # 学习率
    "early_stop_patience": 3,  # 早停耐心值
}

TRADING_CONFIG = {
    "top_n": 10,              # 每日持仓数
    "prob_threshold": 0.60,   # 概率阈值
    "holding_days": 5,        # 持有天数
    "commission": 0.001,      # 手续费 0.1%
    "slippage": 0.001,        # 滑点 0.1%
}
```

### 多模型配置 (`src/models/config.py`)

```python
# LightGBM 配置
LightGBMConfig = {
    "num_leaves": 24,          # 叶子数
    "max_depth": 5,            # 最大深度
    "learning_rate": 0.05,     # 学习率
    "n_estimators": 150,       # 迭代次数
    "reg_alpha": 0.15,         # L1 正则
    "reg_lambda": 0.15,        # L2 正则
    "prob_threshold": 0.55,    # 概率阈值
}

# MLP 配置
MLPConfig = {
    "hidden_sizes": [64, 32],  # 网络结构
    "dropout": 0.6,            # Dropout率
    "epochs": 15,              # 训练轮数
    "learning_rate": 0.001,    # 学习率
    "weight_decay": 0.02,      # L2 正则
    "prob_threshold": 0.60,    # 概率阈值
}

# Ensemble 配置
EnsembleConfig = {
    "models": ["lightgbm", "mlp"],  # 子模型
    "voting": "soft",               # 投票方式
    "weights": [0.2, 0.8],          # 权重 (LightGBM:MLP)
    "prob_threshold": 0.55,         # 概率阈值
}
```

## 运行命令

### LSTM 框架

```bash
# 测试框架
python src/lstm/scripts/test_framework.py

# 运行扩展窗口策略实验
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 运行完整实验（含指标计算）
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics \
    --update_claude_md

# 自定义交易参数
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --trading_params '{"top_n": 20, "prob_threshold": 0.65}'
```

### 多模型框架

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

# 多模型对比（运行所有模型）
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp,ensemble \
    --start_date 2025-04-01 \
    --end_date 2026-01-15
```

## 训练策略

当前实现的策略：

1. **扩展窗口策略** (`expanding_window`):
   - 累积历史数据进行训练
   - 使用样本权重（近期数据权重更高）
   - 支持每日重训练或定期重训练

计划实现的策略：
- K折交叉验证
- 多尺度集成
- 自适应重训练
- 增量学习
- 无验证集贝叶斯优化

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Total Return | 总收益率 | > 50% |
| Sharpe Ratio | 夏普比率 | > 1.0 |
| Max Drawdown | 最大回撤 | < 50% |
| Win Rate | 胜率 | > 50% |

## Walk-Forward 验证

实验框架采用严格的 Walk-Forward 方法避免前瞻偏差：

- 训练数据：累积历史数据（扩展窗口）或固定窗口
- 验证数据：训练结束后的第一天
- 测试数据：从验证后的下一天开始
- 模型从未"看到"未来数据

## 验证要点

- [ ] 训练损失持续下降
- [ ] 验证准确率在合理范围（52%-58%）
- [ ] 测试集指标与验证集接近
- [ ] 模型文件正常保存
- [ ] 无前瞻偏差（训练日期 < 测试日期）

## 常见问题

### 过拟合
- 增加 dropout 率
- 减少模型复杂度（hidden_size, num_layers）
- 增加早停耐心值
- 使用样本权重

### 欠拟合
- 增加模型复杂度
- 增加训练轮数
- 降低 dropout 率
- 检查特征工程质量

### 准确率异常高（>65%）
⚠️ 可能存在数据泄露，检查：
- 特征是否使用了未来数据
- 标签是否计算正确
- 数据划分是否按时间顺序

## 输出示例

```json
{
  "strategy": "expanding_window",
  "config": {
    "min_train_days": 60,
    "use_sample_weight": true,
    "weight_decay_days": 30
  },
  "predictions": [...],
  "retrain_dates": [...],
  "performance_history": [...]
}
```

## 下一步

训练完成后：
1. 查看实验结果 JSON 文件
2. 如果启用了 `--calculate_metrics`，检查回测指标
3. 如果结果理想，可以考虑实盘验证
4. 持续优化模型参数和策略配置

## 相关文档

- 快速入门：`docs/QUICKSTART_LSTM.md`
- LSTM 框架指南：`src/lstm/README.md`
- 多模型配置：`src/models/config.py`
- 迁移文档：`docs/LSTM_FRAMEWORK_MIGRATION.md`
