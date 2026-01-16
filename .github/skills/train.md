# 模型训练 (Training)

使用GPU加速训练LSTM/Transformer模型，预测股票涨跌。

## 职责

基于特征工程的输出，训练深度学习模型进行股票涨跌预测，生成可用于回测的模型检查点。

## 输入

- **训练数据**: `.pipeline_data/train/train.npz`
- **验证数据**: `.pipeline_data/train/val.npz`
- **测试数据**: `.pipeline_data/train/test.npz`
- **标准化参数**: `.pipeline_data/checkpoints/X_mean.npy`, `X_std.npy`

## 输出

- **模型检查点**: `.pipeline_data/checkpoints/best_model.pt`
- **训练历史**: `.pipeline_data/checkpoints/training_history.json`
- **训练日志**: `.pipeline_data/checkpoints/training.log`
- **标准化参数**: `.pipeline_data/checkpoints/X_mean.npy`, `X_std.npy`

## 模型配置

当前使用的配置（在 `pipeline/shared/config.py` 中）：

```python
TRAIN_CONFIG = {
    "model_type": "lstm",      # lstm / transformer / gru
    "hidden_size": 64,         # 隐藏层大小
    "num_layers": 1,           # 层数
    "dropout": 0.5,            # Dropout率
    "batch_size": 256,         # 批次大小(GPU) / 64(CPU)
    "epochs": 100,             # 训练轮数
    "learning_rate": 0.001,    # 学习率
    "weight_decay": 1e-3,      # L2正则化
    "patience": 10,            # 早停patience
    "use_amp": True,           # 混合精度训练(GPU)
}
```

## GPU优化技术

1. **混合精度训练 (AMP)**: 使用float16减少显存占用，加速计算
2. **梯度裁剪**: 防止梯度爆炸
3. **多进程数据加载**: `num_workers=8`, `pin_memory=True`
4. **预取机制**: `prefetch_factor=4`

## 运行命令

```powershell
# 使用默认配置训练
python -m pipeline.training.train

# 自定义配置
python -m pipeline.training.train --batch_size 1024 --epochs 200 --learning_rate 0.0005

# 快速测试（少量epoch）
python -m pipeline.training.train --epochs 10
```

## 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Accuracy | 分类准确率 | > 55% |
| Precision | 正类精确率 | > 50% |
| Recall | 正类召回率 | > 50% |
| F1 Score | F1分数 | > 50% |

## 时间序列训练规则

⚠️ **关键**: 本项目使用严格的时间序列验证，避免前瞻偏差：

1. ✅ 训练集、验证集、测试集严格按时间顺序划分
2. ✅ 标准化参数仅从训练集计算
3. ✅ 绝不使用未来数据

**Walk-Forward验证**：

```powershell
# 滚动时间窗口训练和验证
python -m pipeline.training.walk_forward --start_date 2025-01-02 --end_date 2025-06-30

# 自定义窗口大小
python -m pipeline.training.walk_forward --train_days 120 --val_days 20 --retrain_freq 5
```

## 训练监控

训练过程中应关注：
- 训练损失持续下降
- 验证损失不应远高于训练损失（过拟合检测）
- 准确率应在合理范围内（不应过高，可能有数据泄露）

## 验证要点

数据校验角色应检查：

- [ ] 训练损失持续下降
- [ ] 验证损失无严重过拟合（train_loss vs val_loss差距 < 0.1）
- [ ] 测试集指标与验证集接近（差距 < 5%）
- [ ] 模型文件正常保存（best_model.pt存在且大小合理）
- [ ] 标准化参数已保存（X_mean.npy, X_std.npy）

## 常见问题

### 过拟合
- 增加dropout率
- 减少模型复杂度（hidden_size, num_layers）
- 增加L2正则化（weight_decay）
- 使用更多数据

### 欠拟合
- 增加模型复杂度
- 增加训练轮数
- 降低dropout率
- 检查特征工程质量

### 准确率异常高（>70%）
⚠️ 可能存在数据泄露，检查：
- 特征是否使用了未来数据
- 标签是否计算正确
- 数据划分是否按时间顺序

## 下一步

训练完成后：
1. 检查训练历史和指标
2. 运行策略回测验证模型效果：`python backtest_v5.py`
3. 或使用pipeline回测：`python -m pipeline.backtest.backtest`
