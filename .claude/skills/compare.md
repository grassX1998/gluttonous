# 多模型对比 (Compare)

运行多个模型并生成对比分析报告。

## 职责

同时运行多个模型（LightGBM、MLP、Ensemble），生成标准化的对比报告，帮助选择最优模型。

## 输入

- **特征数据**: `.pipeline_data/features_monthly/*.parquet`
- **价格数据**: `.pipeline_data/daily/*.parquet`

## 输出

- **实验结果**: `src/models/data/results/{model}_{timestamp}.json`
- **对比报告**: `src/models/data/results/reports/comparison_{timestamp}.md`
- **对比图表**: `src/models/data/results/reports/comparison_{timestamp}.png`
- **训练日志**: `logs/{date}/models_comparison.log`

## 可用模型

| 模型 | 说明 | 特点 |
|------|------|------|
| `lightgbm` | 梯度提升树 | 训练快速，可解释性强，对特征工程敏感 |
| `mlp` | 多层感知器 | 非线性拟合能力强，需要更多数据 |
| `ensemble` | 集成模型 | 综合多模型优势，表现更稳定 |

## 运行命令

```bash
# 运行所有模型对比
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp,ensemble \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 只对比两个模型
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 从已有结果生成对比报告
python src/models/scripts/generate_report.py \
    --compare \
    --results src/models/data/results/lightgbm_xxx.json,src/models/data/results/mlp_xxx.json
```

## 模型配置

配置文件：`src/models/config.py`

```python
# LightGBM 配置
LightGBMConfig:
    num_leaves: 24          # 叶子数
    max_depth: 5            # 最大深度
    learning_rate: 0.05     # 学习率
    prob_threshold: 0.55    # 概率阈值

# MLP 配置
MLPConfig:
    hidden_sizes: [64, 32]  # 网络结构
    dropout: 0.6            # Dropout率
    prob_threshold: 0.60    # 概率阈值

# Ensemble 配置
EnsembleConfig:
    models: [lightgbm, mlp] # 子模型
    weights: [0.2, 0.8]     # 权重
    voting: soft            # 投票方式
    prob_threshold: 0.55    # 概率阈值
```

## 对比指标

| 指标 | 说明 | 重要性 |
|------|------|--------|
| Total Return | 总收益率 | 核心指标 |
| Annual Return | 年化收益率 | 可比性 |
| Sharpe Ratio | 风险调整收益 | 核心指标 |
| Max Drawdown | 最大回撤 | 风险控制 |
| Win Rate | 胜率 | 稳定性 |
| Trade Count | 交易次数 | 活跃度 |

## 对比报告内容

Markdown 报告包含：

1. **执行概要**：时间、期间、参与模型
2. **指标对比表**：所有模型的核心指标横向对比
3. **收益曲线对比**：多条收益曲线在同一图表
4. **风险分析**：回撤对比、波动率对比
5. **最佳模型推荐**：基于综合指标的推荐
6. **配置详情**：各模型的完整配置

## 验证要点

- [ ] 所有模型使用相同的时间范围
- [ ] 所有模型使用相同的交易成本设置
- [ ] 对比指标计算方法一致
- [ ] 报告中数据与实际结果匹配

## 典型工作流

```bash
# 1. 确保数据已准备
python -m pipeline.data_cleaning.features

# 2. 运行多模型对比
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp,ensemble \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 3. 查看对比报告
cat src/models/data/results/reports/comparison_*.md

# 4. 根据结果选择最优模型进行后续实验
```

## 下一步

对比完成后：

1. 分析各模型的优劣势
2. 选择表现最好的模型进行参数优化
3. 考虑使用 Ensemble 结合多模型优势
4. 更新 CLAUDE.md 中的实验结果表格

## 相关文档

- 模型配置：`src/models/config.py`
- LSTM 框架对比：`src/lstm/README.md`
- 回测技能：[backtest.md](./backtest.md)
