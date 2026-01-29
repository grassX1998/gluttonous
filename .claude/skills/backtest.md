# 策略回测 (Backtest)

支持 LSTM 框架和多模型框架（LightGBM/MLP/Ensemble）进行策略回测和性能评估。

## 职责

基于训练好的模型，在历史数据上模拟真实交易，评估策略的收益、风险和稳定性。

## 输入

- **特征数据**: `.pipeline_data/features_monthly/*.parquet`
- **价格数据**: `.pipeline_data/daily/*.parquet`
- **模型检查点**: `src/lstm/data/checkpoints/*.pt` (如有)

## 输出

### 标准输出结构

所有回测必须生成以下标准化输出：

#### LSTM 框架

1. **实验结果** (`src/lstm/data/results/experiments/`):
   - `{strategy}_{timestamp}.json` - 预测结果和配置

2. **回测结果** (`src/lstm/data/results/backtest/`):
   - `backtest_{timestamp}.json` - 完整回测数据（订单、持仓、每日记录）

3. **报告** (`src/lstm/data/results/reports/`):
   - `{strategy}_report_{timestamp}.md` - Markdown格式详细报告
   - `{strategy}_report_{timestamp}.png` - 综合性能图表（6子图）

4. **优化记录** (`src/lstm/data/results/optimization/`):
   - `optimization_log_{timestamp}.json` - 参数优化历史
   - `best_params_{strategy}.json` - 当前最佳参数配置

#### 多模型框架

1. **实验结果** (`src/models/data/results/`):
   - `{model}_{timestamp}.json` - 预测结果和配置

2. **对比报告** (`src/models/data/results/reports/`):
   - `comparison_{timestamp}.md` - 多模型对比报告
   - `comparison_{timestamp}.png` - 对比图表

#### 通用归档

5. **实验归档** (`logs/{date}/experiments/{experiment_id}/`):
   - `config.json` - 完整配置快照
   - `orders.csv` - 所有交易订单
   - `daily_records.csv` - 每日资金/持仓记录
   - `metrics.json` - 回测指标

### 订单归档结构

```
logs/{date}/experiments/{experiment_id}/
├── config.json           # 完整配置快照
├── orders.csv            # 所有订单记录
│   └── columns: date, symbol, action, price, shares, amount, commission, slippage, pnl, pnl_pct, holding_days
├── daily_records.csv     # 每日资金记录
│   └── columns: date, cash, holdings_value, portfolio_value, return, n_positions, position_ratio
└── metrics.json          # 回测指标
    └── keys: total_return, annual_return, sharpe_ratio, max_drawdown, win_rate, n_trades, n_days
```

### 日志输出

回测日志统一输出到 `logs/{date}/lstm_backtest.log`

### 报告内容规范

Markdown报告必须包含：
- 执行概要（时间、期间、资金）
- 核心指标（收益、风险、交易统计）
- 可视化图表引用
- 策略配置详情
- 风险提示

### 可视化图表规范

PNG图表必须包含6个子图：
1. **累计收益曲线** - 显示收益率随时间变化，标注总收益
2. **净值曲线** - 显示资产净值变化
3. **回撤曲线** - 填充图显示回撤幅度，标注最大回撤
4. **每日持仓数量** - 柱状图，显示目标持仓线
5. **日收益分布** - 直方图，标注均值
6. **统计摘要** - 文本框显示关键指标

### 报告数据格式

回测结果JSON必须包含 `daily_data` 字段用于报告生成：

```python
{
    "strategy": "策略名称",
    "config": {
        "top_n": 10,
        "prob_threshold": 0.60,
        "holding_days": 5,
        "commission": 0.001,
        "slippage": 0.001
    },
    "results": {
        "total_return": 0.75,
        "annual_return": 0.85,
        "sharpe_ratio": 1.5,
        "max_drawdown": 0.38,
        "daily_win_rate": 0.55,
        "trade_win_rate": 0.52,
        "n_trades": 1200
    },
    "daily_data": [
        {
            "date": "2025-04-01",
            "cum_return": 0.01,
            "cum_value": 1.01,
            "drawdown": 0.0,
            "daily_return": 0.01,
            "n_positions": 10
        },
        ...
    ]
}
```

## 回测配置

当前策略配置（在 `src/lstm/config.py` 中）：

```python
TRADING_CONFIG = {
    "top_n": 10,              # 每日持仓数量
    "prob_threshold": 0.60,   # 买入概率阈值
    "holding_days": 5,        # 持仓天数
    "commission": 0.001,      # 手续费 0.1%
    "slippage": 0.001,        # 滑点 0.1%
}

# 扩展窗口策略配置
ExpandingWindowConfig = {
    "min_train_days": 60,        # 最小训练天数
    "max_train_days": 500,       # 最大训练天数
    "val_days": 1,               # 验证天数
    "use_sample_weight": True,   # 使用样本权重
    "weight_decay_days": 30,     # 权重衰减周期
    "retrain_interval": 1,       # 重训练间隔
}
```

## 策略逻辑

### 扩展窗口策略

1. **Walk-Forward 训练**: 累积历史数据进行训练
2. **样本加权**: 近期数据权重更高（指数衰减）
3. **选股**: 每日选取预测概率 > 阈值的前 N 只股票
4. **交易**: T 日收盘买入，T+持有天数后收盘卖出
5. **收益计算**: 考虑双边手续费和滑点

## 运行命令

### LSTM 框架

```bash
# 运行基本回测（只生成预测）
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 运行完整回测（计算指标）
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --start_date 2025-04-01 \
    --end_date 2026-01-15 \
    --calculate_metrics

# 运行并更新 CLAUDE.md
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics \
    --update_claude_md

# 自定义交易参数
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --trading_params '{"top_n": 20, "prob_threshold": 0.65, "holding_days": 3}'

# 从结果文件生成可视化报告
python -m src.lstm.experiments.metrics.report_generator \
    --result src/lstm/data/results/v03_repro_xxx.json \
    --show  # 可选，显示图表
```

### 多模型框架

```bash
# 运行 LightGBM 回测
python src/models/scripts/run_lightgbm.py \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 运行 MLP 回测
python src/models/scripts/run_mlp.py \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 运行 Ensemble 回测
python src/models/scripts/run_ensemble.py \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 多模型对比回测
python src/models/scripts/compare_models.py \
    --models lightgbm,mlp,ensemble \
    --start_date 2025-04-01 \
    --end_date 2026-01-15

# 生成可视化报告
python src/models/scripts/generate_report.py \
    --result src/models/data/results/lightgbm_xxx.json \
    --show
```

## 报告生成API

使用 `BacktestReportGenerator` 生成报告：

```python
from src.lstm.experiments.metrics import BacktestReportGenerator

# 创建生成器
generator = BacktestReportGenerator()

# 生成报告（返回 {png: Path, md: Path}）
paths = generator.generate_report(
    daily_data=result['daily_data'],
    results=result['results'],
    config=result['config'],
    strategy_name="expanding_window",
    show=False  # 不显示窗口
)

print(f"PNG: {paths['png']}")
print(f"MD: {paths['md']}")
```

或直接从JSON文件生成：

```python
from src.lstm.experiments.metrics import generate_report_from_json
from pathlib import Path

generate_report_from_json(
    json_path=Path("src/lstm/data/results/backtest_xxx.json"),
    show=True
)
```

## 回测指标

| 指标 | 说明 | 当前表现(v0.3) | 目标 |
|------|------|----------------|------|
| Total Return | 总收益率 | +74.84% | > 50% |
| Annual Return | 年化收益率 | - | > 50% |
| Sharpe Ratio | 夏普比率 | 1.566 | > 1.0 |
| Max Drawdown | 最大回撤 | 47.04% | < 50% |
| Win Rate | 胜率 | - | > 50% |

## Walk-Forward 验证

回测采用严格的 Walk-Forward 方法避免前瞻偏差：

```
累积扩展窗口：
[--训练集(累积增长)--][验证][测试] → 向前滚动
```

- 每个测试日使用之前的所有历史数据训练模型
- 模型从未"看到"测试期数据
- 真实模拟实盘情况

## 回测结果示例

```json
{
  "strategy": "expanding_window",
  "config": {...},
  "predictions": [
    {
      "date": "2025-04-01",
      "symbol": "SZSE.000001",
      "prob": 0.65
    }
  ],
  "retrain_dates": ["2025-04-01", ...],
  "performance_history": [
    {
      "date": "2025-04-01",
      "train_loss": 0.65,
      "val_loss": 0.67,
      "val_acc": 0.542
    }
  ],
  "metrics": {
    "total_return": 0.7484,
    "annual_return": 0.8542,
    "sharpe_ratio": 1.566,
    "max_drawdown": 0.4704,
    "win_rate": 0.548,
    "n_trades": 1250
  }
}
```

## 验证要点

数据校验角色应检查：

- [ ] 交易记录完整（无缺失日期）
- [ ] 手续费计算正确（买入+卖出双边）
- [ ] 持仓逻辑正确（不超过最大持仓数）
- [ ] 指标计算正确（收益率、夏普比率、回撤）
- [ ] 无前瞻偏差（训练日期 < 测试日期）
- [ ] 预测概率在 [0, 1] 范围内

## 结果分析

### 好的回测结果
- 收益曲线稳定上升
- 回撤幅度和持续时间可控
- 夏普比率 > 1.0
- 胜率在合理范围（50%-60%）

### 需要警惕的信号
- 收益率异常高（可能存在数据泄露）
- 胜率过高（>70%，可能有bug）
- 回撤过大（>50%，风险控制不足）
- 交易次数过少（策略不够活跃）

## 优化方向

如果回测结果不理想：

1. **调整选股阈值**: 修改 `prob_threshold`
2. **优化持仓配置**: 调整 `top_n` 和 `holding_days`
3. **改进模型**: 调整模型参数、训练策略
4. **样本权重**: 调整 `weight_decay_days` 和 `weight_decay_rate`
5. **数据质量**: 检查特征工程和数据清洗

## 查看结果

```bash
# 查看实验结果
ls src/lstm/data/results/experiments/

# 查看 JSON 结果
cat src/lstm/data/results/experiments/expanding_window_*.json
```

## 下一步

回测完成后：
1. 分析回测报告，评估策略有效性
2. 如果结果不理想，调整参数或回到训练环节优化
3. 如果结果理想，可以考虑小资金实盘验证
4. 持续监控实盘表现与回测的差异

## 相关文档

- 快速入门：`docs/QUICKSTART_LSTM.md`
- LSTM 框架指南：`src/lstm/README.md`
- 多模型配置：`src/models/config.py`
- 实验框架：`docs/EXPERIMENT_FRAMEWORK.md`
