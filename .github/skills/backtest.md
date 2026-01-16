# 策略回测 (Backtest)

使用训练好的模型进行历史回测，评估策略表现。

## 职责

基于训练好的模型，在历史数据上模拟真实交易，评估策略的收益、风险和稳定性。

## 输入

- **模型**: `.pipeline_data/checkpoints/best_model.pt`
- **标准化参数**: `.pipeline_data/checkpoints/X_mean.npy`, `X_std.npy`
- **清洗数据**: `.pipeline_data/cleaned/*.parquet`
- **特征数据**: `.pipeline_data/features/*.parquet`

## 输出

- **回测报告**: `.pipeline_data/backtest_results/backtest_v5_{timestamp}.json`
- **详细数据**: 包含每日收益、累计收益、回撤等信息

## 回测配置

当前策略配置（在 `backtest_v5.py` 中）：

```python
Config = {
    "TOP_N": 10,              # 每日持仓数量
    "PROB_THRESHOLD": 0.60,   # 买入概率阈值
    "HOLDING_DAYS": 5,        # 持仓天数
    "COMMISSION": 0.001,      # 手续费 0.1%
    "SLIPPAGE": 0.001,        # 滑点 0.1%
    "TRAIN_MONTHS": 6,        # 训练窗口月数
    "VAL_MONTHS": 1,          # 验证窗口月数
    "SAMPLE_RATIO": 0.5,      # 训练采样比例
}
```

传统回测配置（在 `pipeline/shared/config.py` 中）：

```python
BACKTEST_CONFIG = {
    "initial_cash": 1_000_000,   # 初始资金100万
    "max_positions": 10,         # 最大持仓10只
    "commission_rate": 0.0003,   # 手续费万三
    "slippage": 0.001,           # 滑点千一
    "min_probability": 0.6,      # 最小买入概率
    "stop_loss": -0.08,          # 止损8%
    "take_profit": 0.15,         # 止盈15%
    "max_holding_days": 20,      # 最大持仓20天
}
```

## 策略逻辑

### backtest_v5.py（当前主要使用）

1. **Walk-Forward训练**: 按月滚动训练模型
2. **选股**: 每日选取预测概率 > 0.60 的前10只股票
3. **交易**: T日收盘买入，T+5日收盘卖出
4. **收益计算**: 考虑双边手续费和滑点

### 传统回测逻辑

1. **选股**: 模型预测上涨概率 > `min_probability` 的股票
2. **买入**: 等权分配资金，每次买入整百股
3. **卖出条件**:
   - 触发止损 (`profit < stop_loss`)
   - 触发止盈 (`profit > take_profit`)
   - 达到最大持仓天数
   - 模型信号反转

## 运行命令

```powershell
# 主要回测脚本（推荐）
python backtest_v5.py

# 生成图表
python plot_backtest.py

# 传统Pipeline回测
python -m pipeline.backtest.backtest --start_date 2025-01-01 --end_date 2025-12-31
```

## 回测指标

| 指标 | 说明 | 当前表现(v0.3) | 基准 |
|------|------|----------------|------|
| Total Return | 总收益率 | +74.84% | > 沪深300 |
| Annual Return | 年化收益率 | - | > 10% |
| Sharpe Ratio | 夏普比率 | 1.566 | > 1.0 |
| Max Drawdown | 最大回撤 | 47.04% | < 20% |
| Win Rate | 胜率 | - | > 50% |

## Walk-Forward验证

回测采用严格的Walk-Forward方法避免前瞻偏差：

```
时间线：[---6个月训练---][1个月验证][1个月测试] → 滚动前进
```

- 每个测试月使用之前的数据训练模型
- 模型从未"看到"测试期数据
- 真实模拟实盘情况

## 回测报告示例

```json
{
  "config": {
    "TOP_N": 10,
    "PROB_THRESHOLD": 0.60,
    "HOLDING_DAYS": 5
  },
  "results": {
    "total_return": 0.7484,
    "annual_return": 0.8542,
    "sharpe_ratio": 1.566,
    "max_drawdown": 0.4704,
    "daily_win_rate": 0.612,
    "trade_win_rate": 0.548,
    "n_trades": 1250,
    "n_days": 180
  },
  "daily_data": [...]
}
```

## 验证要点

数据校验角色应检查：

- [ ] 初始资金设置正确
- [ ] 交易记录完整（无缺失日期）
- [ ] 手续费计算正确（买入+卖出双边）
- [ ] 持仓逻辑正确（不超过最大持仓数）
- [ ] 指标计算正确（收益率、夏普比率、回撤）
- [ ] 无前瞻偏差（训练日期 < 测试日期）

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

1. **调整选股阈值**: 修改 `PROB_THRESHOLD`
2. **优化持仓配置**: 调整 `TOP_N` 和 `HOLDING_DAYS`
3. **改进模型**: 增加特征、调整模型结构
4. **风险控制**: 调整止损止盈参数
5. **数据质量**: 检查特征工程和数据清洗

## 可视化

运行 `plot_backtest.py` 生成图表：
- 累计收益曲线
- 回撤曲线
- 每日收益分布
- 持仓统计

## 下一步

回测完成后：
1. 分析回测报告，评估策略有效性
2. 如果结果不理想，回到训练或特征工程环节优化
3. 如果结果理想，可以考虑小资金实盘验证
4. 持续监控实盘表现与回测的差异
