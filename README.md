# Gluttonous

基于机器学习的 A 股量化选股系统。

## 当前版本：v0.3

**全市场选股策略** - 使用 LSTM 模型预测 5 日涨跌

| 指标 | 结果 |
|------|------|
| 收益率 | +74.84% |
| Sharpe | 1.566 |
| 最大回撤 | 47.04% |
| 测试期间 | 2025-04 ~ 2026-01 |

## 快速开始

```bash
# 1. 数据清洗
python -m pipeline.data_cleaning.clean

# 2. 特征工程
python -m pipeline.data_cleaning.features

# 3. 运行回测（含 Walk-Forward 训练）
python backtest_v5.py

# 4. 生成图表
python plot_backtest.py
```

## 项目结构

```
gluttonous/
├── backtest_v5.py          # 主回测脚本
├── plot_backtest.py        # 绘图脚本
├── pipeline/               # 数据处理流水线
│   ├── data_cleaning/      # 数据清洗 & 特征工程
│   ├── data_validation/    # 数据校验
│   ├── training/           # 模型训练
│   └── shared/             # 共享配置
├── docs/                   # 文档 & 图表
├── archive/                # 历史版本归档
│   └── v0.3/              # 当前版本备份
└── .pipeline_data/         # 数据目录（gitignore）
```

## 数据源

NAS 存储：`\\DXP8800PRO-A577\data\stock\gm\`

- 分钟 K 线：2024-06 ~ 2026-01
- 股票数量：~4,210 只（排除 CSI300+500）

## 策略配置

```python
TOP_N = 10              # 每日持仓
PROB_THRESHOLD = 0.60   # 概率阈值
HOLDING_DAYS = 5        # 持有天数
```

## 技术栈

- **数据处理**: Polars
- **深度学习**: PyTorch + CUDA
- **模型**: LSTM (hidden=128, layers=2)

## 文档

- [回测报告](docs/backtest_v5_report.md)
- [版本说明](archive/v0.3/README.md)

## License

MIT
