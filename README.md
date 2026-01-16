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

### 1. 数据准备

```bash
# 数据清洗
python -m pipeline.data_cleaning.clean

# 特征工程
python -m pipeline.data_cleaning.features

# 数据校验
python -m pipeline.data_validation.validate
```

### 2. 模型训练与回测

```bash
# 测试 LSTM 框架
python src/lstm/scripts/test_framework.py

# 运行扩展窗口策略实验
python src/lstm/scripts/run_experiments.py \
    --strategies expanding_window \
    --calculate_metrics

# 详细文档
# - 快速入门: docs/QUICKSTART_LSTM.md
# - 完整指南: src/lstm/README.md
```

## 项目结构

```
gluttonous/
├── src/
│   └── lstm/              # LSTM 训练框架
│       ├── config.py      # 配置文件
│       ├── models/        # 模型定义
│       ├── experiments/   # 实验框架
│       ├── scripts/       # 运行脚本
│       └── data/          # 数据目录（gitignore）
├── pipeline/              # 数据处理流水线
│   ├── data_cleaning/     # 数据清洗 & 特征工程
│   ├── data_validation/   # 数据校验
│   └── shared/            # 共享配置
├── docs/                  # 文档 & 图表
├── archive/               # 历史版本归档
│   └── v0.3/             # 当前版本备份
└── .pipeline_data/        # 数据目录（gitignore）
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
