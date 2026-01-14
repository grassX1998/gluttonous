# Gluttonous v0.0 - 原始策略归档

**归档日期**: 2026-01-14

这是项目早期的基金和股票策略，基于 Futu API 实现。

## 目录结构

```
v0.0/
├── fund/                   # 基金策略
│   ├── back_510880.ipynb   # 510880 回测
│   ├── backtest_avg_1.ipynb
│   ├── backtest_avg_2.ipynb
│   ├── backtest_max_1.ipynb
│   ├── str_fund_max_1.py   # 基金最大值策略
│   └── on_recv.py          # 实时信号接收
├── stock/                  # 股票策略
│   ├── backtest_stock_02.ipynb
│   ├── backtest_stock_max_1.ipynb
│   └── stock_on_recv.py    # 股票信号接收
├── docs/                   # 策略文档
│   └── strategy/
│       ├── fund_max_1.md
│       └── stock_volume_1.md
├── test.ipynb              # 测试 notebook
├── roll_test.ipynb         # 滚动测试
└── check.py                # 检查脚本
```

## 策略说明

### 基金策略 (fund_max_1)
- 基于 510880 (红利ETF) 的高抛低吸策略
- 使用 25 日最高价作为止盈参考

### 股票策略 (stock_volume_1)
- 基于成交量的股票选择策略
- 使用 Futu API 获取实时行情

## 依赖

- Futu OpenD (本地运行)
- futu-api Python SDK

## 注意

这些策略已不再维护，仅作为历史参考。当前活跃版本请参考 v0.3。
