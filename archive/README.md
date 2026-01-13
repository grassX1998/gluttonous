# 归档文件

本目录包含已归档的文件，这些文件保留用于参考，但不再活跃使用。

## 实时监控脚本

| 文件 | 描述 |
|------|------|
| `on_recv.py` | 基金策略实时行情接收器，订阅 ETF 和指数行情，触发 `StrFundMax1` 策略 |
| `stock_on_recv.py` | 个股策略实时行情接收器，监控中证1000成分股的分钟级别放量信号 |
| `check.py` | 策略检查脚本，用于加载和验证策略订阅列表 |

## 策略实现

| 文件 | 描述 |
|------|------|
| `str_fund_max_1.py` | 基金突破策略实现，基于指数25日最高价突破 + 成交量放大信号 |

## 回测 Notebooks

| 文件 | 描述 |
|------|------|
| `back_510880.ipynb` | 沪深300ETF (510880) 回测实验 |
| `backtest_avg_1.ipynb` | 均线策略回测 v1 |
| `backtest_avg_2.ipynb` | 均线策略回测 v2 |
| `backtest_max_1.ipynb` | 最高价突破策略回测 |
| `backtest_stock_02.ipynb` | 个股策略回测 v2 |
| `backtest_stock_max_1.ipynb` | 个股最高价突破策略回测 |
| `roll_test.ipynb` | 滚动测试实验 |
| `test.ipynb` | 通用测试 notebook |

## 注意事项

- 实时监控脚本包含已废弃的 QQ Bot 消息推送逻辑
- 运行这些脚本需要 Futu OpenD 本地服务
- 策略文档已整理至 `docs/strategy/` 目录
