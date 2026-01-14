"""
策略回测模块

主要组件:
- Backtester: 主回测器，整合信号生成和回测执行
- StrategyExecutor: 策略执行器，用于生成交易信号（可选）
"""

from .backtest import Backtester, Trade
from .strategy_executor import StrategyExecutor, Signal, StrategyState

__all__ = ["Backtester", "Trade", "StrategyExecutor", "Signal", "StrategyState"]
