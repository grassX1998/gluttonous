"""
完整回测引擎

记录所有交易订单、持仓信息、收益率变化
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import polars as pl

from pipeline.shared.logging_config import get_lstm_backtest_logger
from pipeline.shared.config import PROJECT_ROOT


class BacktestEngine:
    """完整回测引擎"""

    def __init__(self, initial_capital: float = 1000000.0,
                 top_n: int = 10,
                 prob_threshold: float = 0.70,
                 commission: float = 0.001,
                 slippage: float = 0.001,
                 trailing_stop_pct: float = 0.05,
                 max_holding_days: int = 10,
                 min_holding_days: int = 1,
                 exit_on_low_prob: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        初始化回测引擎（动态回撤止盈止损）

        Args:
            initial_capital: 初始资金
            top_n: 每日最多持仓数
            prob_threshold: 概率阈值
            commission: 手续费率
            slippage: 滑点
            trailing_stop_pct: 动态回撤比例（如0.05表示5%）
            max_holding_days: 最长持有天数
            min_holding_days: 最短持有天数
            exit_on_low_prob: 当预测概率低于阈值时是否卖出
            logger: 日志记录器，如果为 None 则使用默认的 LSTM 回测日志器
        """
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.prob_threshold = prob_threshold
        self.commission = commission
        self.slippage = slippage
        self.trailing_stop_pct = trailing_stop_pct
        self.max_holding_days = max_holding_days
        self.min_holding_days = min_holding_days
        self.exit_on_low_prob = exit_on_low_prob

        # 日志记录器
        self.logger = logger or get_lstm_backtest_logger()

        # 状态变量
        self.cash = initial_capital
        self.positions = {}  # {symbol: {'shares': N, 'entry_date': date, 'entry_price': price}}
        self.portfolio_value = initial_capital

        # 记录
        self.orders = []  # 所有交易订单
        self.daily_records = []  # 每日资金/持仓记录
        self.position_history = []  # 持仓历史

    def load_price_data(self, data_dir: Path, start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载价格数据

        Args:
            data_dir: 数据目录
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            DataFrame with columns: date, symbol, close, open
        """
        # 从特征数据中加载价格
        from src.lstm.config import FEATURE_DATA_MONTHLY_DIR

        # 获取需要的月份
        start_ym = start_date[:7]
        end_ym = end_date[:7]

        available_months = sorted([p.stem for p in FEATURE_DATA_MONTHLY_DIR.glob("*.parquet")])
        needed_months = [m for m in available_months if start_ym <= m <= end_ym]

        # 加载数据
        dfs = []
        for month in needed_months:
            month_file = FEATURE_DATA_MONTHLY_DIR / f"{month}.parquet"
            if month_file.exists():
                df = pl.read_parquet(month_file)
                # 只保留需要的列
                if all(col in df.columns for col in ['date', 'symbol', 'close', 'open']):
                    df = df.select(['date', 'symbol', 'close', 'open'])
                    dfs.append(df)

        if not dfs:
            return None

        # 合并并过滤日期
        df = pl.concat(dfs)
        df = df.with_columns(pl.col("date").cast(pl.Utf8).alias("date_str"))
        df = df.filter(
            (pl.col("date_str") >= start_date) & (pl.col("date_str") <= end_date)
        ).drop("date_str")

        return df.sort(['date', 'symbol'])

    def get_price(self, symbol: str, date: str, price_data: pl.DataFrame,
                  price_type: str = 'close') -> float:
        """
        获取指定股票在指定日期的价格

        Args:
            symbol: 股票代码
            date: 日期
            price_data: 价格数据
            price_type: 价格类型 ('close' or 'open')

        Returns:
            价格，如果没有数据返回None
        """
        # 将date转换为字符串进行比较
        date_str = str(date) if not isinstance(date, str) else date

        row = price_data.filter(
            (pl.col('symbol') == symbol) &
            (pl.col('date').cast(pl.Utf8) == date_str)
        )

        if row.height > 0:
            return float(row[price_type][0])
        return None

    def open_position(self, symbol: str, date: str, price: float, shares: int):
        """
        开仓

        Args:
            symbol: 股票代码
            date: 日期
            price: 价格
            shares: 股数
        """
        cost = price * shares * (1 + self.commission + self.slippage)

        if cost > self.cash:
            # 资金不足，调整股数
            shares = int(self.cash / (price * (1 + self.commission + self.slippage)))
            cost = price * shares * (1 + self.commission + self.slippage)

        if shares <= 0:
            return

        self.cash -= cost
        self.positions[symbol] = {
            'shares': shares,
            'entry_date': date,
            'entry_price': price,
            'cost': cost,
            'highest_price': price  # 记录最高价，用于动态回撤
        }

        # 记录订单
        self.orders.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'amount': cost,
            'commission': price * shares * self.commission,
            'slippage': price * shares * self.slippage
        })

    def close_position(self, symbol: str, date: str, price: float):
        """
        平仓

        Args:
            symbol: 股票代码
            date: 日期
            price: 价格
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        shares = position['shares']
        proceeds = price * shares * (1 - self.commission - self.slippage)

        self.cash += proceeds

        # 计算盈亏
        pnl = proceeds - position['cost']
        pnl_pct = pnl / position['cost']

        # 记录订单
        self.orders.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'amount': proceeds,
            'commission': price * shares * self.commission,
            'slippage': price * shares * self.slippage,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_days': self._get_holding_days(position['entry_date'], date)
        })

        del self.positions[symbol]

    def _get_holding_days(self, entry_date: str, exit_date: str) -> int:
        """计算持有天数"""
        try:
            d1 = datetime.strptime(entry_date, '%Y-%m-%d')
            d2 = datetime.strptime(exit_date, '%Y-%m-%d')
            return (d2 - d1).days
        except:
            return 0

    def _should_close_position(self, symbol: str, date: str, price_data: pl.DataFrame,
                              current_prob: float = None) -> tuple:
        """
        判断是否应该平仓（动态回撤止盈止损）

        Args:
            symbol: 股票代码
            date: 当前日期
            price_data: 价格数据
            current_prob: 当日预测概率（可选）

        Returns:
            (should_close, reason) - 是否平仓及原因
        """
        if symbol not in self.positions:
            return False, None

        position = self.positions[symbol]
        holding_days = self._get_holding_days(position['entry_date'], date)

        # 1. 检查最短持有期
        if holding_days < self.min_holding_days:
            return False, None

        # 2. 获取当前价格
        current_price = self.get_price(symbol, date, price_data, 'open')
        if current_price is None:
            return False, None

        # 3. 更新最高价
        if current_price > position['highest_price']:
            position['highest_price'] = current_price

        # 4. 计算回撤
        drawdown_from_peak = (position['highest_price'] - current_price) / position['highest_price']
        loss_from_entry = (position['entry_price'] - current_price) / position['entry_price']

        # 5. 动态回撤止盈（从最高点回撤超过阈值）
        if drawdown_from_peak >= self.trailing_stop_pct:
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            return True, f"回撤止盈 (峰值回撤{drawdown_from_peak*100:.2f}%, 盈亏{pnl_pct*100:+.2f}%)"

        # 6. 动态回撤止损（从入场价下跌超过阈值）
        if loss_from_entry >= self.trailing_stop_pct:
            return True, f"回撤止损 (跌幅{loss_from_entry*100:.2f}%)"

        # 7. 检查最长持有期
        if holding_days >= self.max_holding_days:
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            return True, f"最长持有 ({holding_days}天, 盈亏{pnl_pct*100:+.2f}%)"

        # 8. 检查预测概率下降（可选）
        if self.exit_on_low_prob and current_prob is not None:
            if current_prob < self.prob_threshold:
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                return True, f"概率下降 ({current_prob:.3f} < {self.prob_threshold}, 盈亏{pnl_pct*100:+.2f}%)"

        return False, None

    def update_portfolio_value(self, date: str, price_data: pl.DataFrame):
        """
        更新投资组合价值

        Args:
            date: 当前日期
            price_data: 价格数据
        """
        holdings_value = 0.0
        position_details = []

        for symbol, position in self.positions.items():
            current_price = self.get_price(symbol, date, price_data, 'close')
            if current_price is None:
                # 如果没有价格数据，使用入场价格
                current_price = position['entry_price']

            value = current_price * position['shares']
            holdings_value += value

            unrealized_pnl = value - position['cost']
            unrealized_pnl_pct = unrealized_pnl / position['cost']

            position_details.append({
                'symbol': symbol,
                'shares': position['shares'],
                'entry_price': position['entry_price'],
                'current_price': current_price,
                'cost': position['cost'],
                'value': value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': unrealized_pnl_pct,
                'entry_date': position['entry_date']
            })

        self.portfolio_value = self.cash + holdings_value

        # 记录每日状态
        self.daily_records.append({
            'date': date,
            'cash': self.cash,
            'holdings_value': holdings_value,
            'portfolio_value': self.portfolio_value,
            'return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'n_positions': len(self.positions),
            'position_ratio': holdings_value / self.portfolio_value if self.portfolio_value > 0 else 0
        })

        # 记录持仓详情
        self.position_history.append({
            'date': date,
            'positions': position_details.copy()
        })

    def run_backtest(self, predictions: List[Dict], price_data: pl.DataFrame,
                    trading_dates: List[str]) -> Dict[str, Any]:
        """
        运行完整回测

        Args:
            predictions: 预测列表 [{date, symbol, prob}, ...]
            price_data: 价格数据
            trading_dates: 交易日列表

        Returns:
            回测结果
        """
        # 将预测转换为DataFrame
        pred_df = pl.DataFrame(predictions)

        self.logger.info("[Backtest] Starting backtest...")
        self.logger.info(f"  Initial capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"  Trading dates: {len(trading_dates)}")

        for i, date in enumerate(trading_dates):
            # 1. 获取当日预测（先获取，用于判断是否平仓）
            day_preds = pred_df.filter(pl.col('date') == date)

            # 创建预测概率字典，用于平仓判断
            prob_map = {}
            if day_preds.height > 0:
                for row in day_preds.iter_rows(named=True):
                    prob_map[row['symbol']] = row['prob']

            # 2. 动态评估是否需要平仓
            symbols_to_close = []
            close_reasons = {}
            for symbol in list(self.positions.keys()):
                current_prob = prob_map.get(symbol)  # 获取当日预测概率
                should_close, reason = self._should_close_position(
                    symbol, date, price_data, current_prob
                )
                if should_close:
                    symbols_to_close.append(symbol)
                    close_reasons[symbol] = reason

            # 执行平仓
            for symbol in symbols_to_close:
                price = self.get_price(symbol, date, price_data, 'open')
                if price is not None:
                    self.close_position(symbol, date, price)

            # 3. 选择新股票开仓
            if day_preds.height > 0:
                # 过滤概率阈值
                day_preds = day_preds.filter(pl.col('prob') >= self.prob_threshold)
                day_preds = day_preds.sort('prob', descending=True)

                # 选择top N
                available_slots = self.top_n - len(self.positions)
                if available_slots > 0:
                    top_stocks = day_preds.head(available_slots)

                    # 计算每只股票的仓位
                    if len(top_stocks) > 0:
                        # 将可用资金平均分配
                        cash_per_stock = self.cash / len(top_stocks)

                        for row in top_stocks.iter_rows(named=True):
                            symbol = row['symbol']
                            if symbol not in self.positions:
                                price = self.get_price(symbol, date, price_data, 'close')
                                if price is not None and price > 0:
                                    shares = int(cash_per_stock / (price * (1 + self.commission + self.slippage)))
                                    if shares > 0:
                                        self.open_position(symbol, date, price, shares)

            # 4. 更新组合价值
            self.update_portfolio_value(date, price_data)

            # 进度显示
            if (i + 1) % 20 == 0 or i == len(trading_dates) - 1:
                progress = (i + 1) / len(trading_dates) * 100
                self.logger.info(f"  Progress: {i+1}/{len(trading_dates)} ({progress:.1f}%) - "
                      f"Portfolio: ${self.portfolio_value:,.2f} "
                      f"(Return: {(self.portfolio_value/self.initial_capital-1)*100:+.2f}%)")

        # 平掉所有剩余持仓
        final_date = trading_dates[-1]
        remaining_symbols = list(self.positions.keys())
        for symbol in remaining_symbols:
            if symbol in self.positions:  # 检查是否还在持仓中
                price = self.get_price(symbol, final_date, price_data, 'close')
                if price is not None:
                    self.close_position(symbol, final_date, price)
                else:
                    # 如果没有价格，使用入场价格强制平仓
                    position = self.positions.get(symbol)
                    if position:
                        self.close_position(symbol, final_date, position['entry_price'])

        # 最终更新
        self.update_portfolio_value(final_date, price_data)

        self.logger.info("[Backtest] Completed!")
        self.logger.info(f"  Final portfolio value: ${self.portfolio_value:,.2f}")
        self.logger.info(f"  Total return: {(self.portfolio_value/self.initial_capital-1)*100:+.2f}%")
        self.logger.info(f"  Total trades: {len([o for o in self.orders if o['action'] == 'BUY'])}")

        return {
            'orders': self.orders,
            'daily_records': self.daily_records,
            'position_history': self.position_history,
            'final_value': self.portfolio_value,
            'total_return': (self.portfolio_value - self.initial_capital) / self.initial_capital,
            'n_trades': len([o for o in self.orders if o['action'] == 'BUY'])
        }

    def calculate_metrics(self) -> Dict[str, float]:
        """计算回测指标"""
        if not self.daily_records:
            return {}

        returns = [r['return'] for r in self.daily_records]
        portfolio_values = [r['portfolio_value'] for r in self.daily_records]

        # 日收益率
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            daily_returns.append(daily_ret)

        # 计算指标
        total_return = returns[-1] if returns else 0
        n_days = len(self.daily_records)

        # 年化收益
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # 夏普比率
        if len(daily_returns) > 0:
            mean_daily_return = np.mean(daily_returns)
            std_daily_return = np.std(daily_returns)
            sharpe_ratio = mean_daily_return / std_daily_return * np.sqrt(252) if std_daily_return > 0 else 0
        else:
            sharpe_ratio = 0

        # 最大回撤
        peak = self.initial_capital
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 胜率
        winning_trades = [o for o in self.orders if o['action'] == 'SELL' and o.get('pnl', 0) > 0]
        total_trades = [o for o in self.orders if o['action'] == 'SELL']
        win_rate = len(winning_trades) / len(total_trades) if total_trades else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': len([o for o in self.orders if o['action'] == 'BUY']),
            'n_days': n_days
        }

    def archive_results(self, experiment_id: str, config: Dict[str, Any] = None) -> Path:
        """
        归档回测结果到标准目录

        归档结构:
        logs/{date}/experiments/{experiment_id}/
            ├── config.json       # 完整配置快照
            ├── orders.csv        # 所有订单
            ├── daily_records.csv # 每日资金记录
            └── metrics.json      # 回测指标

        Args:
            experiment_id: 实验 ID
            config: 配置字典（可选）

        Returns:
            归档目录路径
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        archive_dir = PROJECT_ROOT / "logs" / date_str / "experiments" / experiment_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存配置
        if config is None:
            config = {
                'initial_capital': self.initial_capital,
                'top_n': self.top_n,
                'prob_threshold': self.prob_threshold,
                'commission': self.commission,
                'slippage': self.slippage,
                'trailing_stop_pct': self.trailing_stop_pct,
                'max_holding_days': self.max_holding_days,
                'min_holding_days': self.min_holding_days,
                'exit_on_low_prob': self.exit_on_low_prob
            }

        config_file = archive_dir / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # 2. 保存订单
        if self.orders:
            orders_file = archive_dir / "orders.csv"
            fieldnames = list(self.orders[0].keys())
            with open(orders_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.orders)
            self.logger.info(f"  Archived {len(self.orders)} orders to {orders_file}")

        # 3. 保存每日记录
        if self.daily_records:
            daily_file = archive_dir / "daily_records.csv"
            fieldnames = list(self.daily_records[0].keys())
            with open(daily_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.daily_records)
            self.logger.info(f"  Archived {len(self.daily_records)} daily records to {daily_file}")

        # 4. 保存指标
        metrics = self.calculate_metrics()
        metrics_file = archive_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        self.logger.info(f"[Backtest] Results archived to: {archive_dir}")

        return archive_dir
