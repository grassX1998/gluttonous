"""
回测模块 - 使用训练好的LSTM模型进行策略回测

整合了信号生成和回测执行的完整流程
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, date as dt_date
import json

import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    MODEL_CHECKPOINT_DIR, BACKTEST_RESULT_DIR, BACKTEST_CONFIG,
    FEATURE_DATA_DIR, DEVICE
)
from pipeline.shared.utils import setup_logger, timer
from pipeline.data_cleaning.features import FEATURE_COLS, STRATEGY_PARAMS


logger = setup_logger("backtest", BACKTEST_RESULT_DIR / "backtest.log")


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float
    profit_pct: float
    holding_days: int
    predicted_prob: float
    exit_reason: str


class Backtester:
    """回测器 - 使用LSTM模型进行策略回测"""
    
    def __init__(self, config: dict | None = None):
        self.config = config or BACKTEST_CONFIG
        
        # 资金管理
        self.initial_cash = self.config.get("initial_cash", 1_000_000)
        self.min_probability = self.config.get("min_probability", 0.55)
        self.max_positions = self.config.get("max_positions", 5)
        self.holding_days = STRATEGY_PARAMS["holding_days"]
        self.commission = self.config.get("commission_rate", 0.0003)
        self.slippage = self.config.get("slippage", 0.001)
        
        # 模型相关
        self.model = None
        self.X_mean = None
        self.X_std = None
        self.lookback = self.config.get("lookback_days", 20)
        
        logger.info("="*60)
        logger.info("Backtester initialized")
        logger.info(f"Initial Cash: {self.initial_cash:,}")
        logger.info(f"Max Positions: {self.max_positions}")
        logger.info(f"Min Probability: {self.min_probability}")
        logger.info(f"Holding Days: {self.holding_days}")
        logger.info("="*60)
    
    @timer
    def load_model(self):
        """加载训练好的模型"""
        from pipeline.training.train import LSTMModel
        
        checkpoint_path = MODEL_CHECKPOINT_DIR / "best_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError("No trained model found. Run training first.")
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model_config = checkpoint['config']
        
        # 加载标准化参数
        X_mean_path = MODEL_CHECKPOINT_DIR / "X_mean.npy"
        X_std_path = MODEL_CHECKPOINT_DIR / "X_std.npy"
        
        if X_mean_path.exists() and X_std_path.exists():
            self.X_mean = np.load(X_mean_path)
            self.X_std = np.load(X_std_path)
            input_size = len(self.X_mean)
        else:
            input_size = len(FEATURE_COLS)
        
        # 重建模型
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"]
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        
        logger.info(f"Model loaded: epoch={checkpoint['epoch']}, val_acc={checkpoint['val_acc']:.4f}")
    
    @timer
    def load_feature_data(self) -> pl.DataFrame:
        """加载特征数据"""
        feature_file = FEATURE_DATA_DIR / "features_all.parquet"
        
        if feature_file.exists():
            df = pl.read_parquet(feature_file)
            logger.info(f"Loaded features_all.parquet: {len(df):,} rows")
        else:
            # 合并单个特征文件
            logger.info("Merging individual feature files...")
            feature_files = [f for f in FEATURE_DATA_DIR.glob("*.parquet") 
                           if f.stem != "features_all"]
            
            if not feature_files:
                raise FileNotFoundError("No feature files found. Run feature engineering first.")
            
            all_dfs = [pl.read_parquet(f) for f in feature_files]
            df = pl.concat(all_dfs)
            
            # 保存合并文件供下次使用
            df.write_parquet(feature_file)
            logger.info(f"Created and saved features_all.parquet: {len(df):,} rows")
        
        return df
    
    @timer
    def generate_signals(self, df: pl.DataFrame, start_date: str, end_date: str) -> pl.DataFrame:
        """为所有股票生成交易信号 (仅生成回测日期范围内的信号)"""
        from datetime import date as dt_date, timedelta
        
        start_dt = dt_date.fromisoformat(start_date)
        end_dt = dt_date.fromisoformat(end_date)
        
        logger.info(f"Generating signals for {start_date} to {end_date}...")
        
        # 需要多预留lookback天数的数据用于特征窗口（考虑节假日多预留50%）
        buffer_days = int(self.lookback * 1.5)
        buffer_start = start_dt - timedelta(days=buffer_days)
        
        df = df.filter(pl.col("date") >= buffer_start).sort(["symbol", "date"])
        
        symbols = df.select("symbol").unique().to_series().to_list()
        signals = []
        
        logger.info(f"Processing {len(symbols)} symbols...")
        
        for i, symbol in enumerate(symbols):
            if (i + 1) % 500 == 0:
                logger.info(f"Signal generation: {i+1}/{len(symbols)}")
            
            stock_df = df.filter(pl.col("symbol") == symbol).sort("date")
            
            if len(stock_df) < self.lookback + 5:
                continue
            
            # 获取特征矩阵和日期
            features = stock_df.select(FEATURE_COLS).to_numpy()
            dates_list = stock_df["date"].to_list()
            closes = stock_df["close"].to_list()
            opens = stock_df["open"].to_list()
            
            # 获取策略特征
            breakouts = stock_df["price_breakout"].to_list() if "price_breakout" in stock_df.columns else [0] * len(stock_df)
            vol_ratios = stock_df["volume_ratio_10"].to_list() if "volume_ratio_10" in stock_df.columns else [0] * len(stock_df)
            # 涨停标记
            limit_ups = stock_df["is_limit_up"].to_list() if "is_limit_up" in stock_df.columns else [0] * len(stock_df)
            
            # 标准化
            if self.X_mean is not None:
                features = (features - self.X_mean) / (self.X_std + 1e-8)
            features = np.nan_to_num(features, 0)
            
            # 收集需要预测的样本
            samples = []
            sample_indices = []
            
            for j in range(self.lookback, len(stock_df)):
                d = dates_list[j]
                if d < start_dt or d > end_dt:
                    continue
                samples.append(features[j-self.lookback:j])
                sample_indices.append(j)
            
            if not samples:
                continue
            
            # 批量预测
            X_batch = np.array(samples)
            X_tensor = torch.from_numpy(X_batch).float().to(DEVICE)
            
            with torch.no_grad():
                outputs = self.model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            
            # 构建信号
            for k, j in enumerate(sample_indices):
                vol_ratio = vol_ratios[j] if j < len(vol_ratios) else 0
                is_limit_up = limit_ups[j] == 1.0 if j < len(limit_ups) else False
                signals.append({
                    "symbol": symbol,
                    "date": dates_list[j],
                    "close": closes[j],
                    "open": opens[j],
                    "prob": float(probs[k]),
                    "price_breakout": breakouts[j] == 1.0 if j < len(breakouts) else False,
                    "volume_confirm": vol_ratio >= STRATEGY_PARAMS["vol_ratio"],
                    "vol_ratio": vol_ratio,
                    "is_limit_up": is_limit_up  # 涨停标记
                })
        
        signal_df = pl.DataFrame(signals)
        logger.info(f"Generated {len(signal_df):,} signals")
        return signal_df
    
    @timer
    def run_simulation(self, signal_df: pl.DataFrame, start_date: str, end_date: str):

        """执行回测模拟"""
        logger.info(f"Running simulation from {start_date} to {end_date}")
        
        start_dt = dt_date.fromisoformat(start_date)
        end_dt = dt_date.fromisoformat(end_date)
        
        # 过滤日期范围
        signal_df = signal_df.filter(
            (pl.col("date") >= start_dt) &
            (pl.col("date") <= end_dt)
        ).sort(["date", "prob"], descending=[False, True])
        
        dates = signal_df.select("date").unique().sort("date").to_series().to_list()
        
        if not dates:
            logger.warning("No trading days in the specified period")
            return [], [], self.initial_cash
        
        logger.info(f"Trading days: {len(dates)} ({dates[0]} to {dates[-1]})")
        
        cash = self.initial_cash
        positions = {}  # symbol -> {shares, cost, entry_date, holding_days, prob}
        trades = []
        daily_values = []
        
        for date in dates:
            date_str = str(date)
            day_signals = signal_df.filter(pl.col("date") == date)
            
            # 当日价格映射
            prices = dict(zip(
                day_signals["symbol"].to_list(),
                day_signals["close"].to_list()
            ))
            
            # 1. 检查卖出条件
            sells_to_do = []
            for symbol, pos in positions.items():
                pos["holding_days"] += 1
                current_price = prices.get(symbol, pos["cost"])
                profit = current_price / pos["cost"] - 1
                
                if pos["holding_days"] >= self.holding_days:
                    sells_to_do.append((symbol, current_price, "持仓到期"))
                elif profit <= -0.05:  # 止损
                    sells_to_do.append((symbol, current_price, "止损"))
                elif profit >= 0.15:  # 止盈
                    sells_to_do.append((symbol, current_price, "止盈"))
            
            # 执行卖出
            for symbol, price, reason in sells_to_do:
                pos = positions[symbol]
                sell_price = price * (1 - self.slippage)
                proceeds = pos["shares"] * sell_price * (1 - self.commission)
                profit_pct = sell_price / pos["cost"] - 1
                
                trades.append(Trade(
                    symbol=symbol,
                    buy_date=pos["entry_date"],
                    buy_price=pos["cost"],
                    sell_date=date_str,
                    sell_price=sell_price,
                    profit_pct=profit_pct,
                    holding_days=pos["holding_days"],
                    predicted_prob=pos["prob"],
                    exit_reason=reason
                ))
                
                cash += proceeds
                del positions[symbol]
            
            # 2. 检查买入条件
            if len(positions) < self.max_positions:
                buy_candidates = day_signals.filter(
                    (pl.col("prob") >= self.min_probability) &
                    (pl.col("volume_confirm") | pl.col("price_breakout")) &
                    (pl.col("is_limit_up") == False)  # 排除涨停股，无法买入
                ).to_dicts()
                
                for cand in buy_candidates:
                    if len(positions) >= self.max_positions:
                        break
                    
                    symbol = cand["symbol"]
                    if symbol in positions:
                        continue
                    
                    price = cand["close"]
                    available_slots = self.max_positions - len(positions)
                    position_size = min(cash / available_slots, cash * 0.3)
                    shares = int(position_size / price / 100) * 100
                    
                    if shares <= 0:
                        continue
                    
                    cost = shares * price * (1 + self.commission + self.slippage)
                    if cost > cash:
                        continue
                    
                    cash -= cost
                    positions[symbol] = {
                        "shares": shares,
                        "cost": price,
                        "entry_date": date_str,
                        "holding_days": 0,
                        "prob": cand["prob"]
                    }
            
            # 3. 记录每日净值
            pos_value = sum(
                pos["shares"] * prices.get(sym, pos["cost"])
                for sym, pos in positions.items()
            )
            total_value = cash + pos_value
            
            daily_values.append({
                "date": date_str,
                "value": total_value,
                "cash": cash,
                "positions": len(positions)
            })
        
        # 清仓剩余持仓
        for symbol, pos in list(positions.items()):
            price = prices.get(symbol, pos["cost"])
            sell_price = price * (1 - self.slippage)
            proceeds = pos["shares"] * sell_price * (1 - self.commission)
            profit_pct = sell_price / pos["cost"] - 1
            
            trades.append(Trade(
                symbol=symbol,
                buy_date=pos["entry_date"],
                buy_price=pos["cost"],
                sell_date=str(dates[-1]),
                sell_price=sell_price,
                profit_pct=profit_pct,
                holding_days=pos["holding_days"],
                predicted_prob=pos["prob"],
                exit_reason="回测结束"
            ))
            cash += proceeds
        
        return trades, daily_values, cash
    
    def calculate_metrics(self, trades: list[Trade], daily_values: list[dict], 
                         final_cash: float) -> dict:
        """计算回测指标"""
        total_return = final_cash / self.initial_cash - 1
        
        if trades:
            wins = [t for t in trades if t.profit_pct > 0]
            win_rate = len(wins) / len(trades)
            avg_profit = np.mean([t.profit_pct for t in trades])
            avg_holding = np.mean([t.holding_days for t in trades])
            
            profits = [t.profit_pct for t in trades if t.profit_pct > 0]
            losses = [abs(t.profit_pct) for t in trades if t.profit_pct < 0]
            profit_factor = sum(profits) / sum(losses) if losses else float('inf')
        else:
            win_rate = avg_profit = avg_holding = profit_factor = 0
        
        # 最大回撤
        if daily_values:
            values = [d["value"] for d in daily_values]
            peak = values[0]
            max_dd = 0
            for v in values:
                if v > peak:
                    peak = v
                max_dd = max(max_dd, (peak - v) / peak)
        else:
            max_dd = 0
        
        # 夏普比率
        if len(daily_values) > 1:
            returns = []
            for i in range(1, len(daily_values)):
                r = daily_values[i]["value"] / daily_values[i-1]["value"] - 1
                returns.append(r)
            sharpe = (np.mean(returns) * 250) / (np.std(returns) * np.sqrt(250)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # 年化收益
        days = len(daily_values)
        annual_return = (1 + total_return) ** (250 / max(days, 1)) - 1
        
        return {
            "initial_cash": self.initial_cash,
            "final_value": round(final_cash, 2),
            "total_return": f"{total_return*100:.2f}%",
            "annual_return": f"{annual_return*100:.2f}%",
            "max_drawdown": f"{max_dd*100:.2f}%",
            "sharpe_ratio": round(sharpe, 3),
            "win_rate": f"{win_rate*100:.1f}%",
            "total_trades": len(trades),
            "avg_profit_per_trade": f"{avg_profit*100:.2f}%",
            "avg_holding_days": round(avg_holding, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf"
        }
    
    def save_results(self, metrics: dict, trades: list[Trade], 
                    daily_values: list[dict], start_date: str, end_date: str):
        """保存回测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 打印总结
        logger.info("\n" + "="*60)
        logger.info("Backtest Summary")
        logger.info("="*60)
        for k, v in metrics.items():
            logger.info(f"{k:25}: {v:>15}")
        logger.info("="*60)
        
        # 保存报告
        report = {
            "timestamp": timestamp,
            "period": {"start": start_date, "end": end_date},
            "config": {
                "initial_cash": self.initial_cash,
                "max_positions": self.max_positions,
                "min_probability": self.min_probability,
                "holding_days": self.holding_days
            },
            "metrics": metrics,
            "strategy_params": STRATEGY_PARAMS
        }
        
        BACKTEST_RESULT_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(BACKTEST_RESULT_DIR / f"report_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 保存交易记录
        if trades:
            trades_data = [
                {
                    "symbol": t.symbol,
                    "buy_date": t.buy_date,
                    "buy_price": round(t.buy_price, 2),
                    "sell_date": t.sell_date,
                    "sell_price": round(t.sell_price, 2),
                    "profit_pct": f"{t.profit_pct*100:.2f}%",
                    "holding_days": t.holding_days,
                    "predicted_prob": f"{t.predicted_prob*100:.1f}%",
                    "exit_reason": t.exit_reason
                }
                for t in trades
            ]
            df = pl.DataFrame(trades_data)
            df.write_csv(BACKTEST_RESULT_DIR / f"trades_{timestamp}.csv")
        
        # 保存每日净值
        if daily_values:
            df = pl.DataFrame(daily_values)
            df.write_csv(BACKTEST_RESULT_DIR / f"daily_values_{timestamp}.csv")
        
        # 绘制收益曲线和交易记录图
        self._plot_results(metrics, trades, daily_values, timestamp)
        
        logger.info(f"Results saved to {BACKTEST_RESULT_DIR}")
        return report
    
    def _plot_results(self, metrics: dict, trades: list[Trade], 
                     daily_values: list[dict], timestamp: str):
        """绘制回测结果图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(16, 14))
        
        # ============ 图1: 收益曲线 ============
        ax1 = fig.add_subplot(3, 1, 1)
        
        if daily_values:
            dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_values]
            values = [d["value"] for d in daily_values]
            returns = [(v / self.initial_cash - 1) * 100 for v in values]
            
            ax1.plot(dates, returns, 'b-', linewidth=1.5, label='策略收益')
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # 填充正负收益区域
            ax1.fill_between(dates, returns, 0, 
                            where=[r >= 0 for r in returns], 
                            color='green', alpha=0.2, interpolate=True)
            ax1.fill_between(dates, returns, 0, 
                            where=[r < 0 for r in returns], 
                            color='red', alpha=0.2, interpolate=True)
            
            # 标记买入卖出点
            if trades:
                buy_dates = [datetime.strptime(t.buy_date, "%Y-%m-%d") for t in trades]
                sell_dates = [datetime.strptime(t.sell_date, "%Y-%m-%d") for t in trades]
                
                # 找到对应日期的收益率
                date_to_return = {d: r for d, r in zip(dates, returns)}
                
                buy_returns = []
                for bd in buy_dates:
                    # 找最近的日期
                    closest = min(dates, key=lambda x: abs((x - bd).days))
                    buy_returns.append(date_to_return.get(closest, 0))
                
                sell_returns = []
                for sd in sell_dates:
                    closest = min(dates, key=lambda x: abs((x - sd).days))
                    sell_returns.append(date_to_return.get(closest, 0))
                
                ax1.scatter(buy_dates, buy_returns, marker='^', c='green', 
                           s=30, alpha=0.6, label='买入', zorder=5)
                ax1.scatter(sell_dates, sell_returns, marker='v', c='red', 
                           s=30, alpha=0.6, label='卖出', zorder=5)
            
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        ax1.set_xlabel('日期')
        ax1.set_ylabel('收益率 (%)')
        ax1.set_title(f'回测收益曲线\n'
                     f'总收益: {metrics["total_return"]} | '
                     f'年化: {metrics["annual_return"]} | '
                     f'最大回撤: {metrics["max_drawdown"]} | '
                     f'夏普比率: {metrics["sharpe_ratio"]}', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # ============ 图2: 持仓比例 ============
        ax2 = fig.add_subplot(3, 1, 2)
        
        if daily_values:
            dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in daily_values]
            # 计算持仓比例 = (总市值 - 现金) / 总市值 * 100
            position_pcts = []
            for d in daily_values:
                total_value = d["value"]
                cash = d["cash"]
                pos_pct = (total_value - cash) / total_value * 100 if total_value > 0 else 0
                position_pcts.append(pos_pct)
            
            # 持仓数量
            position_counts = [d["positions"] for d in daily_values]
            
            # 绘制持仓比例面积图
            ax2.fill_between(dates, position_pcts, 0, color='steelblue', alpha=0.4, label='持仓比例')
            ax2.plot(dates, position_pcts, 'steelblue', linewidth=1.5)
            
            # 右轴显示持仓数量
            ax2_right = ax2.twinx()
            ax2_right.bar(dates, position_counts, width=1, color='orange', alpha=0.5, label='持仓数量')
            ax2_right.set_ylabel('持仓数量', color='orange')
            ax2_right.tick_params(axis='y', labelcolor='orange')
            ax2_right.set_ylim(0, self.max_positions + 1)
            
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            ax2.set_ylim(0, 105)
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
            
            # 计算统计信息
            avg_pos_pct = sum(position_pcts) / len(position_pcts) if position_pcts else 0
            max_pos_pct = max(position_pcts) if position_pcts else 0
            min_pos_pct = min(position_pcts) if position_pcts else 0
            
            ax2.set_title(f'持仓比例变化  |  平均: {avg_pos_pct:.1f}%  |  最高: {max_pos_pct:.1f}%  |  最低: {min_pos_pct:.1f}%', fontsize=12)
        
        ax2.set_xlabel('日期')
        ax2.set_ylabel('持仓比例 (%)', color='steelblue')
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # ============ 图3: 交易记录表格 ============
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.axis('off')
        
        if trades:
            # 按卖出日期排序，取最近的20笔
            sorted_trades = sorted(trades, key=lambda x: x.sell_date, reverse=True)[:20]
            
            table_data = []
            for t in sorted_trades:
                profit_str = f"{t.profit_pct*100:+.2f}%"
                prob_str = f"{t.predicted_prob*100:.1f}%"
                table_data.append([
                    t.symbol.split('.')[-1],  # 只显示股票代码
                    t.buy_date,
                    f"{t.buy_price:.2f}",
                    t.sell_date,
                    f"{t.sell_price:.2f}",
                    profit_str,
                    str(t.holding_days),
                    prob_str,
                    t.exit_reason
                ])
            
            columns = ['代码', '买入日期', '买入价', '卖出日期', '卖出价', 
                      '收益率', '持仓天数', '预测概率', '退出原因']
            
            # 创建表格
            table = ax3.table(
                cellText=table_data,
                colLabels=columns,
                loc='center',
                cellLoc='center',
                colColours=['#E6E6FA'] * len(columns)
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # 根据收益率设置行颜色
            for i, t in enumerate(sorted_trades):
                color = '#90EE90' if t.profit_pct > 0 else '#FFB6C1'  # 绿盈红亏
                for j in range(len(columns)):
                    table[(i+1, j)].set_facecolor(color)
            
            ax3.set_title(f'最近20笔交易记录 (共{len(trades)}笔, '
                         f'胜率: {metrics["win_rate"]}, '
                         f'平均收益: {metrics["avg_profit_per_trade"]})', 
                         fontsize=12, pad=20)
        else:
            ax3.text(0.5, 0.5, '无交易记录', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        # 保存图片
        fig_path = BACKTEST_RESULT_DIR / f"backtest_chart_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Chart saved to {fig_path}")
    
    @timer
    def run(self, start_date: str, end_date: str):
        """运行完整回测流程"""
        logger.info("="*60)
        logger.info(f"Starting Backtest: {start_date} to {end_date}")
        logger.info("="*60)
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载数据
        df = self.load_feature_data()
        
        # 3. 生成信号 (仅生成回测期间的信号)
        signal_df = self.generate_signals(df, start_date, end_date)
        
        # 4. 运行模拟
        trades, daily_values, final_cash = self.run_simulation(
            signal_df, start_date, end_date
        )
        
        # 5. 计算指标
        metrics = self.calculate_metrics(trades, daily_values, final_cash)
        
        # 6. 保存结果
        report = self.save_results(metrics, trades, daily_values, start_date, end_date)
        
        return report



def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest Pipeline")
    parser.add_argument("--start_date", type=str, default="2025-01-02",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2025-12-31",
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--initial_cash", type=float, default=1_000_000,
                       help="Initial cash")
    parser.add_argument("--max_positions", type=int, default=5,
                       help="Maximum positions")
    parser.add_argument("--min_probability", type=float, default=0.55,
                       help="Minimum probability threshold")
    
    args = parser.parse_args()
    
    config = {
        "initial_cash": args.initial_cash,
        "max_positions": args.max_positions,
        "min_probability": args.min_probability
    }
    
    backtester = Backtester(config)
    backtester.run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
