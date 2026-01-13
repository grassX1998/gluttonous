"""
个股量比策略回测 V2

策略逻辑:
1. 筛选条件: 当天09:31的1分钟成交量 > 前一交易日全天成交量
2. 买入: 触发条件时以当前收盘价买入
3. 卖出: 次日开盘价卖出
4. 不限制股票数量，所有满足条件的股票都交易

数据来源: /data/backtest_1/{date}/ 预处理的按日期合并数据
"""

import os
import sys
import tomllib
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import polars as pl

# 数据路径
DATA_ROOT = Path("/data/backtest_1")
GM_ROOT = Path("/data/stock/gm")


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    buy_date: str
    buy_price: float
    buy_volume: int  # 当日09:31成交量
    prev_volume: int  # 前日全天成交量
    sell_date: str = ""
    sell_price: float = 0.0
    profit_pct: float = 0.0


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """获取交易日列表"""
    cfg_path = GM_ROOT / "cfg" / "trading_days.toml"
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    
    all_days = sorted(data["trading_days"])
    return [d for d in all_days if start_date <= d <= end_date]


def get_next_trading_day(date: str, trading_days: list[str]) -> str | None:
    """获取下一个交易日"""
    try:
        idx = trading_days.index(date)
        if idx < len(trading_days) - 1:
            return trading_days[idx + 1]
    except ValueError:
        pass
    return None


def load_day_data(date: str) -> tuple[pl.DataFrame | None, pl.DataFrame | None]:
    """加载某天的数据"""
    day_dir = DATA_ROOT / date
    if not day_dir.exists():
        return None, None
    
    mkline_path = day_dir / "mkline.parquet"
    dkline_path = day_dir / "dkline.parquet"
    
    mkline = None
    dkline = None
    
    if mkline_path.exists():
        mkline = pl.read_parquet(mkline_path)
    
    if dkline_path.exists():
        dkline = pl.read_parquet(dkline_path)
    
    return mkline, dkline


def find_signals(date: str, mkline: pl.DataFrame, dkline: pl.DataFrame) -> list[dict]:
    """
    找出当天的买入信号
    
    条件: 09:31的1分钟成交量 > 前日全天成交量
    """
    if dkline is None or dkline.height == 0:
        return []
    
    signals = []
    
    # 获取09:31的分钟K线数据
    target_time = "09:31:00"
    mkline_0931 = mkline.filter(
        pl.col("time").cast(pl.Utf8).str.slice(0, 8) == target_time
    )
    
    if mkline_0931.height == 0:
        return []
    
    # 构建前日成交量字典
    prev_volume_map = {}
    for row in dkline.iter_rows(named=True):
        prev_volume_map[row["symbol"]] = row["prev_volume"]
    
    # 检查每只股票
    for row in mkline_0931.iter_rows(named=True):
        symbol = row["symbol"]
        volume_0931 = row["volume"]
        
        if symbol not in prev_volume_map:
            continue
        
        prev_volume = prev_volume_map[symbol]
        
        # 量比条件: 09:31成交量 > 前日全天成交量
        if volume_0931 > prev_volume and prev_volume > 0:
            signals.append({
                "symbol": symbol,
                "buy_price": row["close"],
                "buy_volume": volume_0931,
                "prev_volume": prev_volume,
                "volume_ratio": volume_0931 / prev_volume,
            })
    
    return signals


def backtest(start_date: str, end_date: str) -> list[Trade]:
    """运行回测"""
    trading_days = get_trading_days(start_date, end_date)
    print(f"Trading days: {len(trading_days)}")
    
    all_trades: list[Trade] = []
    pending_sells: dict[str, Trade] = {}  # symbol -> Trade (待卖出)
    
    for i, date in enumerate(trading_days):
        mkline, dkline = load_day_data(date)
        if mkline is None:
            print(f"[{i+1}/{len(trading_days)}] {date}: No data")
            continue
        
        # 1. 处理待卖出的持仓（次日开盘卖出）
        if pending_sells:
            # 获取09:31的开盘数据作为卖出价（数据从09:31开始，09:31的open即为当天开盘价）
            mkline_0931 = mkline.filter(
                pl.col("time").cast(pl.Utf8).str.slice(0, 8) == "09:31:00"
            )
            
            open_prices = {}
            for row in mkline_0931.iter_rows(named=True):
                open_prices[row["symbol"]] = row["open"]
            
            sold_symbols = []
            for symbol, trade in pending_sells.items():
                if symbol in open_prices:
                    trade.sell_date = date
                    trade.sell_price = open_prices[symbol]
                    trade.profit_pct = (trade.sell_price - trade.buy_price) / trade.buy_price * 100
                    all_trades.append(trade)
                    sold_symbols.append(symbol)
            
            for symbol in sold_symbols:
                del pending_sells[symbol]
        
        # 2. 找出当天的买入信号
        signals = find_signals(date, mkline, dkline)
        
        # 3. 买入
        for sig in signals:
            symbol = sig["symbol"]
            # 如果已经有持仓，跳过
            if symbol in pending_sells:
                continue
            
            trade = Trade(
                symbol=symbol,
                buy_date=date,
                buy_price=sig["buy_price"],
                buy_volume=sig["buy_volume"],
                prev_volume=sig["prev_volume"],
            )
            pending_sells[symbol] = trade
        
        print(f"[{i+1}/{len(trading_days)}] {date}: {len(signals)} signals, {len(pending_sells)} pending")
    
    # 处理最后一天的持仓（按买入价平仓）
    for symbol, trade in pending_sells.items():
        trade.sell_date = trade.buy_date
        trade.sell_price = trade.buy_price
        trade.profit_pct = 0.0
        all_trades.append(trade)
    
    return all_trades


def generate_report(trades: list[Trade], output_dir: Path):
    """生成回测报告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not trades:
        print("No trades to report")
        return
    
    # 转换为DataFrame
    df = pl.DataFrame([
        {
            "symbol": t.symbol,
            "buy_date": t.buy_date,
            "buy_price": t.buy_price,
            "buy_volume": t.buy_volume,
            "prev_volume": t.prev_volume,
            "volume_ratio": t.buy_volume / t.prev_volume if t.prev_volume > 0 else 0,
            "sell_date": t.sell_date,
            "sell_price": t.sell_price,
            "profit_pct": t.profit_pct,
        }
        for t in trades
    ])
    
    # 保存交易记录
    df.write_parquet(output_dir / "trades.parquet")
    df.write_csv(output_dir / "trades.csv")
    
    # 计算统计指标
    total_trades = len(trades)
    win_trades = df.filter(pl.col("profit_pct") > 0).height
    loss_trades = df.filter(pl.col("profit_pct") < 0).height
    even_trades = df.filter(pl.col("profit_pct") == 0).height
    
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    avg_profit = df["profit_pct"].mean()
    max_profit = df["profit_pct"].max()
    min_profit = df["profit_pct"].min()
    total_profit = df["profit_pct"].sum()
    
    # 按日期统计
    daily_stats = df.group_by("buy_date").agg([
        pl.count().alias("trades"),
        pl.col("profit_pct").mean().alias("avg_profit"),
        pl.col("profit_pct").sum().alias("total_profit"),
    ]).sort("buy_date")
    
    daily_stats.write_csv(output_dir / "daily_stats.csv")
    
    # 生成报告
    report = f"""# 个股量比策略回测报告

## 策略概述
- 买入条件: 09:31的1分钟成交量 > 前一交易日全天成交量
- 买入价格: 09:31分钟K线收盘价
- 卖出条件: 次日开盘
- 卖出价格: 次日09:30分钟K线开盘价

## 回测结果

### 交易统计
| 指标 | 值 |
|------|------|
| 总交易次数 | {total_trades} |
| 盈利次数 | {win_trades} |
| 亏损次数 | {loss_trades} |
| 平局次数 | {even_trades} |
| 胜率 | {win_rate:.2f}% |

### 收益统计
| 指标 | 值 |
|------|------|
| 平均收益率 | {avg_profit:.4f}% |
| 最大单笔收益 | {max_profit:.4f}% |
| 最大单笔亏损 | {min_profit:.4f}% |
| 累计收益率(简单加总) | {total_profit:.4f}% |

### 每日交易统计
日期 | 交易数 | 平均收益 | 合计收益
-----|--------|----------|----------
"""
    
    for row in daily_stats.iter_rows(named=True):
        report += f"{row['buy_date']} | {row['trades']} | {row['avg_profit']:.4f}% | {row['total_profit']:.4f}%\n"
    
    # 保存报告
    with open(output_dir / "report.md", "w") as f:
        f.write(report)
    
    # 打印摘要
    print("\n" + "="*60)
    print("回测报告摘要")
    print("="*60)
    print(f"总交易次数: {total_trades}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_profit:.4f}%")
    print(f"最大收益: {max_profit:.4f}%")
    print(f"最大亏损: {min_profit:.4f}%")
    print(f"累计收益率: {total_profit:.4f}%")
    print("="*60)
    print(f"详细报告已保存到: {output_dir}")


def main():
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    output_dir = Path("/home/grasszhang/workspace/projects/gluttonous/backtest/stock-2025/results")
    
    print(f"Running backtest from {start_date} to {end_date}")
    print("-" * 60)
    
    trades = backtest(start_date, end_date)
    
    print("-" * 60)
    print(f"Total trades: {len(trades)}")
    
    generate_report(trades, output_dir)


if __name__ == "__main__":
    main()
