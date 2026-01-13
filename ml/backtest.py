"""
使用训练好的模型进行回测
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
import tomllib

import numpy as np
import polars as pl
import torch

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data.dataset import get_trading_days, load_daily_data
from ml.features.technical import calc_all_features, FEATURE_COLS
from ml.models.lstm import create_model


# 数据路径
DATA_ROOT = Path("/data/backtest_1")
MODEL_DIR = Path(__file__).parent / "checkpoints"
RESULT_DIR = Path(__file__).parent / "results"
RESULT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Trade:
    symbol: str
    buy_date: str
    buy_price: float
    sell_date: str
    sell_price: float
    profit_pct: float
    predicted_prob: float


def load_model():
    """加载训练好的模型"""
    checkpoint_path = MODEL_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("No trained model found. Run train.py first.")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    config = checkpoint['config']
    
    input_size = len(FEATURE_COLS)
    model = create_model(config["model_type"], input_size,
                         hidden_size=config["hidden_size"],
                         num_layers=config["num_layers"],
                         dropout=config["dropout"])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    # 加载标准化参数
    X_mean = np.load(MODEL_DIR / "X_mean.npy")
    X_std = np.load(MODEL_DIR / "X_std.npy")
    
    return model, config, X_mean, X_std


def prepare_stock_features(symbol: str, end_date: str, lookback: int, 
                           X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray | None:
    """准备单只股票的特征数据"""
    # 获取足够的历史数据
    trading_days = get_trading_days("2025-01-01", end_date)
    
    # 需要额外的数据来计算技术指标
    needed_days = lookback + 60
    if len(trading_days) < needed_days:
        return None
    
    days_to_use = trading_days[-needed_days:]
    
    # 加载数据
    daily_data = []
    for date in days_to_use:
        daily = load_daily_data(date)
        if daily is not None:
            stock_data = daily.filter(pl.col("symbol") == symbol)
            if stock_data.height > 0:
                daily_data.append(stock_data)
    
    if len(daily_data) < lookback + 30:
        return None
    
    df = pl.concat(daily_data).sort("date")
    
    # 计算特征
    df = calc_all_features(df)
    df = df.drop_nulls()
    
    if df.height < lookback:
        return None
    
    # 取最后lookback天
    feature_data = df.tail(lookback).select(FEATURE_COLS).to_numpy()
    
    # 标准化
    feature_data = (feature_data - X_mean.squeeze()) / X_std.squeeze()
    
    return feature_data


def predict_stocks(model, symbols: list[str], date: str, lookback: int,
                   X_mean: np.ndarray, X_std: np.ndarray) -> dict[str, float]:
    """预测多只股票的涨跌概率"""
    predictions = {}
    
    for symbol in symbols:
        features = prepare_stock_features(symbol, date, lookback, X_mean, X_std)
        if features is None:
            continue
        
        # 转换为张量
        X = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(X)
            # 获取正类（上涨）的概率
            prob = torch.softmax(output, dim=1)[0, 1].item()
            predictions[symbol] = prob
    
    return predictions


def backtest(start_date: str, end_date: str, top_k: int = 10, 
             prob_threshold: float = 0.6) -> list[Trade]:
    """
    回测策略
    
    策略：
    1. 每天收盘前预测所有股票次日涨跌概率
    2. 买入概率最高的top_k只股票（概率需超过threshold）
    3. 持有1天后卖出
    
    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        top_k: 每天最多买入的股票数量
        prob_threshold: 买入的概率阈值
    
    Returns:
        交易记录列表
    """
    print("Loading model...")
    model, config, X_mean, X_std = load_model()
    lookback = config["lookback"]
    
    trading_days = get_trading_days(start_date, end_date)
    print(f"Backtesting from {start_date} to {end_date}, {len(trading_days)} trading days")
    
    all_trades = []
    
    for i, date in enumerate(trading_days[:-1]):  # 最后一天无法卖出
        next_date = trading_days[i + 1]
        
        # 获取当天的股票列表
        daily = load_daily_data(date)
        if daily is None:
            continue
        
        symbols = daily["symbol"].unique().to_list()
        
        # 预测
        predictions = predict_stocks(model, symbols, date, lookback, X_mean, X_std)
        
        if not predictions:
            continue
        
        # 按概率排序，选择top_k
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        candidates = [(s, p) for s, p in sorted_preds if p >= prob_threshold][:top_k]
        
        if not candidates:
            continue
        
        # 获取买入价（当天收盘价）和卖出价（次日收盘价）
        next_daily = load_daily_data(next_date)
        if next_daily is None:
            continue
        
        buy_prices = {row["symbol"]: row["close"] for row in daily.iter_rows(named=True)}
        sell_prices = {row["symbol"]: row["open"] for row in next_daily.iter_rows(named=True)}
        
        # 执行交易
        for symbol, prob in candidates:
            if symbol not in buy_prices or symbol not in sell_prices:
                continue
            
            buy_price = buy_prices[symbol]
            sell_price = sell_prices[symbol]
            profit_pct = (sell_price / buy_price - 1) * 100
            
            trade = Trade(
                symbol=symbol,
                buy_date=date,
                buy_price=buy_price,
                sell_date=next_date,
                sell_price=sell_price,
                profit_pct=profit_pct,
                predicted_prob=prob
            )
            all_trades.append(trade)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(trading_days)} days, {len(candidates)} trades today")
    
    return all_trades


def generate_report(trades: list[Trade]):
    """生成回测报告"""
    if not trades:
        print("No trades to report")
        return
    
    # 转换为DataFrame
    df = pl.DataFrame([
        {
            "symbol": t.symbol,
            "buy_date": t.buy_date,
            "buy_price": t.buy_price,
            "sell_date": t.sell_date,
            "sell_price": t.sell_price,
            "profit_pct": t.profit_pct,
            "predicted_prob": t.predicted_prob,
        }
        for t in trades
    ])
    
    # 保存交易记录
    df.write_csv(RESULT_DIR / "ml_trades.csv")
    
    # 统计
    total_trades = len(trades)
    win_trades = df.filter(pl.col("profit_pct") > 0).height
    win_rate = win_trades / total_trades * 100
    avg_profit = df["profit_pct"].mean()
    total_profit = df["profit_pct"].sum()
    max_profit = df["profit_pct"].max()
    min_profit = df["profit_pct"].min()
    
    # 按概率分组统计
    high_prob = df.filter(pl.col("predicted_prob") >= 0.7)
    mid_prob = df.filter((pl.col("predicted_prob") >= 0.6) & (pl.col("predicted_prob") < 0.7))
    low_prob = df.filter((pl.col("predicted_prob") >= 0.55) & (pl.col("predicted_prob") < 0.6))
    
    high_avg = high_prob["profit_pct"].mean() if high_prob.height > 0 else 0
    mid_avg = mid_prob["profit_pct"].mean() if mid_prob.height > 0 else 0
    low_avg = low_prob["profit_pct"].mean() if low_prob.height > 0 else 0

    report = f"""# 深度学习策略回测报告

## 策略概述
- 模型: LSTM分类器
- 预测目标: 次日涨跌
- 买入条件: 模型预测上涨概率最高的股票
- 卖出条件: 持有1天后卖出

## 回测结果

### 交易统计
| 指标 | 值 |
|------|------|
| 总交易次数 | {total_trades} |
| 盈利次数 | {win_trades} |
| 胜率 | {win_rate:.2f}% |
| 平均收益率 | {avg_profit:.4f}% |
| 累计收益率 | {total_profit:.4f}% |
| 最大单笔收益 | {max_profit:.4f}% |
| 最大单笔亏损 | {min_profit:.4f}% |

### 按预测概率分组
| 概率区间 | 交易数 | 平均收益 |
|----------|--------|----------|
| >= 70% | {high_prob.height} | {high_avg:.4f}% |
| 60-70% | {mid_prob.height} | {mid_avg:.4f}% |
| 55-60% | {low_prob.height} | {low_avg:.4f}% |

"""
    
    with open(RESULT_DIR / "ml_report.md", "w") as f:
        f.write(report)
    
    print("\n" + "=" * 60)
    print("回测报告摘要")
    print("=" * 60)
    print(f"总交易次数: {total_trades}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"平均收益率: {avg_profit:.4f}%")
    print(f"累计收益率: {total_profit:.4f}%")
    print("=" * 60)
    print(f"详细报告已保存到: {RESULT_DIR}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest ML strategy")
    parser.add_argument("--start_date", type=str, default="2025-11-01")
    parser.add_argument("--end_date", type=str, default="2025-12-31")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.6)
    
    args = parser.parse_args()
    
    trades = backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        top_k=args.top_k,
        prob_threshold=args.threshold
    )
    
    generate_report(trades)


if __name__ == "__main__":
    main()
