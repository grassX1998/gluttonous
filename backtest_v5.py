"""
简化回测 v5 - 正确实现5日持有策略

标签定义：label = 1 if close[T+5] > close[T] else 0
策略：每日选股 -> 持有5天 -> 计算收益

优化：使用采样减少训练时间
"""
import sys
import time
import signal
from pathlib import Path
import json
import gc

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

signal.signal(signal.SIGINT, signal.SIG_IGN)

PIPELINE_DATA_ROOT = Path(".pipeline_data")
FEATURE_DATA_DIR = PIPELINE_DATA_ROOT / "features_monthly"
DAILY_DATA_DIR = PIPELINE_DATA_ROOT / "daily"
RESULT_DIR = PIPELINE_DATA_ROOT / "backtest_results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_60_ratio",
    "volatility_5", "volatility_10", "volatility_20",
    "rsi_14",
    "macd_dif", "macd_dea", "macd_hist",
    "bb_position",
    "volume_ratio_5", "volume_ratio_10", "volume_ratio_20",
    "turnover_ma_ratio",
    "open_gap",
    "price_vs_max_25",
    "price_breakout",
    "stop_loss_signal",
    "ma_10_vs_20",
    "price_vs_ma_10",
    "ret_2d",
    "price_vs_2d_max",
    "is_limit_up",
    "is_limit_down",
]

# 配置
class Config:
    TOP_N = 10             # 保持10只持仓
    PROB_THRESHOLD = 0.60  # 最佳阈值（Config 1表现最好）
    HOLDING_DAYS = 5
    COMMISSION = 0.001
    SLIPPAGE = 0.001
    
    TRAIN_MONTHS = 6
    VAL_MONTHS = 1
    
    # 采样比例（加速训练）
    SAMPLE_RATIO = 0.5  # 使用50%训练数据提高稳定性
    RANDOM_SEED = 42     # 固定随机种子确保可重现
    
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 1024
    EPOCHS = 10
    EARLY_STOP_PATIENCE = 3


class LSTMModel(nn.Module):
    def __init__(self, input_size, config: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} - {msg}", flush=True)


def load_months_data(months: list, data_type: str = "feature"):
    """加载多月数据"""
    data_dir = FEATURE_DATA_DIR if data_type == "feature" else DAILY_DATA_DIR
    dfs = []
    for m in months:
        path = data_dir / f"{m}.parquet"
        if path.exists():
            dfs.append(pl.read_parquet(path))
    return pl.concat(dfs).sort(["symbol", "date"]) if dfs else None


def prepare_features(df):
    """准备特征"""
    available_cols = [col for col in FEATURE_COLS if col in df.columns]
    df_clean = df.drop_nulls(subset=available_cols)
    
    X = df_clean.select(available_cols).to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    dates = df_clean["date"].to_list()
    symbols = df_clean["symbol"].to_list()
    labels = df_clean["label"].to_numpy().astype(int) if "label" in df_clean.columns else None
    
    return X, dates, symbols, labels


def train_model(X_train, y_train, X_val, y_val, config: Config):
    """训练模型（带采样）"""
    # 设置随机种子确保可重现
    if hasattr(config, 'RANDOM_SEED'):
        np.random.seed(config.RANDOM_SEED)
        torch.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.RANDOM_SEED)
    
    # 采样减少数据量
    n_train = len(X_train)
    sample_size = int(n_train * config.SAMPLE_RATIO)
    indices = np.random.choice(n_train, sample_size, replace=False)
    X_train_sample = X_train[indices]
    y_train_sample = y_train[indices]
    
    # 标准化
    X_mean = np.mean(X_train_sample, axis=0)
    X_std = np.std(X_train_sample, axis=0)
    X_std[X_std == 0] = 1
    
    X_train_norm = (X_train_sample - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    
    X_train_seq = X_train_norm[:, np.newaxis, :]
    X_val_seq = X_val_norm[:, np.newaxis, :]
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_seq).float(),
        torch.from_numpy(y_train_sample).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_seq).float(),
        torch.from_numpy(y_val).long()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    model = LSTMModel(input_size=X_train.shape[1], config=config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_state = None
    patience_counter = 0
    
    for epoch in range(config.EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                outputs = model(X_batch)
                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
        
        val_acc = correct / total if total > 0 else 0
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, X_mean, X_std, best_val_acc


def predict(model, X, X_mean, X_std):
    """批量预测"""
    X_norm = (X - X_mean) / X_std
    X_seq = X_norm[:, np.newaxis, :]
    X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)
    
    model.eval()
    batch_size = 10000
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i+batch_size]
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    
    return np.array(all_probs)


def main():
    config = Config()
    
    log("=" * 60)
    log("Simplified Backtest v5 (Correct 5-day Holding)")
    log(f"Device: {DEVICE}")
    log(f"Top N: {config.TOP_N}, Prob: {config.PROB_THRESHOLD}")
    log(f"Holding Days: {config.HOLDING_DAYS}")
    log(f"Sample Ratio: {config.SAMPLE_RATIO}")
    log("=" * 60)
    
    # 获取月份
    all_months = sorted([p.stem for p in FEATURE_DATA_DIR.glob("*.parquet")])
    log(f"Available months: {len(all_months)}")
    
    # 加载日线数据
    log("Loading daily data...")
    all_daily = load_months_data(all_months, "daily")
    log(f"Daily data: {len(all_daily)} rows")
    
    # 日期索引和价格表
    all_dates = sorted(all_daily.select("date").unique().to_series().to_list())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    price_map = {}
    for row in all_daily.select(["date", "symbol", "close"]).iter_rows():
        price_map[(row[0], row[1])] = row[2]
    
    log(f"Dates: {len(all_dates)}, Prices: {len(price_map)}")
    
    # Walk-Forward 回测
    min_months = config.TRAIN_MONTHS + config.VAL_MONTHS + 1
    all_predictions = []
    
    for test_idx in range(min_months - 1, len(all_months)):
        test_month = all_months[test_idx]
        
        # 训练/验证月份
        val_end_idx = test_idx
        val_start_idx = val_end_idx - config.VAL_MONTHS
        train_end_idx = val_start_idx
        train_start_idx = max(0, train_end_idx - config.TRAIN_MONTHS)
        
        train_months = all_months[train_start_idx:train_end_idx]
        val_months = all_months[val_start_idx:val_end_idx]
        
        log(f"Test {test_month}: Train {train_months[0]}~{train_months[-1]}, Val {val_months[0]}")
        
        # 加载数据
        train_df = load_months_data(train_months, "feature")
        val_df = load_months_data(val_months, "feature")
        test_df = load_months_data([test_month], "feature")
        
        if train_df is None or val_df is None or test_df is None:
            log(f"  Skip - missing data")
            continue
        
        # 准备特征
        X_train, _, _, y_train = prepare_features(train_df)
        X_val, _, _, y_val = prepare_features(val_df)
        X_test, test_dates, test_symbols, _ = prepare_features(test_df)
        
        log(f"  Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
        
        # 训练
        model, X_mean, X_std, val_acc = train_model(X_train, y_train, X_val, y_val, config)
        log(f"  Trained: val_acc={val_acc:.4f}")
        
        # 预测
        probs = predict(model, X_test, X_mean, X_std)
        
        for i in range(len(probs)):
            all_predictions.append({
                "date": str(test_dates[i]),
                "symbol": test_symbols[i],
                "prob": float(probs[i])
            })
        
        # 清理
        del train_df, val_df, test_df, model
        gc.collect()
        torch.cuda.empty_cache()
    
    log(f"Total predictions: {len(all_predictions)}")
    
    # 模拟交易 - 5日持有
    trades = []
    pred_df = pl.DataFrame(all_predictions)
    unique_dates = sorted(pred_df["date"].unique().to_list())
    
    log(f"Simulating trades: {unique_dates[0]} to {unique_dates[-1]}")
    
    for date in unique_dates:
        day_pred = pred_df.filter(pl.col("date") == date)
        
        # 选股
        candidates = day_pred.filter(
            pl.col("prob") >= config.PROB_THRESHOLD
        ).sort("prob", descending=True).head(config.TOP_N)
        
        if candidates.is_empty():
            continue
        
        for row in candidates.iter_rows(named=True):
            symbol = row["symbol"]
            entry_date = row["date"]
            prob = row["prob"]
            
            # 买入价格（收盘价）
            entry_price = price_map.get((entry_date, symbol))
            if entry_price is None:
                continue
            
            # T+5 卖出
            if entry_date not in date_to_idx:
                continue
            
            entry_idx = date_to_idx[entry_date]
            exit_idx = entry_idx + config.HOLDING_DAYS
            
            if exit_idx >= len(all_dates):
                continue
            
            exit_date = all_dates[exit_idx]
            exit_price = price_map.get((exit_date, symbol))
            
            if exit_price is None:
                continue
            
            # 5日持有收益
            gross_return = exit_price / entry_price - 1
            net_return = gross_return - (config.COMMISSION + config.SLIPPAGE) * 2
            
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "symbol": symbol,
                "prob": prob,
                "gross_return": gross_return,
                "net_return": net_return
            })
    
    log(f"Total trades: {len(trades)}")
    
    if not trades:
        log("No trades!")
        return
    
    # 按入场日期汇总
    trades_df = pl.DataFrame(trades)
    daily_group = trades_df.group_by("entry_date").agg([
        pl.col("net_return").mean().alias("avg_return"),
        pl.col("net_return").count().alias("n_trades"),
    ]).sort("entry_date")
    
    returns = daily_group["avg_return"].to_numpy()
    
    # 统计
    cum_returns = np.cumprod(1 + returns) - 1
    total_return = cum_returns[-1]
    
    n_days = len(returns)
    annual_return = (1 + total_return) ** (250 / n_days) - 1 if n_days > 0 else 0
    
    daily_std = np.std(returns)
    sharpe = np.sqrt(250) * np.mean(returns) / daily_std if daily_std > 0 else 0
    
    cum_values = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_values)
    drawdown = (peak - cum_values) / peak
    max_drawdown = np.max(drawdown)
    
    daily_win_rate = np.mean(returns > 0)
    trade_win_rate = np.mean([t["net_return"] > 0 for t in trades])
    
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"Period: {unique_dates[0]} to {unique_dates[-1]}")
    log(f"Trading Days: {n_days}")
    log(f"Total Trades: {len(trades)}")
    log(f"Total Return: {total_return*100:+.2f}%")
    log(f"Annual Return: {annual_return*100:+.2f}%")
    log(f"Sharpe Ratio: {sharpe:.3f}")
    log(f"Max Drawdown: {max_drawdown*100:.2f}%")
    log(f"Daily Win Rate: {daily_win_rate*100:.2f}%")
    log(f"Trade Win Rate: {trade_win_rate*100:.2f}%")
    log("=" * 60)
    
    log("\nComparison with v0.2 Benchmark:")
    log("v0.2 (CSI500+1000): Return=+38.74%, Sharpe=1.867, MaxDD=9.53%")
    log(f"v3 (Full Market):  Return={total_return*100:+.2f}%, Sharpe={sharpe:.3f}, MaxDD={max_drawdown*100:.2f}%")
    log("=" * 60)
    
    # 保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 每日详细数据（用于绘图）
    daily_data = []
    entry_dates = daily_group["entry_date"].to_list()
    for i, date in enumerate(entry_dates):
        daily_data.append({
            "date": date,
            "return": float(returns[i]),
            "cum_return": float(cum_returns[i]),
            "cum_value": float(cum_values[i]),
            "drawdown": float(drawdown[i]),
            "n_trades": int(daily_group["n_trades"][i]),
        })
    
    result = {
        "config": vars(config),
        "results": {
            "total_return": float(total_return),
            "annual_return": float(annual_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "daily_win_rate": float(daily_win_rate),
            "trade_win_rate": float(trade_win_rate),
            "n_trades": len(trades),
            "n_days": n_days,
        },
        "daily_data": daily_data,
    }
    
    output_file = RESULT_DIR / f"backtest_v5_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    log(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
