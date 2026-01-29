"""
v0.3 复现策略执行器

复现归档 v0.3 的完整逻辑：
- 月度滚动训练（6月训练+1月验证）
- 50% 采样
- 固定 5 天持有
- 概率阈值 0.60

用于验证 v0.3 结果是否可复现。

注意：当前框架使用按股票组织的特征数据（features目录），
      本执行器会自动加载并按月分组。
"""

import sys
import gc
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.config import (
    V03ReproConfig,
    DEVICE,
    MODEL_CONFIG,
)
from src.lstm.models import SimpleLSTMModel
from pipeline.data_cleaning.features import FEATURE_COLS
from pipeline.shared.config import FEATURE_DATA_DIR


class V03ReproExecutor:
    """v0.3 复现执行器 - 独立实现，不继承 BaseStrategyExecutor"""

    def __init__(self, config: V03ReproConfig = None):
        self.config = config or V03ReproConfig()
        self.model = None
        self.scaler_params = None
        self._all_data_cache = None  # 缓存所有数据

    def _load_all_data(self) -> pl.DataFrame:
        """加载所有特征数据（按股票组织）"""
        if self._all_data_cache is not None:
            return self._all_data_cache

        print("Loading all feature data...")
        feature_files = list(FEATURE_DATA_DIR.glob("*.parquet"))

        if not feature_files:
            print(f"ERROR: No feature files found in {FEATURE_DATA_DIR}")
            return None

        dfs = []
        for f in feature_files:
            try:
                df = pl.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                continue

        if not dfs:
            return None

        all_data = pl.concat(dfs)
        # 确保日期是字符串格式
        all_data = all_data.with_columns(pl.col("date").cast(pl.Utf8).alias("date_str"))

        self._all_data_cache = all_data
        print(f"Loaded {len(all_data)} rows from {len(feature_files)} files")
        return all_data

    def _get_available_months(self) -> List[str]:
        """获取所有可用月份"""
        all_data = self._load_all_data()
        if all_data is None:
            return []

        # 提取年月
        months = all_data.select(
            pl.col("date_str").str.slice(0, 7).alias("month")
        ).unique().sort("month")["month"].to_list()

        return months

    def _load_months_data(self, months: List[str]) -> pl.DataFrame:
        """加载指定月份的数据"""
        all_data = self._load_all_data()
        if all_data is None:
            return None

        # 过滤月份
        month_filter = pl.col("date_str").str.slice(0, 7).is_in(months)
        filtered = all_data.filter(month_filter)

        return filtered.sort(["symbol", "date_str"]) if filtered.height > 0 else None

    def _prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, List, List, np.ndarray]:
        """准备特征"""
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols)

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 使用 date_str 列（字符串格式）
        date_col = "date_str" if "date_str" in df_clean.columns else "date"
        dates = df_clean[date_col].to_list()
        # 确保日期是字符串
        dates = [str(d) for d in dates]

        symbols = df_clean["symbol"].to_list()
        labels = df_clean["label"].to_numpy().astype(int) if "label" in df_clean.columns else None

        return X, dates, symbols, labels

    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> float:
        """训练模型（带采样）"""
        config = self.config

        # 设置随机种子
        np.random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)

        # 采样减少数据量
        n_train = len(X_train)
        sample_size = int(n_train * config.sample_ratio)
        indices = np.random.choice(n_train, sample_size, replace=False)
        X_train_sample = X_train[indices]
        y_train_sample = y_train[indices]

        # 标准化
        X_mean = np.mean(X_train_sample, axis=0)
        X_std = np.std(X_train_sample, axis=0)
        X_std[X_std == 0] = 1

        X_train_norm = (X_train_sample - X_mean) / X_std
        X_val_norm = (X_val - X_mean) / X_std

        # 保存标准化参数
        self.scaler_params = {'mean': X_mean, 'std': X_std}

        # 转换为序列格式
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

        train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'])

        # 创建模型
        self.model = SimpleLSTMModel(
            input_size=X_train.shape[1],
            hidden_size=MODEL_CONFIG['hidden_size'],
            num_layers=MODEL_CONFIG['num_layers'],
            dropout=MODEL_CONFIG['dropout']
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=MODEL_CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        best_state = None
        patience_counter = 0

        for epoch in range(MODEL_CONFIG['epochs']):
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # 验证
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    outputs = self.model(X_batch)
                    _, predicted = outputs.max(1)
                    total += y_batch.size(0)
                    correct += predicted.eq(y_batch).sum().item()

            val_acc = correct / total if total > 0 else 0

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= MODEL_CONFIG['early_stop_patience']:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return best_val_acc

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        X_norm = (X - self.scaler_params['mean']) / self.scaler_params['std']
        X_seq = X_norm[:, np.newaxis, :]
        X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)

        self.model.eval()
        batch_size = 10000
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs)

    def run(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        运行 v0.3 复现回测

        Args:
            start_date: 开始日期（可选，默认使用所有数据）
            end_date: 结束日期（可选）

        Returns:
            回测结果
        """
        config = self.config
        print("=" * 60)
        print("V0.3 Reproduction Backtest")
        print(f"Device: {DEVICE}")
        print(f"Top N: {config.top_n}, Prob: {config.prob_threshold}")
        print(f"Holding Days: {config.holding_days}")
        print(f"Sample Ratio: {config.sample_ratio}")
        print(f"Train Months: {config.train_months}, Val Months: {config.val_months}")
        print("=" * 60)

        # 获取所有可用月份（从按股票组织的数据中提取）
        all_months = self._get_available_months()
        print(f"Available months: {len(all_months)}")

        if not all_months:
            print("ERROR: No feature data found!")
            return {'error': 'No feature data'}

        # 加载所有数据用于价格查询
        print("Loading all data...")
        all_data = self._load_all_data()
        if all_data is None:
            print("ERROR: No data found!")
            return {'error': 'No data'}

        print(f"All data: {len(all_data)} rows")

        # 构建日期索引和价格表（使用 date_str 列）
        all_dates = sorted(all_data.select("date_str").unique().to_series().to_list())
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        price_map = {}
        for row in all_data.select(["date_str", "symbol", "close"]).iter_rows():
            price_map[(row[0], row[1])] = row[2]

        print(f"Dates: {len(all_dates)}, Prices: {len(price_map)}")

        # Walk-Forward 月度回测
        min_months = config.train_months + config.val_months + 1
        all_predictions = []

        for test_idx in range(min_months - 1, len(all_months)):
            test_month = all_months[test_idx]

            # 训练/验证月份
            val_end_idx = test_idx
            val_start_idx = val_end_idx - config.val_months
            train_end_idx = val_start_idx
            train_start_idx = max(0, train_end_idx - config.train_months)

            train_months = all_months[train_start_idx:train_end_idx]
            val_months_list = all_months[val_start_idx:val_end_idx]

            print(f"Test {test_month}: Train {train_months[0]}~{train_months[-1]}, Val {val_months_list[0]}")

            # 加载数据
            train_df = self._load_months_data(train_months)
            val_df = self._load_months_data(val_months_list)
            test_df = self._load_months_data([test_month])

            if train_df is None or val_df is None or test_df is None:
                print(f"  Skip - missing data")
                continue

            # 准备特征
            X_train, _, _, y_train = self._prepare_features(train_df)
            X_val, _, _, y_val = self._prepare_features(val_df)
            X_test, test_dates, test_symbols, _ = self._prepare_features(test_df)

            print(f"  Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

            # 训练
            val_acc = self._train_model(X_train, y_train, X_val, y_val)
            print(f"  Trained: val_acc={val_acc:.4f}")

            # 预测
            probs = self._predict(X_test)

            for i in range(len(probs)):
                all_predictions.append({
                    "date": str(test_dates[i]),
                    "symbol": test_symbols[i],
                    "prob": float(probs[i])
                })

            # 清理内存
            del train_df, val_df, test_df
            gc.collect()
            torch.cuda.empty_cache()

        print(f"Total predictions: {len(all_predictions)}")

        if not all_predictions:
            return {'error': 'No predictions generated'}

        # ===== 模拟固定持有期交易 =====
        trades = []
        pred_df = pl.DataFrame(all_predictions)
        unique_dates = sorted(pred_df["date"].unique().to_list())

        print(f"Simulating trades: {unique_dates[0]} to {unique_dates[-1]}")

        for date in unique_dates:
            day_pred = pred_df.filter(pl.col("date") == date)

            # 选股
            candidates = day_pred.filter(
                pl.col("prob") >= config.prob_threshold
            ).sort("prob", descending=True).head(config.top_n)

            if candidates.is_empty():
                continue

            for row in candidates.iter_rows(named=True):
                symbol = row["symbol"]
                entry_date = row["date"]
                prob = row["prob"]

                # 买入价格
                entry_price = price_map.get((entry_date, symbol))
                if entry_price is None:
                    continue

                # T+5 卖出
                if entry_date not in date_to_idx:
                    continue

                entry_idx = date_to_idx[entry_date]
                exit_idx = entry_idx + config.holding_days

                if exit_idx >= len(all_dates):
                    continue

                exit_date = all_dates[exit_idx]
                exit_price = price_map.get((exit_date, symbol))

                if exit_price is None:
                    continue

                # 计算收益
                gross_return = exit_price / entry_price - 1
                net_return = gross_return - (config.commission + config.slippage) * 2

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "symbol": symbol,
                    "prob": prob,
                    "gross_return": gross_return,
                    "net_return": net_return
                })

        print(f"Total trades: {len(trades)}")

        if not trades:
            return {'error': 'No trades'}

        # ===== 计算指标 =====
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

        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Period: {unique_dates[0]} to {unique_dates[-1]}")
        print(f"Trading Days: {n_days}")
        print(f"Total Trades: {len(trades)}")
        print(f"Total Return: {total_return*100:+.2f}%")
        print(f"Annual Return: {annual_return*100:+.2f}%")
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"Daily Win Rate: {daily_win_rate*100:.2f}%")
        print(f"Trade Win Rate: {trade_win_rate*100:.2f}%")
        print("=" * 60)

        print("\nComparison with v0.3 Archive:")
        print("v0.3 Archive: Return=+74.84%, Sharpe=1.566, MaxDD=47.04%")
        print(f"v0.3 Repro:   Return={total_return*100:+.2f}%, Sharpe={sharpe:.3f}, MaxDD={max_drawdown*100:.2f}%")
        print("=" * 60)

        # 构建每日数据（用于绘图）
        entry_dates = daily_group["entry_date"].to_list()
        n_trades_list = daily_group["n_trades"].to_list()
        daily_data = []
        for i, date in enumerate(entry_dates):
            daily_data.append({
                'date': date,
                'daily_return': float(returns[i]),
                'cum_return': float(cum_returns[i]),
                'cum_value': float(cum_values[i]),
                'drawdown': float(drawdown[i]),
                'n_positions': int(n_trades_list[i]),
            })

        return {
            'strategy': 'v03_repro',
            'config': {
                'top_n': config.top_n,
                'prob_threshold': config.prob_threshold,
                'holding_days': config.holding_days,
                'train_months': config.train_months,
                'val_months': config.val_months,
                'sample_ratio': config.sample_ratio,
            },
            'results': {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'daily_win_rate': float(daily_win_rate),
                'trade_win_rate': float(trade_win_rate),
                'n_trades': len(trades),
                'n_days': n_days,
            },
            'daily_data': daily_data,
            'predictions': all_predictions,
            'trades': trades,
        }
