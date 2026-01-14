"""
Walk-Forward 滚动训练模块

实现正确的时间序列训练方式，避免前瞻偏差
"""

import sys
from pathlib import Path
from datetime import date as dt_date, timedelta
from dataclasses import dataclass
import json

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    FEATURE_DATA_DIR, MODEL_CHECKPOINT_DIR, BACKTEST_RESULT_DIR, DEVICE
)
from pipeline.shared.utils import setup_logger, timer
from pipeline.data_cleaning.features import FEATURE_COLS, STRATEGY_PARAMS
from pipeline.training.train import LSTMModel


logger = setup_logger("walk_forward", BACKTEST_RESULT_DIR / "walk_forward.log")


@dataclass
class WalkForwardConfig:
    """Walk-Forward 配置"""
    train_days: int = 120        # 训练窗口（天）
    val_days: int = 20           # 验证窗口（天）
    retrain_freq: int = 5        # 重训练频率（每N天）
    lookback: int = 60           # 特征回看天数
    min_samples: int = 1000      # 最小训练样本数
    
    # 模型参数
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    
    # 训练参数
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001


class WalkForwardTrainer:
    """滚动时间窗口训练器"""
    
    def __init__(self, config: WalkForwardConfig | None = None):
        self.config = config or WalkForwardConfig()
        self.device = DEVICE
        self.model = None
        self.X_mean = None
        self.X_std = None
        
        logger.info("="*60)
        logger.info("Walk-Forward Trainer initialized")
        logger.info(f"Train Window: {self.config.train_days} days")
        logger.info(f"Val Window: {self.config.val_days} days")
        logger.info(f"Retrain Frequency: every {self.config.retrain_freq} days")
        logger.info("="*60)
    
    def load_data(self) -> pl.DataFrame:
        """加载特征数据"""
        feature_file = FEATURE_DATA_DIR / "features_all.parquet"
        
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        
        df = pl.read_parquet(feature_file)
        logger.info(f"Loaded {len(df)} rows")
        
        return df.sort(["date", "symbol"])
    
    def prepare_sequences(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备序列数据，保持时间顺序"""
        lookback = self.config.lookback
        
        all_X = []
        all_y = []
        all_dates = []
        
        symbols = df.select("symbol").unique().to_series().to_list()
        
        for symbol in symbols:
            stock_df = df.filter(pl.col("symbol") == symbol).sort("date")
            
            if len(stock_df) < lookback + 1:
                continue
            
            features = stock_df.select(FEATURE_COLS).to_numpy()
            labels = stock_df["label"].to_numpy()
            dates = stock_df["date"].to_numpy()
            
            for i in range(lookback, len(features)):
                X = features[i-lookback:i]
                y = labels[i]
                d = dates[i]
                
                if not np.any(np.isnan(X)) and not np.isnan(y):
                    all_X.append(X)
                    all_y.append(y)
                    all_dates.append(d)
        
        X = np.array(all_X)
        y = np.array(all_y)
        dates = np.array(all_dates)
        
        # 按日期排序
        sort_idx = np.argsort(dates)
        X = X[sort_idx]
        y = y[sort_idx]
        dates = dates[sort_idx]
        
        return X, y, dates
    
    def create_model(self, input_size: int) -> nn.Module:
        """创建模型"""
        model = LSTMModel(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            num_classes=2
        )
        return model.to(self.device)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
        """训练模型（只用训练集数据）"""
        
        # 用训练集计算标准化参数
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.X_mean = np.mean(X_train_flat, axis=0)
        self.X_std = np.std(X_train_flat, axis=0)
        self.X_std[self.X_std == 0] = 1
        
        # 标准化
        X_train_norm = (X_train - self.X_mean) / self.X_std
        X_val_norm = (X_val - self.X_mean) / self.X_std
        
        # 转为 Tensor
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_norm).float(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_norm).float(),
            torch.from_numpy(y_val).long()
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        # 创建模型
        model = self.create_model(X_train.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # 训练
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = model(X_batch)
                    _, predicted = outputs.max(1)
                    total += y_batch.size(0)
                    correct += predicted.eq(y_batch).sum().item()
            
            val_acc = correct / total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_state)
        logger.info(f"Model trained: val_acc={best_val_acc:.4f}")
        
        return model
    
    def predict(self, model: nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """预测"""
        X_norm = (X - self.X_mean) / self.X_std
        X_tensor = torch.from_numpy(X_norm).float().to(self.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        return preds, probs
    
    @timer
    def run(self, start_date: str, end_date: str):
        """运行 Walk-Forward 训练和测试"""
        logger.info("="*60)
        logger.info(f"Walk-Forward Training: {start_date} to {end_date}")
        logger.info("="*60)
        
        # 加载数据
        df = self.load_data()
        
        # 准备序列（保持时间顺序）
        X, y, dates = self.prepare_sequences(df)
        logger.info(f"Prepared {len(X)} samples")
        
        # 获取唯一日期
        unique_dates = np.unique(dates)
        logger.info(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
        
        # 转换日期
        start_dt = np.datetime64(start_date)
        end_dt = np.datetime64(end_date)
        
        # 筛选测试期日期
        test_dates = unique_dates[(unique_dates >= start_dt) & (unique_dates <= end_dt)]
        logger.info(f"Test dates: {len(test_dates)}")
        
        results = []
        model = None
        last_train_date = None
        
        for i, test_date in enumerate(test_dates):
            # 判断是否需要重新训练
            need_retrain = (model is None) or (i % self.config.retrain_freq == 0)
            
            if need_retrain:
                # 计算训练窗口
                train_end_date = test_date - np.timedelta64(self.config.val_days, 'D')
                train_start_date = train_end_date - np.timedelta64(self.config.train_days, 'D')
                val_end_date = test_date
                
                # 划分数据（严格按时间）
                train_mask = (dates >= train_start_date) & (dates < train_end_date)
                val_mask = (dates >= train_end_date) & (dates < val_end_date)
                
                X_train, y_train = X[train_mask], y[train_mask]
                X_val, y_val = X[val_mask], y[val_mask]
                
                if len(X_train) < self.config.min_samples:
                    logger.warning(f"Not enough training samples at {test_date}")
                    continue
                
                # 验证时间顺序
                train_dates = dates[train_mask]
                val_dates = dates[val_mask]
                assert train_dates.max() < val_dates.min(), "时间泄露！训练集包含验证集日期"
                
                logger.info(f"[{test_date}] Retraining with {len(X_train)} train, {len(X_val)} val samples")
                logger.info(f"  Train: {train_dates.min()} to {train_dates.max()}")
                logger.info(f"  Val: {val_dates.min()} to {val_dates.max()}")
                
                # 训练模型
                model = self.train_model(X_train, y_train, X_val, y_val)
                last_train_date = str(train_dates.max())
            
            # 预测当天数据
            test_mask = dates == test_date
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            if len(X_test) == 0:
                continue
            
            preds, probs = self.predict(model, X_test)
            
            # 记录结果
            accuracy = np.mean(preds == y_test)
            results.append({
                "date": str(test_date),
                "samples": len(X_test),
                "accuracy": accuracy,
                "predictions": preds.tolist(),
                "probabilities": probs.tolist(),
                "actuals": y_test.tolist(),
                "model_train_end": last_train_date
            })
            
            if (i + 1) % 10 == 0:
                avg_acc = np.mean([r["accuracy"] for r in results[-10:]])
                logger.info(f"Processed {i+1}/{len(test_dates)} days, recent avg acc: {avg_acc:.4f}")
        
        # 汇总结果
        all_preds = []
        all_actuals = []
        for r in results:
            all_preds.extend(r["predictions"])
            all_actuals.extend(r["actuals"])
        
        overall_acc = np.mean(np.array(all_preds) == np.array(all_actuals))
        
        summary = {
            "period": {"start": start_date, "end": end_date},
            "config": {
                "train_days": self.config.train_days,
                "val_days": self.config.val_days,
                "retrain_freq": self.config.retrain_freq
            },
            "results": {
                "total_days": len(results),
                "total_predictions": len(all_preds),
                "overall_accuracy": overall_acc,
                "daily_results": results
            }
        }
        
        # 保存结果
        output_file = BACKTEST_RESULT_DIR / f"walk_forward_{start_date}_{end_date}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("="*60)
        logger.info("Walk-Forward Summary")
        logger.info("="*60)
        logger.info(f"Total Days: {len(results)}")
        logger.info(f"Total Predictions: {len(all_preds)}")
        logger.info(f"Overall Accuracy: {overall_acc:.4f}")
        logger.info(f"Results saved to: {output_file}")
        logger.info("="*60)
        
        return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Training")
    parser.add_argument("--start_date", default="2025-01-02")
    parser.add_argument("--end_date", default="2025-06-30")
    parser.add_argument("--train_days", type=int, default=120)
    parser.add_argument("--val_days", type=int, default=20)
    parser.add_argument("--retrain_freq", type=int, default=5)
    
    args = parser.parse_args()
    
    config = WalkForwardConfig(
        train_days=args.train_days,
        val_days=args.val_days,
        retrain_freq=args.retrain_freq
    )
    
    trainer = WalkForwardTrainer(config)
    trainer.run(args.start_date, args.end_date)


if __name__ == "__main__":
    main()
