"""
策略执行器基类

所有训练策略执行器都应继承此基类，实现以下方法：
- prepare_data: 准备训练数据
- should_retrain: 判断是否需要重训练
- train_model: 训练模型
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入配置和模型
from src.lstm.config import (
    FEATURE_DATA_MONTHLY_DIR,
    DAILY_DATA_DIR,
    MODEL_CHECKPOINT_DIR,
    DEVICE,
    MODEL_CONFIG,
    TrainStrategyConfig
)
from src.lstm.models import SimpleLSTMModel
from pipeline.data_cleaning.features import FEATURE_COLS


class BaseStrategyExecutor(ABC):
    """策略执行器基类"""

    def __init__(self, config: TrainStrategyConfig):
        """
        初始化执行器

        Args:
            config: 策略配置对象
        """
        self.config = config
        self.model = None
        self.scaler_params = None  # {'mean': ndarray, 'std': ndarray}
        self.performance_history = []  # 性能历史记录

        # 模型配置（可以被子类覆盖）
        self.model_config = MODEL_CONFIG.copy()

    @abstractmethod
    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备训练数据

        Args:
            current_date: 当前日期（字符串格式 'YYYY-MM-DD'）

        Returns:
            X_train, y_train, X_val, y_val
        """
        pass

    @abstractmethod
    def should_retrain(self, current_date: str) -> bool:
        """
        判断是否需要重新训练

        Args:
            current_date: 当前日期

        Returns:
            是否需要重训练
        """
        pass

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """
        训练模型（通用实现）

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            验证集准确率
        """
        # 标准化
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_std[X_std == 0] = 1

        X_train_norm = (X_train - X_mean) / X_std
        X_val_norm = (X_val - X_mean) / X_std

        # 保存标准化参数
        self.scaler_params = {'mean': X_mean, 'std': X_std}

        # 转换为序列格式 (batch, seq_len=1, features)
        X_train_seq = X_train_norm[:, np.newaxis, :]
        X_val_seq = X_val_norm[:, np.newaxis, :]

        # 创建数据集
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_seq).float(),
            torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val_seq).float(),
            torch.from_numpy(y_val).long()
        )

        train_loader = DataLoader(train_dataset,
                                 batch_size=self.model_config['batch_size'],
                                 shuffle=True)
        val_loader = DataLoader(val_dataset,
                               batch_size=self.model_config['batch_size'])

        # 创建模型
        input_size = X_train.shape[1]
        self.model = SimpleLSTMModel(
            input_size=input_size,
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(self.model.parameters(),
                                     lr=self.model_config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        best_state = None
        patience_counter = 0

        # 训练循环
        for epoch in range(self.model_config['epochs']):
            # 训练阶段
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # 验证阶段
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

            # 早停
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_config['early_stop_patience']:
                    break

        # 加载最佳模型
        if best_state:
            self.model.load_state_dict(best_state)

        return best_val_acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测（返回正类概率）

        Args:
            X: 特征矩阵

        Returns:
            正类概率数组
        """
        if self.model is None or self.scaler_params is None:
            raise RuntimeError("Model not trained yet")

        # 标准化
        X_norm = (X - self.scaler_params['mean']) / self.scaler_params['std']
        X_seq = X_norm[:, np.newaxis, :]
        X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)

        # 预测
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

    def save_model(self, path: Path):
        """保存模型和标准化参数"""
        if self.model is not None:
            checkpoint = {
                'model_state': self.model.state_dict(),
                'scaler_params': self.scaler_params,
                'config': vars(self.config),
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, path)

    def load_model(self, path: Path):
        """加载模型和标准化参数"""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        # 重建模型
        input_size = len(FEATURE_COLS)
        self.model = SimpleLSTMModel(
            input_size=input_size,
            hidden_size=self.model_config['hidden_size'],
            num_layers=self.model_config['num_layers'],
            dropout=self.model_config['dropout']
        ).to(DEVICE)

        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler_params = checkpoint['scaler_params']

    def _load_month_data(self, month: str) -> pl.DataFrame:
        """加载单月特征数据"""
        path = FEATURE_DATA_MONTHLY_DIR / f"{month}.parquet"
        if path.exists():
            return pl.read_parquet(path)
        return None

    def _load_date_range_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        加载日期范围内的数据

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame
        """
        # 获取需要加载的月份
        start_ym = start_date[:7]  # 'YYYY-MM'
        end_ym = end_date[:7]

        # 获取所有可用月份
        available_months = sorted([p.stem for p in FEATURE_DATA_MONTHLY_DIR.glob("*.parquet")])

        # 过滤需要的月份
        needed_months = [m for m in available_months if start_ym <= m <= end_ym]

        # 加载数据
        dfs = []
        for month in needed_months:
            df = self._load_month_data(month)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return None

        # 合并并过滤日期
        df = pl.concat(dfs).filter(
            (pl.col("date") >= start_date) & (pl.col("date") <= end_date)
        )

        return df

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取日期范围内的所有交易日"""
        # 从已有数据中提取交易日
        all_months = sorted([p.stem for p in FEATURE_DATA_MONTHLY_DIR.glob("*.parquet")])

        dates_set = set()
        for month in all_months:
            df = self._load_month_data(month)
            if df is not None:
                dates = df.select("date").unique().to_series().to_list()
                dates_set.update(dates)

        # 过滤并排序
        trading_dates = sorted([d for d in dates_set if start_date <= d <= end_date])
        return trading_dates

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        执行回测（核心流程）

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            回测结果字典
        """
        results = {
            'strategy': self.config.strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'predictions': [],  # List of {date, symbol, prob}
            'retrain_dates': [],  # 记录重训练日期
            'performance_history': []  # 记录每次训练的性能
        }

        # 获取所有交易日
        all_dates = self._get_trading_dates(start_date, end_date)
        print(f"[{self.config.strategy_name}] Trading dates: {len(all_dates)}")

        # 逐日预测
        for i, date in enumerate(all_dates):
            # 判断是否需要重训练
            if self.should_retrain(date):
                print(f"[{self.config.strategy_name}] Training on {date}...")

                try:
                    X_train, y_train, X_val, y_val = self.prepare_data(date)
                    val_acc = self.train_model(X_train, y_train, X_val, y_val)

                    results['retrain_dates'].append(date)
                    results['performance_history'].append({
                        'date': date,
                        'val_acc': float(val_acc),
                        'train_size': len(X_train),
                        'val_size': len(X_val)
                    })

                    print(f"  Val Acc: {val_acc:.4f}, Train: {len(X_train)}, Val: {len(X_val)}")

                except Exception as e:
                    print(f"  Error training: {e}")
                    continue

            # 预测当日
            if self.model is not None:
                try:
                    # 加载当日数据
                    day_data = self._load_date_range_data(date, date)
                    if day_data is None or day_data.is_empty():
                        continue

                    # 准备特征
                    available_cols = [col for col in FEATURE_COLS if col in day_data.columns]
                    X_today = day_data.select(available_cols).to_numpy()
                    X_today = np.nan_to_num(X_today, nan=0.0, posinf=0.0, neginf=0.0)

                    symbols = day_data["symbol"].to_list()

                    # 预测
                    probs = self.predict(X_today)

                    # 保存预测结果
                    for symbol, prob in zip(symbols, probs):
                        results['predictions'].append({
                            'date': date,
                            'symbol': symbol,
                            'prob': float(prob)
                        })

                except Exception as e:
                    print(f"  Error predicting on {date}: {e}")

            # 进度显示
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(all_dates)} dates processed")

        print(f"[{self.config.strategy_name}] Completed: {len(results['predictions'])} predictions")

        return results
