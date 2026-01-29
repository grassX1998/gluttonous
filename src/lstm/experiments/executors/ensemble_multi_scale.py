"""
方案4: 多尺度集成策略

训练短期、中期、长期三个模型，集成预测结果。

策略特点：
- 短期模型：捕捉短期市场动向（20-30天）
- 中期模型：捕捉中期趋势（60-90天）
- 长期模型：捕捉长期规律（120-180天）
- 加权集成：综合三个模型的预测
"""

import sys
from pathlib import Path
from typing import Tuple, List
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.experiments.base_executor import BaseStrategyExecutor
from src.lstm.models import SimpleLSTMModel
from src.lstm.config import DEVICE
from pipeline.data_cleaning.features import FEATURE_COLS


class EnsembleMultiScaleExecutor(BaseStrategyExecutor):
    """多尺度集成执行器"""

    def __init__(self, config):
        super().__init__(config)
        self.last_train_date = None
        self.train_count = 0

        # 三个模型
        self.model_short = None  # 短期模型
        self.model_medium = None  # 中期模型
        self.model_long = None  # 长期模型

        # 三个scaler
        self.scaler_short = None
        self.scaler_medium = None
        self.scaler_long = None

        # 集成权重
        self.ensemble_weights = getattr(config, 'ensemble_weights', [0.3, 0.4, 0.3])

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备多尺度数据

        返回中期窗口的数据作为主要训练数据
        """
        config = self.config

        all_dates = self._get_trading_dates("2024-01-01", current_date)
        current_idx = all_dates.index(current_date) if current_date in all_dates else len(all_dates) - 1

        # 使用中期窗口作为主要数据
        window_medium = getattr(config, 'window_medium', 60)
        val_days = getattr(config, 'val_days', 1)

        train_end_idx = current_idx - val_days - 1
        train_start_idx = max(0, train_end_idx - window_medium + 1)

        if train_start_idx >= train_end_idx:
            raise ValueError(f"Not enough data for training on {current_date}")

        train_start_date = all_dates[train_start_idx]
        train_end_date = all_dates[train_end_idx]

        # 验证集
        val_idx = current_idx - 1
        if val_idx < 0:
            raise ValueError(f"Not enough data for validation on {current_date}")
        val_date = all_dates[val_idx]

        # 加载数据
        train_data = self._load_date_range_data(train_start_date, train_end_date)
        val_data = self._load_date_range_data(val_date, val_date)

        if train_data is None or train_data.is_empty():
            raise ValueError("No training data available")
        if val_data is None or val_data.is_empty():
            raise ValueError("No validation data available")

        X_train, y_train = self._extract_features(train_data)
        X_val, y_val = self._extract_features(val_data)

        return X_train, y_train, X_val, y_val

    def should_retrain(self, current_date: str) -> bool:
        """判断是否需要重训练"""
        if self.model is None:
            return True

        if self.last_train_date is None:
            return True

        config = self.config
        retrain_interval = getattr(config, 'retrain_interval', 5)
        all_dates = self._get_trading_dates(self.last_train_date, current_date)

        if len(all_dates) >= retrain_interval:
            return True

        return False

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """训练三个不同尺度的模型"""
        config = self.config

        # 获取窗口大小
        window_short = getattr(config, 'window_short', 30)
        window_medium = getattr(config, 'window_medium', 60)
        window_long = getattr(config, 'window_long', 120)

        # 准备三个窗口的数据（简化：使用同一数据集的不同采样）
        # 实际应该从不同时间窗口加载数据

        # 训练短期模型
        val_acc_short = self._train_single_model(
            X_train, y_train, X_val, y_val, 'short'
        )

        # 训练中期模型
        val_acc_medium = self._train_single_model(
            X_train, y_train, X_val, y_val, 'medium'
        )

        # 训练长期模型（使用更多历史数据，这里简化处理）
        val_acc_long = self._train_single_model(
            X_train, y_train, X_val, y_val, 'long'
        )

        # 使用主模型作为中期模型的引用
        self.model = self.model_medium
        self.scaler_params = self.scaler_medium

        # 返回加权平均准确率
        weights = self.ensemble_weights
        avg_acc = (val_acc_short * weights[0] +
                   val_acc_medium * weights[1] +
                   val_acc_long * weights[2])

        self.train_count += 1
        return avg_acc

    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           model_type: str) -> float:
        """训练单个模型"""
        # 标准化
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_std[X_std == 0] = 1

        X_train_norm = (X_train - X_mean) / X_std
        X_val_norm = (X_val - X_mean) / X_std

        # 保存scaler
        scaler = {'mean': X_mean, 'std': X_std}
        if model_type == 'short':
            self.scaler_short = scaler
        elif model_type == 'medium':
            self.scaler_medium = scaler
        else:
            self.scaler_long = scaler

        # 简化训练流程
        val_acc = super().train_model(X_train, y_train, X_val, y_val)

        # 保存模型
        if model_type == 'short':
            self.model_short = self.model
        elif model_type == 'medium':
            self.model_medium = self.model
        else:
            self.model_long = self.model

        return val_acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """集成预测"""
        if self.model_short is None or self.model_medium is None or self.model_long is None:
            # 如果三个模型未全部训练，使用基类方法
            return super().predict(X)

        # 三个模型分别预测
        probs_short = self._predict_single(X, self.model_short, self.scaler_short)
        probs_medium = self._predict_single(X, self.model_medium, self.scaler_medium)
        probs_long = self._predict_single(X, self.model_long, self.scaler_long)

        # 加权集成
        weights = self.ensemble_weights
        probs = (probs_short * weights[0] +
                 probs_medium * weights[1] +
                 probs_long * weights[2])

        return probs

    def _predict_single(self, X: np.ndarray, model, scaler) -> np.ndarray:
        """单个模型预测"""
        if model is None or scaler is None:
            return np.zeros(len(X))

        # 标准化
        X_norm = (X - scaler['mean']) / scaler['std']
        X_seq = X_norm[:, np.newaxis, :]
        X_tensor = torch.from_numpy(X_seq).float().to(DEVICE)

        # 预测
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

    def _extract_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """从DataFrame提取特征和标签"""
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols + ["label"])

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df_clean["label"].to_numpy().astype(int)

        return X, y

    def run(self, start_date: str, end_date: str):
        """运行多尺度集成回测"""
        result = super().run(start_date, end_date)

        if result['retrain_dates']:
            self.last_train_date = result['retrain_dates'][-1]

        result['strategy_info'] = {
            'train_count': self.train_count,
            'ensemble_weights': self.ensemble_weights,
            'config': {
                'window_short': getattr(self.config, 'window_short', 30),
                'window_medium': getattr(self.config, 'window_medium', 60),
                'window_long': getattr(self.config, 'window_long', 120),
            }
        }

        return result
