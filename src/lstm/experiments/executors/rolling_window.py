"""
方案2: 滚动窗口策略

固定长度训练集，每次训练使用最近N天的数据。

策略特点：
- 训练集大小固定（如60天）
- 窗口随时间向前滚动
- 适合市场环境快速变化的场景
"""

import sys
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.experiments.base_executor import BaseStrategyExecutor
from src.lstm.config import RollingKFoldConfig
from pipeline.data_cleaning.features import FEATURE_COLS


class RollingWindowExecutor(BaseStrategyExecutor):
    """滚动窗口执行器"""

    def __init__(self, config: RollingKFoldConfig):
        super().__init__(config)
        self.last_train_date = None  # 上次训练的日期
        self.train_count = 0  # 训练次数

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备滚动窗口数据

        策略：
        1. 训练集：当前日期前N天（固定窗口）
        2. 验证集：当前日期前1天
        """
        config: RollingKFoldConfig = self.config

        # 获取所有可用交易日
        all_dates = self._get_trading_dates("2024-01-01", current_date)
        current_idx = all_dates.index(current_date) if current_date in all_dates else len(all_dates) - 1

        # 确定训练窗口
        train_end_idx = current_idx - config.val_days - 1
        train_start_idx = max(0, train_end_idx - config.train_days + 1)

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

        # 准备特征
        X_train, y_train = self._extract_features(train_data)
        X_val, y_val = self._extract_features(val_data)

        return X_train, y_train, X_val, y_val

    def should_retrain(self, current_date: str) -> bool:
        """
        滚动窗口策略：定期重训练

        Args:
            current_date: 当前日期

        Returns:
            是否需要重训练
        """
        config: RollingKFoldConfig = self.config

        # 首次训练
        if self.model is None:
            return True

        # 按照固定间隔重训练
        if self.last_train_date is None:
            return True

        all_dates = self._get_trading_dates(self.last_train_date, current_date)

        # 如果距离上次训练超过retrain_interval天
        if len(all_dates) >= config.retrain_interval:
            return True

        return False

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """训练模型"""
        val_acc = super().train_model(X_train, y_train, X_val, y_val)
        self.train_count += 1
        return val_acc

    def _extract_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """从DataFrame提取特征和标签"""
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols + ["label"])

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df_clean["label"].to_numpy().astype(int)

        return X, y

    def run(self, start_date: str, end_date: str):
        """运行滚动窗口回测"""
        result = super().run(start_date, end_date)

        # 记录最后一次训练的日期
        if result['retrain_dates']:
            self.last_train_date = result['retrain_dates'][-1]

        # 添加策略特定信息
        result['strategy_info'] = {
            'train_count': self.train_count,
            'config': {
                'train_days': self.config.train_days,
                'retrain_interval': self.config.retrain_interval,
            }
        }

        return result
