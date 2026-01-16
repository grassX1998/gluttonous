"""
方案1: 扩展窗口策略

累积扩展训练集，保留所有历史数据，使用样本权重让近期数据更重要。

策略特点：
- 训练集持续增长（从min_train_days到max_train_days）
- 使用指数衰减权重：近期数据权重高，远期数据权重低
- 适合市场存在长期趋势的场景
"""

import sys
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.experiments.base_executor import BaseStrategyExecutor
from src.lstm.config import ExpandingWindowConfig
from pipeline.data_cleaning.features import FEATURE_COLS


class ExpandingWindowExecutor(BaseStrategyExecutor):
    """扩展窗口执行器"""

    def __init__(self, config: ExpandingWindowConfig):
        super().__init__(config)
        self.train_start_date = None  # 训练集起始日期
        self.last_train_date = None  # 上次训练的日期
        self.train_count = 0  # 训练次数

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        准备扩展窗口数据

        策略：
        1. 确定训练起始日期（如果首次训练）
        2. 训练集：从起始日期到当前日期前N天
        3. 验证集：当前日期前1天
        4. 应用样本权重（近期数据权重更高）
        """
        config: ExpandingWindowConfig = self.config

        # 获取所有可用交易日
        all_dates = self._get_trading_dates("2024-01-01", current_date)
        current_idx = all_dates.index(current_date) if current_date in all_dates else len(all_dates) - 1

        # 确定训练起始日期（首次训练）
        if self.train_start_date is None:
            start_idx = max(0, current_idx - config.min_train_days)
            self.train_start_date = all_dates[start_idx]

        # 训练集结束日期 = 当前日期 - val_days - 1
        train_end_idx = current_idx - config.val_days - 1
        if train_end_idx < 0:
            raise ValueError(f"Not enough data for training on {current_date}")

        train_end_date = all_dates[train_end_idx]

        # 验证集日期 = 当前日期 - 1
        val_idx = current_idx - 1
        if val_idx < 0:
            raise ValueError(f"Not enough data for validation on {current_date}")

        val_date = all_dates[val_idx]

        # 检查训练集大小限制
        train_start_idx = all_dates.index(self.train_start_date)
        train_size = train_end_idx - train_start_idx + 1

        # 如果训练集过大，截断前面的数据
        if train_size > config.max_train_days:
            train_start_idx = train_end_idx - config.max_train_days + 1
            self.train_start_date = all_dates[train_start_idx]

        # 加载数据
        train_data = self._load_date_range_data(self.train_start_date, train_end_date)
        val_data = self._load_date_range_data(val_date, val_date)

        if train_data is None or train_data.is_empty():
            raise ValueError("No training data available")

        if val_data is None or val_data.is_empty():
            raise ValueError("No validation data available")

        # 准备特征
        X_train, y_train = self._extract_features(train_data)
        X_val, y_val = self._extract_features(val_data)

        # 应用样本权重（可选）
        if config.use_sample_weight:
            # 计算每个样本距离当前日期的天数
            train_dates = train_data["date"].to_list()
            weights = self._calculate_sample_weights(train_dates, current_date)

            # 根据权重进行采样（权重高的样本被重复采样）
            n_samples = len(X_train)
            sample_indices = np.random.choice(
                n_samples,
                size=n_samples,
                replace=True,
                p=weights / weights.sum()
            )

            X_train = X_train[sample_indices]
            y_train = y_train[sample_indices]

        return X_train, y_train, X_val, y_val

    def should_retrain(self, current_date: str) -> bool:
        """
        扩展窗口策略：定期重训练

        Args:
            current_date: 当前日期

        Returns:
            是否需要重训练
        """
        config: ExpandingWindowConfig = self.config

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
        """
        训练模型（调用父类方法，并记录训练日期）
        """
        val_acc = super().train_model(X_train, y_train, X_val, y_val)

        # 更新训练状态
        self.train_count += 1

        return val_acc

    def _extract_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        从DataFrame提取特征和标签

        Args:
            df: 特征DataFrame

        Returns:
            X (features), y (labels)
        """
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols + ["label"])

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df_clean["label"].to_numpy().astype(int)

        return X, y

    def _calculate_sample_weights(self, dates: list, current_date: str) -> np.ndarray:
        """
        计算样本权重（指数衰减）

        Args:
            dates: 样本日期列表
            current_date: 当前日期

        Returns:
            权重数组
        """
        config: ExpandingWindowConfig = self.config

        # 转换为datetime
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')
        date_dts = [datetime.strptime(d, '%Y-%m-%d') for d in dates]

        # 计算天数差距
        days_diff = np.array([(current_dt - dt).days for dt in date_dts])

        # 指数衰减权重
        # weight = decay_rate ^ (days_diff / decay_days)
        weights = config.weight_decay_rate ** (days_diff / config.weight_decay_days)

        return weights

    def run(self, start_date: str, end_date: str):
        """
        运行扩展窗口回测（重写以记录最后训练日期）
        """
        result = super().run(start_date, end_date)

        # 记录最后一次训练的日期
        if result['retrain_dates']:
            self.last_train_date = result['retrain_dates'][-1]

        # 添加策略特定信息
        result['strategy_info'] = {
            'train_start_date': self.train_start_date,
            'train_count': self.train_count,
            'config': {
                'min_train_days': self.config.min_train_days,
                'max_train_days': self.config.max_train_days,
                'use_sample_weight': self.config.use_sample_weight,
                'weight_decay_days': self.config.weight_decay_days,
                'weight_decay_rate': self.config.weight_decay_rate,
            }
        }

        return result
