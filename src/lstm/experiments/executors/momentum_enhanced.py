"""
方案6: 动量增强策略

结合短期动量信号增强LSTM预测。

策略特点：
- LSTM预测基础概率
- 短期动量作为补充信号
- 加权组合两者的预测结果
- 给予近期强势股票更高权重
"""

import sys
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.experiments.base_executor import BaseStrategyExecutor
from pipeline.data_cleaning.features import FEATURE_COLS


class MomentumEnhancedExecutor(BaseStrategyExecutor):
    """动量增强执行器"""

    def __init__(self, config):
        super().__init__(config)
        self.train_start_date = None
        self.last_train_date = None
        self.train_count = 0
        self.momentum_weight = getattr(config, 'momentum_weight', 0.3)  # 动量权重
        self.momentum_lookback = getattr(config, 'momentum_lookback', 10)  # 动量回溯期

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备数据（扩展窗口 + 样本权重）"""
        config = self.config

        all_dates = self._get_trading_dates("2024-01-01", current_date)
        current_idx = all_dates.index(current_date) if current_date in all_dates else len(all_dates) - 1

        # 确定训练起始日期（首次训练）
        if self.train_start_date is None:
            min_train_days = getattr(config, 'min_train_days', 60)
            start_idx = max(0, current_idx - min_train_days)
            self.train_start_date = all_dates[start_idx]

        # 训练集范围
        val_days = getattr(config, 'val_days', 1)
        train_end_idx = current_idx - val_days - 1
        if train_end_idx < 0:
            raise ValueError(f"Not enough data for training on {current_date}")

        train_end_date = all_dates[train_end_idx]

        # 验证集
        val_idx = current_idx - 1
        if val_idx < 0:
            raise ValueError(f"Not enough data for validation on {current_date}")
        val_date = all_dates[val_idx]

        # 限制训练集大小
        max_train_days = getattr(config, 'max_train_days', 500)
        train_start_idx = all_dates.index(self.train_start_date)
        train_size = train_end_idx - train_start_idx + 1

        if train_size > max_train_days:
            train_start_idx = train_end_idx - max_train_days + 1
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

        # 应用样本权重
        train_dates = train_data["date"].to_list()
        weights = self._calculate_sample_weights(train_dates, current_date)

        # 根据权重采样
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
        """判断是否需要重训练"""
        if self.model is None:
            return True

        if self.last_train_date is None:
            return True

        config = self.config
        retrain_interval = getattr(config, 'retrain_interval', 1)
        all_dates = self._get_trading_dates(self.last_train_date, current_date)

        if len(all_dates) >= retrain_interval:
            return True

        return False

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> float:
        """训练模型"""
        val_acc = super().train_model(X_train, y_train, X_val, y_val)
        self.train_count += 1
        return val_acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测（结合LSTM预测和动量信号）

        Args:
            X: 特征矩阵

        Returns:
            增强后的预测概率
        """
        # 获取LSTM基础预测
        lstm_probs = super().predict(X)

        # 计算动量得分（从特征中提取）
        momentum_scores = self._calculate_momentum_scores(X)

        # 组合预测
        enhanced_probs = (1 - self.momentum_weight) * lstm_probs + self.momentum_weight * momentum_scores

        # 确保在[0, 1]范围内
        enhanced_probs = np.clip(enhanced_probs, 0.0, 1.0)

        return enhanced_probs

    def _calculate_momentum_scores(self, X: np.ndarray) -> np.ndarray:
        """
        从特征中计算动量得分

        Args:
            X: 特征矩阵

        Returns:
            动量得分（0-1之间）
        """
        # 尝试从特征中提取动量指标
        # 假设return_5d, return_10d等在FEATURE_COLS中
        momentum_score = np.zeros(len(X))

        try:
            # 查找收益率特征的索引
            return_5d_idx = None
            return_10d_idx = None

            for i, col in enumerate(FEATURE_COLS):
                if col == 'return_5d':
                    return_5d_idx = i
                elif col == 'return_10d':
                    return_10d_idx = i

            # 基于短期收益率计算动量得分
            if return_5d_idx is not None:
                returns_5d = X[:, return_5d_idx]
                # 归一化到0-1范围（使用sigmoid）
                momentum_score = 1 / (1 + np.exp(-returns_5d * 10))

            elif return_10d_idx is not None:
                returns_10d = X[:, return_10d_idx]
                momentum_score = 1 / (1 + np.exp(-returns_10d * 10))

        except Exception as e:
            # 如果计算失败，返回0.5（中性）
            momentum_score = np.ones(len(X)) * 0.5

        return momentum_score

    def _calculate_sample_weights(self, dates: list, current_date: str) -> np.ndarray:
        """计算样本权重（指数衰减）"""
        config = self.config

        # 转换为datetime
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')

        date_dts = []
        for d in dates:
            if isinstance(d, str):
                date_dts.append(datetime.strptime(d, '%Y-%m-%d'))
            else:
                date_dts.append(datetime.combine(d, datetime.min.time()))

        # 计算天数差距
        days_diff = np.array([(current_dt - dt).days for dt in date_dts])

        # 指数衰减权重
        weight_decay_days = getattr(config, 'weight_decay_days', 30)
        weight_decay_rate = getattr(config, 'weight_decay_rate', 0.95)
        weights = weight_decay_rate ** (days_diff / weight_decay_days)

        return weights

    def _extract_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """从DataFrame提取特征和标签"""
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols + ["label"])

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df_clean["label"].to_numpy().astype(int)

        return X, y

    def run(self, start_date: str, end_date: str):
        """运行动量增强回测"""
        result = super().run(start_date, end_date)

        if result['retrain_dates']:
            self.last_train_date = result['retrain_dates'][-1]

        result['strategy_info'] = {
            'train_start_date': self.train_start_date,
            'train_count': self.train_count,
            'config': {
                'momentum_weight': self.momentum_weight,
                'momentum_lookback': self.momentum_lookback,
                'weight_decay_rate': getattr(self.config, 'weight_decay_rate', 0.95),
            }
        }

        return result
