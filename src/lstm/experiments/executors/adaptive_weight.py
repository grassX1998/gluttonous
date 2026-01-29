"""
方案3: 自适应权重策略

根据验证准确率动态调整样本权重衰减率。

策略特点：
- 累积历史数据（类似扩展窗口）
- 根据验证准确率自适应调整权重衰减
- 准确率高时降低衰减（保留更多历史数据）
- 准确率低时增加衰减（更关注近期数据）
"""

import sys
from pathlib import Path
from typing import Tuple
from datetime import datetime, timedelta

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.lstm.experiments.base_executor import BaseStrategyExecutor
from src.lstm.config import ExpandingWindowConfig  # 复用配置
from pipeline.data_cleaning.features import FEATURE_COLS


class AdaptiveWeightExecutor(BaseStrategyExecutor):
    """自适应权重执行器"""

    def __init__(self, config):
        super().__init__(config)
        self.train_start_date = None
        self.last_train_date = None
        self.train_count = 0
        self.recent_val_accs = []  # 最近几次的验证准确率
        self.current_decay_rate = getattr(config, 'base_weight_decay', 0.95)

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备数据（类似扩展窗口）"""
        config = self.config

        # 获取所有可用交易日
        all_dates = self._get_trading_dates("2024-01-01", current_date)
        current_idx = all_dates.index(current_date) if current_date in all_dates else len(all_dates) - 1

        # 确定训练起始日期（首次训练）
        if self.train_start_date is None:
            min_train_days = getattr(config, 'min_train_days', 60)
            start_idx = max(0, current_idx - min_train_days)
            self.train_start_date = all_dates[start_idx]

        # 训练集结束日期
        val_days = getattr(config, 'val_days', 1)
        train_end_idx = current_idx - val_days - 1
        if train_end_idx < 0:
            raise ValueError(f"Not enough data for training on {current_date}")

        train_end_date = all_dates[train_end_idx]

        # 验证集日期
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

        # 应用自适应权重
        train_dates = train_data["date"].to_list()
        weights = self._calculate_adaptive_weights(train_dates, current_date)

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
        """训练模型并记录验证准确率"""
        val_acc = super().train_model(X_train, y_train, X_val, y_val)

        # 记录准确率
        self.recent_val_accs.append(val_acc)
        if len(self.recent_val_accs) > 5:  # 只保留最近5次
            self.recent_val_accs.pop(0)

        # 根据准确率调整衰减率
        self._adjust_decay_rate()

        self.train_count += 1
        return val_acc

    def _calculate_adaptive_weights(self, dates: list, current_date: str) -> np.ndarray:
        """计算自适应权重"""
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

        # 使用当前衰减率计算权重
        weight_decay_days = getattr(self.config, 'weight_decay_days', 30)
        weights = self.current_decay_rate ** (days_diff / weight_decay_days)

        return weights

    def _adjust_decay_rate(self):
        """根据近期验证准确率调整衰减率"""
        if len(self.recent_val_accs) < 2:
            return

        config = self.config
        avg_acc = np.mean(self.recent_val_accs)
        base_rate = getattr(config, 'base_weight_decay', 0.95)
        acc_high = getattr(config, 'acc_threshold_high', 0.65)
        acc_low = getattr(config, 'acc_threshold_low', 0.55)

        if avg_acc >= acc_high:
            # 准确率高，降低衰减（保留更多历史）
            self.current_decay_rate = min(0.98, base_rate + 0.02)
        elif avg_acc <= acc_low:
            # 准确率低，增加衰减（更关注近期）
            self.current_decay_rate = max(0.90, base_rate - 0.03)
        else:
            # 中等准确率，使用基础衰减率
            self.current_decay_rate = base_rate

    def _extract_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """从DataFrame提取特征和标签"""
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols + ["label"])

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df_clean["label"].to_numpy().astype(int)

        return X, y

    def run(self, start_date: str, end_date: str):
        """运行自适应权重回测"""
        result = super().run(start_date, end_date)

        if result['retrain_dates']:
            self.last_train_date = result['retrain_dates'][-1]

        result['strategy_info'] = {
            'train_start_date': self.train_start_date,
            'train_count': self.train_count,
            'final_decay_rate': self.current_decay_rate,
            'config': {
                'base_weight_decay': getattr(self.config, 'base_weight_decay', 0.95),
                'acc_threshold_high': getattr(self.config, 'acc_threshold_high', 0.65),
                'acc_threshold_low': getattr(self.config, 'acc_threshold_low', 0.55),
            }
        }

        return result
