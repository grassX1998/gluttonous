"""
方案5: 波动率自适应策略

根据市场波动率动态调整训练窗口和样本权重。

策略特点：
- 计算市场波动率指标
- 高波动期：使用较短窗口，更关注近期数据
- 低波动期：使用较长窗口，保留更多历史数据
- 动态适应市场环境变化
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


class VolatilityAdaptiveExecutor(BaseStrategyExecutor):
    """波动率自适应执行器"""

    def __init__(self, config):
        super().__init__(config)
        self.train_start_date = None
        self.last_train_date = None
        self.train_count = 0
        self.recent_volatility = []  # 近期波动率

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """准备数据（根据波动率调整窗口）"""
        config = self.config

        all_dates = self._get_trading_dates("2024-01-01", current_date)
        current_idx = all_dates.index(current_date) if current_date in all_dates else len(all_dates) - 1

        # 计算市场波动率
        vol_lookback = getattr(config, 'vol_lookback', 20)
        market_vol = self._calculate_market_volatility(current_date, vol_lookback)

        # 记录波动率
        self.recent_volatility.append(market_vol)
        if len(self.recent_volatility) > 20:
            self.recent_volatility.pop(0)

        # 根据波动率调整训练窗口大小
        train_days = self._get_adaptive_window_size(market_vol)

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

        # 根据自适应窗口大小调整起始日期
        train_start_idx = max(0, train_end_idx - train_days + 1)
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
        retrain_interval = getattr(config, 'retrain_interval', 3)
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

    def _calculate_market_volatility(self, current_date: str, lookback: int) -> float:
        """
        计算市场波动率（简化版本）

        Args:
            current_date: 当前日期
            lookback: 回溯天数

        Returns:
            波动率（0-1之间的值）
        """
        try:
            # 获取最近N天的数据
            all_dates = self._get_trading_dates("2024-01-01", current_date)
            current_idx = all_dates.index(current_date)

            start_idx = max(0, current_idx - lookback)
            start_date = all_dates[start_idx]

            # 加载数据
            data = self._load_date_range_data(start_date, current_date)
            if data is None or data.is_empty():
                return 0.02  # 默认波动率

            # 计算每日平均收益率和波动率
            if "return_1d" in data.columns:
                returns = data["return_1d"].to_numpy()
                returns = returns[~np.isnan(returns)]

                if len(returns) > 0:
                    vol = np.std(returns)
                    return float(np.clip(vol, 0.01, 0.10))  # 限制在1%-10%之间

            return 0.02  # 默认2%波动率

        except Exception as e:
            return 0.02  # 出错时返回默认值

    def _get_adaptive_window_size(self, market_vol: float) -> int:
        """
        根据市场波动率确定训练窗口大小

        Args:
            market_vol: 市场波动率

        Returns:
            训练窗口天数
        """
        config = self.config
        vol_high = getattr(config, 'vol_high_threshold', 0.03)
        vol_low = getattr(config, 'vol_low_threshold', 0.015)

        if market_vol >= vol_high:
            # 高波动：使用短窗口（更关注近期）
            return 40
        elif market_vol <= vol_low:
            # 低波动：使用长窗口（保留更多历史）
            return 120
        else:
            # 中等波动：使用中等窗口
            return 60

    def _extract_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """从DataFrame提取特征和标签"""
        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        df_clean = df.drop_nulls(subset=available_cols + ["label"])

        X = df_clean.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        y = df_clean["label"].to_numpy().astype(int)

        return X, y

    def run(self, start_date: str, end_date: str):
        """运行波动率自适应回测"""
        result = super().run(start_date, end_date)

        if result['retrain_dates']:
            self.last_train_date = result['retrain_dates'][-1]

        result['strategy_info'] = {
            'train_start_date': self.train_start_date,
            'train_count': self.train_count,
            'avg_volatility': np.mean(self.recent_volatility) if self.recent_volatility else 0.02,
            'config': {
                'vol_lookback': getattr(self.config, 'vol_lookback', 20),
                'vol_high_threshold': getattr(self.config, 'vol_high_threshold', 0.03),
                'vol_low_threshold': getattr(self.config, 'vol_low_threshold', 0.015),
            }
        }

        return result
