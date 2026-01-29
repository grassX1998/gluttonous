"""
执行器抽象基类

提供数据加载、回测循环等通用逻辑，子类只需实现模型创建和训练。
复用 LSTM 框架的数据加载和缓存机制。
"""

import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import polars as pl

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.config import (
    FEATURE_DATA_MONTHLY_DIR,
    MODEL_CHECKPOINT_DIR,
)
from pipeline.data_cleaning.features import FEATURE_COLS
from pipeline.shared.logging_config import get_lstm_training_logger


class BaseExecutor(ABC):
    """执行器抽象基类"""

    # ===== 类级别缓存（所有实例共享） =====
    _month_data_cache: Dict[str, pl.DataFrame] = {}
    _trading_dates_cache: List[str] = None

    @classmethod
    def clear_cache(cls):
        """清除所有缓存"""
        cls._month_data_cache.clear()
        cls._trading_dates_cache = None

    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """
        初始化执行器

        Args:
            config: 策略配置对象
            logger: 日志记录器
        """
        self.config = config
        self.model = None
        self.logger = logger or get_lstm_training_logger()

        # 特征列（可能需要过滤）
        self.feature_cols = FEATURE_COLS

        # 标准化参数
        self.scaler_params = None

    @abstractmethod
    def create_model(self):
        """创建模型实例"""
        pass

    def run(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        执行回测（核心流程）

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            回测结果字典
        """
        strategy_name = getattr(self.config, 'strategy_name', self.__class__.__name__)

        results = {
            'strategy': strategy_name,
            'start_date': start_date,
            'end_date': end_date,
            'predictions': [],
            'retrain_dates': [],
            'performance_history': []
        }

        # 获取所有交易日
        all_dates = self._get_trading_dates(start_date, end_date)
        self.logger.info(f"[{strategy_name}] Trading dates: {len(all_dates)}")

        # 记录上次训练日期（用于判断是否需要重训练）
        last_train_date = None

        # 逐日预测
        for i, date in enumerate(all_dates):
            # 判断是否需要重训练
            if self.should_retrain(date, last_train_date):
                self.logger.info(f"[{strategy_name}] Training on {date}...")

                try:
                    X_train, y_train, X_val, y_val = self.prepare_data(date)

                    if len(X_train) == 0:
                        self.logger.warning(f"  No training data for {date}")
                        continue

                    # 训练模型
                    metrics = self.train_model(X_train, y_train, X_val, y_val)

                    last_train_date = date
                    results['retrain_dates'].append(date)
                    results['performance_history'].append({
                        'date': date,
                        'metrics': metrics,
                        'train_size': len(X_train),
                        'val_size': len(X_val) if X_val is not None else 0
                    })

                    self.logger.info(f"  Metrics: {metrics}, Train: {len(X_train)}")

                except Exception as e:
                    self.logger.error(f"  Error training: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            # 预测当日
            if self.model is not None:
                try:
                    predictions = self.predict_day(date)
                    results['predictions'].extend(predictions)
                except Exception as e:
                    self.logger.error(f"  Error predicting on {date}: {e}")

            # 进度显示
            if (i + 1) % 20 == 0:
                self.logger.info(f"  Progress: {i+1}/{len(all_dates)} dates processed")

        self.logger.info(f"[{strategy_name}] Completed: {len(results['predictions'])} predictions")

        return results

    def should_retrain(self, current_date: str, last_train_date: Optional[str]) -> bool:
        """
        判断是否需要重训练

        Args:
            current_date: 当前日期
            last_train_date: 上次训练日期

        Returns:
            是否需要重训练
        """
        if last_train_date is None:
            return True

        # 计算距离上次训练的天数
        all_dates = self._get_trading_dates(last_train_date, current_date)
        days_since_train = len(all_dates) - 1

        retrain_interval = getattr(self.config, 'retrain_interval', 1)
        return days_since_train >= retrain_interval

    def prepare_data(self, current_date: str) -> Tuple[np.ndarray, np.ndarray,
                                                        Optional[np.ndarray], Optional[np.ndarray]]:
        """
        准备训练和验证数据

        Args:
            current_date: 当前日期

        Returns:
            X_train, y_train, X_val, y_val
        """
        train_days = getattr(self.config, 'train_days', 60)
        val_days = getattr(self.config, 'val_days', 5)
        holding_days = getattr(self.config, 'holding_days', 5)
        sample_ratio = getattr(self.config, 'sample_ratio', 0.5)
        random_seed = getattr(self.config, 'random_seed', 42)

        # 获取历史交易日
        # 需要更早的日期来获取足够的训练数据
        earliest_date = "2024-01-01"
        all_dates = self._get_trading_dates(earliest_date, current_date)

        # 找到当前日期的索引
        try:
            current_idx = all_dates.index(current_date)
        except ValueError:
            # 如果当前日期不在列表中，找到最近的日期
            current_idx = len(all_dates) - 1

        # 计算数据范围
        # val_end = current_date - holding_days（避免标签泄露）
        val_end_idx = current_idx - holding_days
        if val_end_idx < 0:
            return np.array([]), np.array([]), None, None

        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - 1
        train_start_idx = max(0, train_end_idx - train_days + 1)

        if train_start_idx >= train_end_idx or val_start_idx > val_end_idx:
            return np.array([]), np.array([]), None, None

        train_start = all_dates[train_start_idx]
        train_end = all_dates[train_end_idx]
        val_start = all_dates[val_start_idx]
        val_end = all_dates[val_end_idx]

        # 加载数据
        train_df = self._load_date_range_data(train_start, train_end)
        val_df = self._load_date_range_data(val_start, val_end)

        if train_df is None or train_df.is_empty():
            return np.array([]), np.array([]), None, None

        # 只使用指数成分股
        if "in_index" in train_df.columns:
            train_df = train_df.filter(pl.col("in_index") == 1)
        if val_df is not None and "in_index" in val_df.columns:
            val_df = val_df.filter(pl.col("in_index") == 1)

        # 提取特征和标签
        available_cols = [col for col in self.feature_cols if col in train_df.columns]

        X_train = train_df.select(available_cols).to_numpy()
        y_train = train_df["label"].to_numpy() if "label" in train_df.columns else np.zeros(len(X_train))

        # 处理缺失值
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        # 采样
        if sample_ratio < 1.0:
            np.random.seed(random_seed)
            n_samples = int(len(X_train) * sample_ratio)
            indices = np.random.choice(len(X_train), size=n_samples, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]

        # 验证集
        X_val, y_val = None, None
        if val_df is not None and not val_df.is_empty():
            X_val = val_df.select(available_cols).to_numpy()
            y_val = val_df["label"].to_numpy() if "label" in val_df.columns else np.zeros(len(X_val))
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        return X_train, y_train, X_val, y_val

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> Dict:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练指标
        """
        # 标准化
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        X_std[X_std == 0] = 1

        X_train_norm = (X_train - X_mean) / X_std
        self.scaler_params = {'mean': X_mean, 'std': X_std}

        X_val_norm = None
        if X_val is not None:
            X_val_norm = (X_val - X_mean) / X_std

        # 创建并训练模型
        self.model = self.create_model()
        metrics = self.model.fit(X_train_norm, y_train, X_val_norm, y_val)

        return metrics

    def predict_day(self, date: str) -> List[Dict]:
        """
        预测单日所有股票

        Args:
            date: 日期

        Returns:
            预测结果列表 [{date, symbol, prob}, ...]
        """
        if self.model is None or self.scaler_params is None:
            return []

        # 加载当日数据
        day_data = self._load_date_range_data(date, date)
        if day_data is None or day_data.is_empty():
            return []

        # 过滤指数成分股
        if "in_index" in day_data.columns:
            day_data = day_data.filter(pl.col("in_index") == 1)
            if day_data.is_empty():
                return []

        # 准备特征
        available_cols = [col for col in self.feature_cols if col in day_data.columns]
        X = day_data.select(available_cols).to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 标准化
        X_norm = (X - self.scaler_params['mean']) / self.scaler_params['std']

        # 预测
        probs = self.model.predict_proba(X_norm)
        prob_positive = probs[:, 1]  # 正类概率

        # 获取股票代码
        symbols = day_data["symbol"].to_list()

        # 构建结果
        predictions = []
        for symbol, prob in zip(symbols, prob_positive):
            predictions.append({
                'date': date,
                'symbol': symbol,
                'prob': float(prob)
            })

        return predictions

    # ===== 数据加载方法（复用 LSTM 框架的缓存机制） =====

    def _load_month_data(self, month: str) -> Optional[pl.DataFrame]:
        """加载单月特征数据（带缓存）"""
        if month in self._month_data_cache:
            return self._month_data_cache[month]

        path = FEATURE_DATA_MONTHLY_DIR / f"{month}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            self._month_data_cache[month] = df
            return df

        return None

    def _load_date_range_data(self, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """加载日期范围内的数据"""
        start_ym = start_date[:7]
        end_ym = end_date[:7]

        available_months = sorted([p.stem for p in FEATURE_DATA_MONTHLY_DIR.glob("*.parquet")])
        needed_months = [m for m in available_months if start_ym <= m <= end_ym]

        dfs = []
        for month in needed_months:
            df = self._load_month_data(month)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return None

        df = pl.concat(dfs)
        df = df.with_columns(pl.col("date").cast(pl.Utf8).alias("date_str"))
        df = df.filter(
            (pl.col("date_str") >= start_date) & (pl.col("date_str") <= end_date)
        ).drop("date_str")

        return df

    def _get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取日期范围内的所有交易日（带缓存）"""
        if self._trading_dates_cache is None:
            all_months = sorted([p.stem for p in FEATURE_DATA_MONTHLY_DIR.glob("*.parquet")])

            dates_set = set()
            for month in all_months:
                df = self._load_month_data(month)
                if df is not None:
                    dates = df.select(pl.col("date").cast(pl.Utf8)).unique().to_series().to_list()
                    dates_set.update(dates)

            BaseExecutor._trading_dates_cache = sorted(dates_set)

        trading_dates = [d for d in self._trading_dates_cache if start_date <= d <= end_date]
        return trading_dates
