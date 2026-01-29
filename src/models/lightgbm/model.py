"""
LightGBM 模型实现

使用保守的超参数设置防止过拟合，适合量化交易场景。
"""

from typing import Dict, Optional, List
from pathlib import Path
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError("请安装 lightgbm: pip install lightgbm")

from ..base import BaseModel
from ..config import LightGBMConfig


class LightGBMModel(BaseModel):
    """LightGBM 二分类模型"""

    def __init__(self, config: LightGBMConfig, feature_names: Optional[List[str]] = None):
        """
        初始化模型

        Args:
            config: 模型配置
            feature_names: 特征名称列表（用于特征重要性分析）
        """
        self.config = config
        self.feature_names = feature_names
        self.model: Optional[lgb.Booster] = None
        self._best_iteration = 0

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        """
        训练模型

        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        Returns:
            训练指标
        """
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': self.config.num_leaves,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'min_child_samples': self.config.min_child_samples,
            'verbose': -1,
            'seed': self.config.random_seed,
        }

        # 创建数据集
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # 训练
        callbacks = [
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=0),  # 禁用日志输出
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._best_iteration = self.model.best_iteration

        # 计算验证集指标
        metrics = {
            'best_iteration': self._best_iteration,
        }

        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = np.mean((val_pred > 0.5).astype(int) == y_val)
            metrics['val_acc'] = float(val_acc)
            metrics['val_auc'] = float(self.model.best_score.get('valid', {}).get('auc', 0))

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵

        Returns:
            概率矩阵 (n_samples, 2)
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练")

        prob_1 = self.model.predict(X)
        prob_0 = 1 - prob_1
        return np.column_stack([prob_0, prob_1])

    def save(self, path: str) -> None:
        """保存模型"""
        if self.model is None:
            raise RuntimeError("模型尚未训练")
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """加载模型"""
        self.model = lgb.Booster(model_file=path)

    def feature_importance(self) -> Dict[str, float]:
        """返回特征重要性"""
        if self.model is None:
            return {}

        importance = self.model.feature_importance(importance_type='gain')
        names = self.model.feature_name()

        return dict(zip(names, importance))
