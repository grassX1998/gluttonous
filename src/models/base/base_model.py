"""
模型抽象基类

定义所有模型必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np


class BaseModel(ABC):
    """模型抽象基类"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        """
        训练模型

        Args:
            X: 训练特征 (n_samples, n_features)
            y: 训练标签 (n_samples,)
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）

        Returns:
            训练指标字典
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵 (n_samples, n_features)

        Returns:
            概率矩阵 (n_samples, 2)，第一列为负类概率，第二列为正类概率
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征矩阵

        Returns:
            类别数组
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型

        Args:
            path: 保存路径
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        加载模型

        Args:
            path: 模型路径
        """
        pass

    def feature_importance(self) -> Dict[str, float]:
        """
        返回特征重要性（可选实现）

        Returns:
            特征名称到重要性分数的映射
        """
        return {}
