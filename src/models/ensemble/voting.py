"""
投票集成模型

支持软投票（概率加权平均）和硬投票（多数表决）。
"""

from typing import Dict, List, Optional
import numpy as np

from ..base import BaseModel


class VotingEnsemble(BaseModel):
    """投票集成模型"""

    def __init__(self, models: List[BaseModel],
                 weights: Optional[List[float]] = None,
                 voting: str = 'soft'):
        """
        初始化集成模型

        Args:
            models: 子模型列表
            weights: 权重列表，None 表示等权重
            voting: 投票方式，'soft' 或 'hard'
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.voting = voting

        # 验证权重
        if len(self.weights) != len(self.models):
            raise ValueError("权重数量必须与模型数量一致")

        # 归一化权重
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        """
        训练所有子模型

        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        Returns:
            训练指标
        """
        results = []
        for i, model in enumerate(self.models):
            result = model.fit(X, y, X_val, y_val)
            results.append({
                'model_index': i,
                'weight': self.weights[i],
                'metrics': result,
            })

        return {'sub_models': results}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        集成预测概率

        Args:
            X: 特征矩阵

        Returns:
            概率矩阵 (n_samples, 2)
        """
        if self.voting == 'soft':
            return self._soft_voting(X)
        else:
            return self._hard_voting(X)

    def _soft_voting(self, X: np.ndarray) -> np.ndarray:
        """
        软投票：加权平均概率

        Args:
            X: 特征矩阵

        Returns:
            概率矩阵
        """
        probs = np.zeros((X.shape[0], 2))

        for model, weight in zip(self.models, self.weights):
            model_probs = model.predict_proba(X)
            probs += weight * model_probs

        return probs

    def _hard_voting(self, X: np.ndarray) -> np.ndarray:
        """
        硬投票：加权多数表决

        Args:
            X: 特征矩阵

        Returns:
            概率矩阵（根据投票结果转换）
        """
        votes = np.zeros((X.shape[0], 2))

        for model, weight in zip(self.models, self.weights):
            pred = model.predict_proba(X).argmax(axis=1)
            for i, p in enumerate(pred):
                votes[i, p] += weight

        # 归一化为概率
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除零
        probs = votes / row_sums

        return probs

    def save(self, path: str) -> None:
        """保存所有子模型"""
        for i, model in enumerate(self.models):
            model.save(f"{path}_model_{i}")

    def load(self, path: str) -> None:
        """加载所有子模型"""
        for i, model in enumerate(self.models):
            model.load(f"{path}_model_{i}")

    def feature_importance(self) -> Dict[str, float]:
        """
        返回集成特征重要性（各模型的加权平均）

        Returns:
            特征重要性字典
        """
        combined = {}

        for model, weight in zip(self.models, self.weights):
            importance = model.feature_importance()
            for feature, value in importance.items():
                if feature not in combined:
                    combined[feature] = 0
                combined[feature] += weight * value

        return combined
