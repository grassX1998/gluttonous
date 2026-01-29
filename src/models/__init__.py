"""
多模型量化交易框架

提供多种机器学习模型用于 A 股量化选股：
- LightGBM: 梯度提升树基线模型
- MLP: 简单前馈神经网络
- Ensemble: 多模型投票集成

与 LSTM 框架并行，复用相同的交易参数和指标计算逻辑。
"""

from .config import TRADING_CONFIG, LightGBMConfig, MLPConfig, EnsembleConfig

__all__ = [
    'TRADING_CONFIG',
    'LightGBMConfig',
    'MLPConfig',
    'EnsembleConfig',
]
