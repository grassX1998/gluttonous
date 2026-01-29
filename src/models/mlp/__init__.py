"""
MLP 模块

提供简单的前馈神经网络用于量化选股。
"""

from .model import MLPModel, SimpleMLP
from .executor import MLPExecutor

__all__ = ['MLPModel', 'SimpleMLP', 'MLPExecutor']
