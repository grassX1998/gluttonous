"""
LightGBM 模块

提供 LightGBM 梯度提升树模型用于量化选股。
"""

from .model import LightGBMModel
from .executor import LightGBMExecutor

__all__ = ['LightGBMModel', 'LightGBMExecutor']
