"""
LSTM 实验框架

用于管理和运行多种训练策略的实验框架。
"""

from .experiment_manager import ExperimentManager
from .base_executor import BaseStrategyExecutor
from .registry import ExperimentRegistry, get_registry

__all__ = ["ExperimentManager", "BaseStrategyExecutor", "ExperimentRegistry", "get_registry"]
