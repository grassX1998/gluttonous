"""
持续优化框架

包含多方案并行训练、参数自动调优、仓位管理优化等功能。
"""

from .param_optimizer import ParamOptimizer, load_best_params

__all__ = [
    'ParamOptimizer',
    'load_best_params',
]
