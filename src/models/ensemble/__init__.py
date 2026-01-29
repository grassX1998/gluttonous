"""
集成模块

提供多模型投票集成用于量化选股。
"""

from .voting import VotingEnsemble
from .executor import EnsembleExecutor

__all__ = ['VotingEnsemble', 'EnsembleExecutor']
