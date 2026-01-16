"""
LSTM 模型训练配置

所有 LSTM 训练相关的配置都在这里定义。
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import torch

# ===== 路径配置 =====
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# LSTM 数据根目录
LSTM_DATA_ROOT = PROJECT_ROOT / "src" / "lstm" / "data"

# 特征数据路径
FEATURE_DATA_DIR = LSTM_DATA_ROOT / "features"

# 模型检查点路径
MODEL_CHECKPOINT_DIR = LSTM_DATA_ROOT / "checkpoints"

# 实验结果路径
EXPERIMENT_RESULT_DIR = LSTM_DATA_ROOT / "results"

# 创建所有必要的目录
for dir_path in [FEATURE_DATA_DIR, MODEL_CHECKPOINT_DIR, EXPERIMENT_RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 从原始 pipeline 导入数据路径（用于读取原始数据）
from pipeline.shared.config import (
    FEATURE_DATA_MONTHLY_DIR,
    DAILY_DATA_DIR,
    RAW_DATA_ROOT
)

# ===== 硬件配置 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()

# ===== 模型配置 =====
MODEL_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 1024,
    'epochs': 10,
    'learning_rate': 0.001,
    'early_stop_patience': 3,
}

# ===== 交易配置 =====
TRADING_CONFIG = {
    'top_n': 10,              # 每日持仓数
    'prob_threshold': 0.60,   # 概率阈值
    'holding_days': 5,        # 持有天数
    'commission': 0.001,      # 手续费
    'slippage': 0.001,        # 滑点
}


# ===== 训练策略配置 =====

@dataclass
class TrainStrategyConfig:
    """训练策略配置基类"""
    strategy_name: str
    description: str


@dataclass
class ExpandingWindowConfig(TrainStrategyConfig):
    """方案1: 扩展窗口 - 累积历史数据，样本加权"""
    strategy_name: str = "expanding_window"
    description: str = "累积扩展训练集，保留所有历史数据"
    min_train_days: int = 60
    max_train_days: int = 500
    val_days: int = 1
    use_sample_weight: bool = True
    weight_decay_days: int = 30
    weight_decay_rate: float = 0.98
    retrain_interval: int = 1


@dataclass
class RollingKFoldConfig(TrainStrategyConfig):
    """方案2: 固定滚动窗口 + K折验证"""
    strategy_name: str = "rolling_kfold"
    description: str = "固定60天窗口，时间序列K折交叉验证"
    train_days: int = 60
    val_days: int = 1
    n_folds: int = 3
    retrain_interval: int = 5


@dataclass
class MultiScaleEnsembleConfig(TrainStrategyConfig):
    """方案3: 多尺度集成 - 短中长期模型组合"""
    strategy_name: str = "multiscale_ensemble"
    description: str = "短中长期模型集成"
    windows: List[int] = field(default_factory=lambda: [20, 60, 120])
    val_days: int = 1
    ensemble_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    retrain_interval: int = 5


@dataclass
class AdaptiveRetrainConfig(TrainStrategyConfig):
    """方案4: 自适应重训练 - 性能监控，按需训练"""
    strategy_name: str = "adaptive_retrain"
    description: str = "性能监控，按需重训练"
    train_days: int = 60
    val_days: int = 1
    retrain_interval: int = 5
    performance_threshold: float = 0.52
    patience: int = 3


@dataclass
class IncrementalLearningConfig(TrainStrategyConfig):
    """方案5: 增量学习 - 在线学习，持续微调"""
    strategy_name: str = "incremental_learning"
    description: str = "在线学习，持续微调"
    initial_train_days: int = 60
    val_days: int = 1
    finetune_lr: float = 0.0001
    finetune_epochs: int = 3
    forget_rate: float = 0.95
    update_interval: int = 1


@dataclass
class NoValBayesianConfig(TrainStrategyConfig):
    """方案6: 无验证集 + 贝叶斯优化超参数"""
    strategy_name: str = "no_val_bayesian"
    description: str = "无显式验证集，依赖正则化和预优化超参数"
    train_days: int = 60
    val_days: int = 0
    use_bayesian_opt: bool = True
    bayesian_n_trials: int = 20
    retrain_interval: int = 5
    dropout: float = 0.5
    weight_decay: float = 0.01


# 默认策略配置
DEFAULT_STRATEGY_CONFIG = ExpandingWindowConfig()


# 所有可用策略
ALL_STRATEGY_CONFIGS = {
    "expanding_window": ExpandingWindowConfig,
    "rolling_kfold": RollingKFoldConfig,
    "multiscale_ensemble": MultiScaleEnsembleConfig,
    "adaptive_retrain": AdaptiveRetrainConfig,
    "incremental_learning": IncrementalLearningConfig,
    "no_val_bayesian": NoValBayesianConfig,
}
