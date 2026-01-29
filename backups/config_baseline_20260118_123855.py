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
    FEATURE_DATA_DIR as PIPELINE_FEATURE_DATA_DIR,  # 实际特征数据目录
    DAILY_DATA_DIR,
    RAW_DATA_ROOT
)

# 使用 pipeline 的特征数据目录作为实际数据源
FEATURE_DATA_DIR = PIPELINE_FEATURE_DATA_DIR

# ===== 硬件配置 =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()

# ===== 模型配置 =====
MODEL_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'batch_size': 2048,  # 增大 batch_size 以提高 GPU 利用率
    'epochs': 10,
    'learning_rate': 0.001,
    'early_stop_patience': 3,
}

# ===== DataLoader 优化配置 =====
DATALOADER_CONFIG = {
    'num_workers': 4,           # 多进程加载数据
    'pin_memory': True,         # 加速 CPU->GPU 传输
    'persistent_workers': True, # 保持 worker 进程存活
    'prefetch_factor': 2,       # 预取批次数
}

# ===== 训练优化配置 =====
TRAINING_OPTIMIZATION = {
    'use_mixed_precision': torch.cuda.is_available(),  # 混合精度训练
    'finetune_epochs': 3,          # 微调时的 epoch 数
    'finetune_lr_factor': 0.1,     # 微调学习率倍数
    'full_retrain_interval': 20,   # 每 N 次训练进行一次完整训练
}

# ===== 交易配置 =====
TRADING_CONFIG = {
    'top_n': 10,              # 每日持仓数
    'prob_threshold': 0.70,   # 概率阈值（提高到0.70）
    'commission': 0.001,      # 手续费
    'slippage': 0.001,        # 滑点

    # 动态回撤止盈止损
    'trailing_stop_pct': 0.05,    # 动态回撤比例：5%
    'max_holding_days': 10,       # 最长持有天数
    'min_holding_days': 1,        # 最短持有天数（至少持有1天）
    'exit_on_low_prob': False,    # 关闭低概率退出（避免过度交易）
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
    val_days: int = 5                # 验证集天数（增加到5天，约5000样本）
    holding_days: int = 5            # 持有天数（标签滞后天数，避免前瞻偏差）
    use_sample_weight: bool = True
    weight_decay_days: int = 40      # 每20天衰减（加快衰减速度）
    weight_decay_rate: float = 0.50  # 衰减率（增强衰减效果）
    retrain_interval: int = 1

    # 交易参数（可覆盖全局配置）
    prob_threshold: float = 0.70
    top_n: int = 10

    @classmethod
    def with_params(cls, **kwargs) -> "ExpandingWindowConfig":
        """
        创建带自定义参数的配置

        Example:
            config = ExpandingWindowConfig.with_params(
                weight_decay_days=25,
                prob_threshold=0.65
            )
        """
        return cls(**kwargs)


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


@dataclass
class V03ReproConfig(TrainStrategyConfig):
    """v0.3 复现配置 - 月度滚动训练 + 固定5天持有"""
    strategy_name: str = "v03_repro"
    description: str = "复现 v0.3 归档策略：月度滚动训练，固定5天持有"

    # 训练配置（月度滚动）
    train_months: int = 6          # 训练月数
    val_months: int = 1            # 验证月数

    # 采样配置
    sample_ratio: float = 0.5      # 50% 采样
    random_seed: int = 42          # 固定随机种子

    # 交易配置（固定持有期）
    top_n: int = 10
    prob_threshold: float = 0.60   # v0.3 使用 0.60
    holding_days: int = 5          # 固定 5 天持有
    commission: float = 0.001
    slippage: float = 0.001

    @classmethod
    def with_params(cls, **kwargs) -> "V03ReproConfig":
        """
        创建带自定义参数的配置

        Example:
            config = V03ReproConfig.with_params(
                train_months=8,
                prob_threshold=0.65
            )
        """
        return cls(**kwargs)


@dataclass
class V03DailyConfig(TrainStrategyConfig):
    """v0.3 每日重训练配置 - 每天重训练 + 固定5天持有"""
    strategy_name: str = "v03_daily"
    description: str = "基于 v0.3 策略，改为每天重训练"

    # 训练配置（每日重训练）
    # 注意：min_days = train_days + val_days + holding_days 决定最早测试日期
    # 数据从 2024-09-10 开始，到 2025-04-01 约 145 个交易日
    # 设置 train_days=60, val_days=5 使得 min_days=70，可以更早开始测试
    train_days: int = 60           # 训练天数（约3个月）
    val_days: int = 5              # 验证天数

    # 采样配置
    sample_ratio: float = 0.5      # 50% 采样
    random_seed: int = 42          # 固定随机种子

    # 交易配置（固定持有期）
    top_n: int = 10
    prob_threshold: float = 0.60   # 与 v0.3 相同
    holding_days: int = 5          # 固定 5 天持有
    commission: float = 0.001
    slippage: float = 0.001

    @classmethod
    def with_params(cls, **kwargs) -> "V03DailyConfig":
        return cls(**kwargs)


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
    "v03_repro": V03ReproConfig,
    "v03_daily": V03DailyConfig,
}
