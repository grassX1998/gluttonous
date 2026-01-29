"""
多模型框架配置

包含各模型的超参数配置和共享的交易参数。
"""

from dataclasses import dataclass, field
from typing import List
from pathlib import Path

# ===== 路径配置 =====
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DATA_ROOT = PROJECT_ROOT / "src" / "models" / "data"

# 模型检查点路径
MODEL_CHECKPOINT_DIR = MODELS_DATA_ROOT / "checkpoints"

# 实验结果路径
EXPERIMENT_RESULT_DIR = MODELS_DATA_ROOT / "results"

# 创建所有必要的目录
for dir_path in [MODEL_CHECKPOINT_DIR, EXPERIMENT_RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 从 pipeline 导入数据路径
from pipeline.shared.config import (
    FEATURE_DATA_MONTHLY_DIR,
    FEATURE_DATA_DIR,
    DAILY_DATA_DIR,
)

# ===== 交易配置（与 LSTM 保持一致） =====
TRADING_CONFIG = {
    'top_n': 10,              # 每日持仓数
    'prob_threshold': 0.60,   # 概率阈值
    'holding_days': 5,        # 持有天数
    'commission': 0.001,      # 手续费 0.1%
    'slippage': 0.001,        # 滑点 0.1%
}


@dataclass
class LightGBMConfig:
    """LightGBM 模型配置（v2：中等复杂度 + 增强正则化 + 低阈值）"""

    # 模型参数（v2：平衡复杂度和泛化）
    num_leaves: int = 24          # 中等复杂度
    max_depth: int = 5            # 适中深度
    learning_rate: float = 0.05   # 标准学习率
    n_estimators: int = 150       # 适中迭代次数
    min_child_samples: int = 30   # 中等叶子最小样本数
    subsample: float = 0.6        # 适中采样
    colsample_bytree: float = 0.6 # 适中列采样
    reg_alpha: float = 0.15       # 增加 L1 正则
    reg_lambda: float = 0.15      # 增加 L2 正则

    # 训练参数
    train_days: int = 60          # 训练天数
    val_days: int = 5             # 验证天数
    sample_ratio: float = 0.5     # 采样比例
    retrain_interval: int = 1     # 每天重训练
    random_seed: int = 42

    # 交易参数（v2：降低阈值增加交易机会）
    top_n: int = 10
    prob_threshold: float = 0.55  # 降低阈值
    holding_days: int = 5
    commission: float = 0.001
    slippage: float = 0.001

    @classmethod
    def with_params(cls, **kwargs) -> "LightGBMConfig":
        """创建带自定义参数的配置"""
        return cls(**kwargs)


@dataclass
class MLPConfig:
    """MLP 配置（v2：简化结构 + 强正则化防过拟合）"""

    # 模型参数（v2：简化结构 + 增强正则化）
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 32])  # 恢复简单结构
    dropout: float = 0.6         # 增强 dropout 防过拟合
    batch_norm: bool = True

    # 训练参数（v2：减少训练防过拟合）
    batch_size: int = 1024
    epochs: int = 15             # 减少训练轮数
    learning_rate: float = 0.001 # 恢复标准学习率
    weight_decay: float = 0.02   # 大幅增加 L2 正则
    early_stop_patience: int = 3 # 更早停止防过拟合

    # 数据参数
    train_days: int = 60
    val_days: int = 5
    sample_ratio: float = 0.5
    retrain_interval: int = 1
    random_seed: int = 42

    # 交易参数
    top_n: int = 10
    prob_threshold: float = 0.60
    holding_days: int = 5
    commission: float = 0.001
    slippage: float = 0.001

    @classmethod
    def with_params(cls, **kwargs) -> "MLPConfig":
        """创建带自定义参数的配置"""
        return cls(**kwargs)


@dataclass
class EnsembleConfig:
    """集成模型配置（v3：大幅偏向表现优秀的 MLP）"""

    # 子模型
    models: List[str] = field(default_factory=lambda: ['lightgbm', 'mlp'])

    # 投票方式（v3：大幅偏向 MLP，因为 MLP 表现远优于 LightGBM）
    voting: str = 'soft'          # 'soft' 或 'hard'
    weights: List[float] = field(default_factory=lambda: [0.2, 0.8])  # LightGBM:MLP = 2:8

    # 数据参数（与子模型共享）
    train_days: int = 60
    val_days: int = 5
    sample_ratio: float = 0.5
    retrain_interval: int = 1
    random_seed: int = 42

    # 交易参数（v3：适中阈值平衡交易数量和质量）
    top_n: int = 10
    prob_threshold: float = 0.55  # 适中阈值
    holding_days: int = 5
    commission: float = 0.001
    slippage: float = 0.001

    @classmethod
    def with_params(cls, **kwargs) -> "EnsembleConfig":
        """创建带自定义参数的配置"""
        return cls(**kwargs)
