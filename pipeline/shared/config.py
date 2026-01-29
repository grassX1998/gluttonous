"""
Pipeline共享配置
优化利用本地硬件资源：GPU、SSD、32GB内存
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import torch

# ===== 路径配置 =====
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 原始数据路径（NAS网络存储）
RAW_DATA_ROOT = Path(r"\\DXP8800PRO-A577\data\stock\gm")

# Pipeline数据路径（建议放在SSD上）
PIPELINE_DATA_ROOT = PROJECT_ROOT / ".pipeline_data"

# 清洗后的数据路径
CLEANED_DATA_DIR = PIPELINE_DATA_ROOT / "cleaned"

# 特征数据路径
FEATURE_DATA_DIR = PIPELINE_DATA_ROOT / "features"
# 月度组织的特征数据（用于 backtest_v5.py 等）
FEATURE_DATA_MONTHLY_DIR = PIPELINE_DATA_ROOT / "features_monthly"
# 日线数据（用于回测）
DAILY_DATA_DIR = PIPELINE_DATA_ROOT / "daily"

# 训练数据路径
TRAIN_DATA_DIR = PIPELINE_DATA_ROOT / "train"

# 模型检查点路径
MODEL_CHECKPOINT_DIR = PIPELINE_DATA_ROOT / "checkpoints"

# 回测结果路径
BACKTEST_RESULT_DIR = PIPELINE_DATA_ROOT / "backtest_results"

# 日志目录
LOG_DIR = PIPELINE_DATA_ROOT / "logs"

# ===== NAS 原始数据扩展目录（新增数据类型）=====
# Tick 数据目录
TICK_DATA_DIR = RAW_DATA_ROOT / "tick_l1"

# 财务数据目录
FINANCE_DATA_DIR = RAW_DATA_ROOT / "finance"
FINANCE_BALANCE_DIR = FINANCE_DATA_DIR / "balance"      # 资产负债表
FINANCE_INCOME_DIR = FINANCE_DATA_DIR / "income"        # 利润表
FINANCE_CASHFLOW_DIR = FINANCE_DATA_DIR / "cashflow"    # 现金流量表
FINANCE_DERIV_DIR = FINANCE_DATA_DIR / "deriv"          # 财务衍生指标

# 行业分类目录
INDUSTRY_DATA_DIR = RAW_DATA_ROOT / "meta" / "industry"

# 分红复权目录
DIVIDEND_DATA_DIR = RAW_DATA_ROOT / "dividend"
ADJ_FACTOR_DIR = RAW_DATA_ROOT / "adj_factor"

# 股本数据目录
SHARE_DATA_DIR = RAW_DATA_ROOT / "share"

# 指数成分股历史数据目录
INDEX_CONSTITUENTS_HISTORY_DIR = RAW_DATA_ROOT / "meta" / "index_history"

# ===== AKShare 数据目录（独立数据源，免费替代方案）=====
AKSHARE_DATA_ROOT = Path(r"\\DXP8800PRO-A577\data\stock\akshare")
AKSHARE_INDUSTRY_DIR = AKSHARE_DATA_ROOT / "industry"
AKSHARE_FINANCE_DIR = AKSHARE_DATA_ROOT / "finance"
AKSHARE_DIVIDEND_DIR = AKSHARE_DATA_ROOT / "dividend"
AKSHARE_HIST_DIR = AKSHARE_DATA_ROOT / "hist"
AKSHARE_INDEX_DIR = AKSHARE_DATA_ROOT / "index"  # 指数日线数据

# 创建所有必要的目录
for dir_path in [CLEANED_DATA_DIR, FEATURE_DATA_DIR, FEATURE_DATA_MONTHLY_DIR,
                 DAILY_DATA_DIR, TRAIN_DATA_DIR, MODEL_CHECKPOINT_DIR,
                 BACKTEST_RESULT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ===== 掘金 API 配置 =====
JUEJIN_CONFIG = {
    "server": "192.168.31.252:7001",
    "token": "2a34fdeb7eb2d63c6fe3a1c05230b29a236171c5",
}


# ===== 每日任务配置 =====
DAILY_TASK_CONFIG = {
    "run_after": "17:00:00",    # 17:00 后开始采集
    "check_interval": 600,      # 检查间隔（秒），10分钟
    "max_retries": 3,           # 任务失败最大重试次数
    "retry_interval": 1800,     # 重试间隔（秒），30分钟
}


# ===== 扩展数据采集配置 =====
EXTENDED_DATA_CONFIG = {
    # 财务数据更新配置（季报发布后更新）
    "finance_update_months": [1, 4, 8, 10],  # 季报发布月份
    "finance_update_day": 1,  # 每月第几天检查更新

    # 行业分类更新配置
    "industry_update_interval_days": 30,  # 每30天更新一次

    # 分红数据更新配置
    "dividend_lookback_days": 365,  # 初次采集回溯天数

    # 股本数据更新配置
    "share_update_daily": True,  # 每日更新

    # 复权因子更新配置
    "adj_factor_update_daily": True,  # 每日更新

    # Tick 数据采集配置
    "tick_batch_size": 50,  # 每批采集股票数（避免 API 限流）
    "tick_retry_delay": 1.0,  # 重试间隔（秒）

    # 多线程采集配置
    "collection_workers": 12,           # 默认工作线程数
    "api_rate_limit": 50,               # 每秒最大请求数
    "api_burst_limit": 100,             # 突发请求上限
    "progress_log_interval": 100,       # 进度日志间隔（每完成多少任务输出一次）
}


# ===== 邮件通知配置 =====
EMAIL_CONFIG = {
    "enabled": False,                          # 是否启用邮件通知（暂时禁用）
    "smtp_server": "smtp.office365.com",       # SMTP 服务器
    "smtp_port": 587,                          # SMTP 端口（587=TLS, 465=SSL）
    "use_tls": True,                           # 使用 TLS
    "sender_email": "grassX1998@outlook.com",  # 发件人邮箱
    "sender_password": "fxskhnitwxymvtgt",                     # 发件人密码/应用密码（需要手动填写）
    "recipient_email": "grassX1998@outlook.com",  # 收件人邮箱
}


# ===== 硬件配置 =====
# GPU配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = torch.cuda.is_available()  # 使用混合精度训练加速

# 内存配置（32GB总内存）
# Polars数据处理 - 使用流式处理避免内存溢出
MAX_MEMORY_GB = 20  # 为数据处理预留最多20GB
POLARS_STREAMING = True  # 启用Polars流式处理

# DataLoader配置 - 充分利用内存和CPU
NUM_WORKERS = 8  # 数据加载线程数（根据CPU核心数调整）
PIN_MEMORY = True if torch.cuda.is_available() else False  # GPU时使用pin memory加速
PREFETCH_FACTOR = 4  # 每个worker预取的batch数

# 缓存配置
ENABLE_CACHE = True  # 启用磁盘缓存
CACHE_SIZE_GB = 8  # 内存缓存大小


# ===== 数据清洗配置 =====
CLEANING_CONFIG = {
    "batch_size": 500,  # 每批处理的股票数量
    "min_trading_days": 60,  # 最少交易天数
    "max_missing_ratio": 0.1,  # 最大缺失率
    "outlier_std": 5.0,  # 异常值标准差倍数
    "save_format": "parquet",  # 使用parquet格式节省空间和加速读取
}


# ===== 特征工程配置 =====
FEATURE_CONFIG = {
    "lookback_days": 60,  # 回看天数
    "feature_groups": [
        "returns",  # 收益率特征
        "ma",  # 移动平均
        "volatility",  # 波动率
        "rsi",  # RSI指标
        "macd",  # MACD指标
        "bollinger",  # 布林带
        "volume",  # 成交量特征
        "strategy",  # 策略相关特征
        "market",  # 市场状态特征
    ],
    "normalize": True,  # 特征标准化
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}


# ===== 训练配置 =====
TRAIN_CONFIG = {
    # 模型配置
    "model_type": "lstm",  # lstm / transformer / gru
    "hidden_size": 64,     # 更小的模型减少过拟合
    "num_layers": 1,       # 单层
    "dropout": 0.5,        # 更高的dropout
    "num_classes": 2,      # 二分类：涨/跌
    
    # 训练配置
    "batch_size": 256 if torch.cuda.is_available() else 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 1e-3,  # 更强的L2正则化
    "patience": 10,
    
    # 类别权重（处理不平衡）
    "use_class_weight": True,
    
    # 优化器配置
    "optimizer": "adamw",
    "scheduler": "reduce_on_plateau",
    
    # 数据分割
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    
    # GPU优化
    "use_amp": USE_MIXED_PRECISION,  # 自动混合精度
    "gradient_accumulation_steps": 1,  # 梯度累积
    "compile_model": False,  # PyTorch 2.0编译（需要PyTorch>=2.0）
}


# ===== 回测配置 =====
BACKTEST_CONFIG = {
    "initial_cash": 1_000_000,  # 初始资金
    "max_positions": 10,  # 最大持仓数
    "commission_rate": 0.0003,  # 手续费率
    "slippage": 0.001,  # 滑点
    "min_probability": 0.6,  # 最小买入概率
    "stop_loss": -0.08,  # 止损线
    "take_profit": 0.15,  # 止盈线
    "max_holding_days": 20,  # 最大持仓天数
}


# ===== 数据校验配置 =====
VALIDATION_CONFIG = {
    "check_missing": True,
    "check_outliers": True,
    "check_distribution": True,
    "check_label_balance": True,
    "generate_report": True,
}


# ===== 多方案实验框架配置 =====

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
    weight_decay_days: int = 30  # 权重衰减周期
    weight_decay_rate: float = 0.98  # 每个周期的衰减率
    retrain_interval: int = 1  # 每天重训练


@dataclass
class RollingKFoldConfig(TrainStrategyConfig):
    """方案2: 固定滚动窗口 + K折验证"""
    strategy_name: str = "rolling_kfold"
    description: str = "固定60天窗口，时间序列K折交叉验证"
    train_days: int = 60
    val_days: int = 1
    n_folds: int = 3  # 时间序列K折数
    retrain_interval: int = 5  # 每5天重训练


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
    retrain_interval: int = 5  # 检查间隔
    performance_threshold: float = 0.52  # 性能阈值（准确率）
    patience: int = 3  # 连续低于阈值N次才重训练


@dataclass
class IncrementalLearningConfig(TrainStrategyConfig):
    """方案5: 增量学习 - 在线学习，持续微调"""
    strategy_name: str = "incremental_learning"
    description: str = "在线学习，持续微调"
    initial_train_days: int = 60
    val_days: int = 1
    finetune_lr: float = 0.0001  # 微调学习率
    finetune_epochs: int = 3
    forget_rate: float = 0.95  # 遗忘率（类似EWM）
    update_interval: int = 1  # 每天更新


@dataclass
class NoValBayesianConfig(TrainStrategyConfig):
    """方案6: 无验证集 + 贝叶斯优化超参数"""
    strategy_name: str = "no_val_bayesian"
    description: str = "无显式验证集，依赖正则化和预优化超参数"
    train_days: int = 60
    val_days: int = 0  # 无验证集
    use_bayesian_opt: bool = True
    bayesian_n_trials: int = 20  # 贝叶斯优化试验次数
    retrain_interval: int = 5
    # 更强的正则化
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
