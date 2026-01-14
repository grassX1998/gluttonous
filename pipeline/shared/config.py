"""
Pipeline共享配置
优化利用本地硬件资源：GPU、SSD、32GB内存
"""

from pathlib import Path
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

# 训练数据路径
TRAIN_DATA_DIR = PIPELINE_DATA_ROOT / "train"

# 模型检查点路径
MODEL_CHECKPOINT_DIR = PIPELINE_DATA_ROOT / "checkpoints"

# 回测结果路径
BACKTEST_RESULT_DIR = PIPELINE_DATA_ROOT / "backtest_results"

# 创建所有必要的目录
for dir_path in [CLEANED_DATA_DIR, FEATURE_DATA_DIR, TRAIN_DATA_DIR, 
                 MODEL_CHECKPOINT_DIR, BACKTEST_RESULT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


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
