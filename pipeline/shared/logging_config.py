"""
统一日志配置模块

所有流程日志按日期组织：
- .pipeline_data/logs/{date}/data_collection.log
- .pipeline_data/logs/{date}/data_cleaning.log
- .pipeline_data/logs/{date}/features.log
- .pipeline_data/logs/{date}/training.log
- .pipeline_data/logs/{date}/backtest.log
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT

# 日志根目录（项目根目录下的 logs 文件夹）
LOG_ROOT = PROJECT_ROOT / "logs"


class FlushFileHandler(logging.FileHandler):
    """立即刷新的文件处理器"""

    def emit(self, record):
        super().emit(record)
        self.flush()


def get_logger(
    name: str,
    log_type: str = "general",
    date: Optional[str] = None,
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """
    获取统一配置的 Logger

    Args:
        name: Logger 名称（通常是模块名）
        log_type: 日志类型（data_collection/data_cleaning/features/training/backtest）
        date: 日期，默认为当天
        console: 是否输出到控制台
        file: 是否输出到文件

    Returns:
        配置好的 Logger
    """
    # 强制 UTF-8 编码
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # 确定日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 创建日志目录
    log_dir = LOG_ROOT / date
    log_dir.mkdir(parents=True, exist_ok=True)

    # 获取或创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除已有的 handler（避免重复添加）
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件输出（使用立即刷新的处理器）
    if file:
        log_file = log_dir / f"{log_type}.log"
        file_handler = FlushFileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_collection_logger(date: Optional[str] = None) -> logging.Logger:
    """获取数据采集日志器"""
    return get_logger("data_collection", "data_collection", date)


def get_cleaning_logger(date: Optional[str] = None) -> logging.Logger:
    """获取数据清洗日志器"""
    return get_logger("data_cleaning", "data_cleaning", date)


def get_features_logger(date: Optional[str] = None) -> logging.Logger:
    """获取特征工程日志器"""
    return get_logger("features", "features", date)


def get_training_logger(date: Optional[str] = None) -> logging.Logger:
    """获取模型训练日志器"""
    return get_logger("training", "training", date)


def get_backtest_logger(date: Optional[str] = None) -> logging.Logger:
    """获取回测日志器"""
    return get_logger("backtest", "backtest", date)


def get_lstm_training_logger(date: Optional[str] = None) -> logging.Logger:
    """获取 LSTM 训练日志器"""
    return get_logger("lstm_training", "lstm_training", date)


def get_lstm_backtest_logger(date: Optional[str] = None) -> logging.Logger:
    """获取 LSTM 回测日志器"""
    return get_logger("lstm_backtest", "lstm_backtest", date)


def get_akshare_logger(date: Optional[str] = None) -> logging.Logger:
    """获取 AKShare 采集日志器"""
    return get_logger("akshare_collection", "akshare_collection", date)


def get_experiment_logger(experiment_id: str, date: Optional[str] = None) -> logging.Logger:
    """
    获取实验专用日志器

    日志输出到 logs/{date}/experiments/{experiment_id}.log

    Args:
        experiment_id: 实验 ID
        date: 日期，默认为当天

    Returns:
        配置好的 Logger
    """
    import sys

    # 强制 UTF-8 编码
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # 确定日期
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # 创建实验日志目录
    exp_log_dir = LOG_ROOT / date / "experiments"
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    # 获取或创建 Logger
    logger_name = f"experiment_{experiment_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 清除已有的 handler
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    log_file = exp_log_dir / f"{experiment_id}.log"
    file_handler = FlushFileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
