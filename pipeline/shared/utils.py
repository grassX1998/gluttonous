"""
Pipeline共享工具函数
"""

import sys
import time
from pathlib import Path
from functools import wraps
from typing import Any, Callable
import logging

import polars as pl
import torch


# ===== 日志配置 =====
def setup_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 控制台处理器 - 输出到 stdout 避免 PowerShell 的 stderr 问题
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ===== 性能监控装饰器 =====
def timer(func: Callable) -> Callable:
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[Timer] {func.__name__} took {end - start:.2f}s")
        return result
    return wrapper


def memory_monitor(func: Callable) -> Callable:
    """内存监控装饰器（GPU）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / 1024**2
        
        result = func(*args, **kwargs)
        
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated() / 1024**2
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[Memory] {func.__name__}: "
                  f"Start={start_mem:.1f}MB, End={end_mem:.1f}MB, "
                  f"Peak={peak_mem:.1f}MB")
        
        return result
    return wrapper


# ===== 数据加载工具 =====
def load_parquet_lazy(path: Path, **kwargs) -> pl.LazyFrame:
    """延迟加载Parquet文件（节省内存）"""
    return pl.scan_parquet(path, **kwargs)


def load_parquet_streaming(path: Path, batch_size: int = 10000) -> pl.DataFrame:
    """流式加载Parquet文件"""
    return pl.read_parquet(path, use_pyarrow=True)


def save_parquet_optimized(df: pl.DataFrame, path: Path, **kwargs):
    """优化保存Parquet文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        path,
        compression="zstd",  # 使用zstd压缩（平衡压缩率和速度）
        compression_level=3,
        **kwargs
    )


# ===== GPU工具 =====
def get_gpu_info() -> dict[str, Any]:
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "cuda_version": torch.version.cuda,
    }


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===== 数据验证工具 =====
def check_dataframe_quality(df: pl.DataFrame, name: str = "DataFrame") -> dict:
    """检查DataFrame质量"""
    report = {
        "name": name,
        "shape": df.shape,
        "columns": df.columns,
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "missing": {},
        "duplicates": 0,
    }
    
    # 检查缺失值
    for col in df.columns:
        null_count = df[col].null_count()
        if null_count > 0:
            report["missing"][col] = {
                "count": null_count,
                "ratio": null_count / df.height
            }
    
    # 检查重复行
    report["duplicates"] = df.height - df.unique().height
    
    return report


# ===== 文件管理工具 =====
def get_file_size(path: Path) -> float:
    """获取文件大小（MB）"""
    if path.is_file():
        return path.stat().st_size / 1024**2
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
    return 0.0


def clean_old_files(directory: Path, keep_latest: int = 5):
    """清理旧文件，保留最新的N个"""
    if not directory.exists():
        return

    files = sorted(directory.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
    for file in files[keep_latest:]:
        if file.is_file():
            file.unlink()
            print(f"Deleted old file: {file.name}")


# ===== 指数成分股工具 =====
def get_index_constituents_by_date(index_dir: Path, index_code: str, date: str) -> list[str]:
    """
    读取指定日期的指数成分股

    Args:
        index_dir: 指数数据目录 (e.g., RAW_DATA_ROOT / "meta" / "index")
        index_code: 指数代码 (e.g., "SZSE.000852" for 中证1000)
        date: 日期字符串 (e.g., "2025-04-01")

    Returns:
        成分股代码列表
    """
    import tomllib

    # 查找该日期的成分股文件
    date_path = index_dir / date / f"{index_code}.toml"

    if date_path.exists():
        with open(date_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("sec_ids", [])

    # 如果当天没有数据，向前查找最近的一个交易日
    all_dates = sorted([d.name for d in index_dir.iterdir() if d.is_dir()])

    # 找到小于等于目标日期的最近日期
    valid_dates = [d for d in all_dates if d <= date]
    if not valid_dates:
        return []

    nearest_date = valid_dates[-1]
    nearest_path = index_dir / nearest_date / f"{index_code}.toml"

    if nearest_path.exists():
        with open(nearest_path, "rb") as f:
            data = tomllib.load(f)
            return data.get("sec_ids", [])

    return []


def get_all_historical_constituents(index_dir: Path, index_codes: list[str],
                                    start_date: str, end_date: str) -> dict[str, set[str]]:
    """
    获取指定时间段内所有历史成分股

    Args:
        index_dir: 指数数据目录
        index_codes: 指数代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        字典，key为日期，value为该日期的成分股集合
    """
    import tomllib

    # 获取所有可用日期
    all_dates = sorted([d.name for d in index_dir.iterdir() if d.is_dir()])

    # 过滤日期范围
    valid_dates = [d for d in all_dates if start_date <= d <= end_date]

    historical_constituents = {}
    all_symbols = set()

    for date in valid_dates:
        date_symbols = set()
        for index_code in index_codes:
            toml_path = index_dir / date / f"{index_code}.toml"
            if toml_path.exists():
                try:
                    with open(toml_path, "rb") as f:
                        data = tomllib.load(f)
                        symbols = data.get("sec_ids", [])
                        date_symbols.update(symbols)
                        all_symbols.update(symbols)
                except Exception:
                    continue

        if date_symbols:
            historical_constituents[date] = date_symbols

    return historical_constituents
