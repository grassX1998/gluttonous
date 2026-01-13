"""
特征工程 - 技术指标计算

从分钟K线数据计算各种技术指标特征
"""

import polars as pl
import numpy as np


def calc_returns(df: pl.DataFrame, col: str = "close", periods: list[int] = [1, 5, 10, 20]) -> pl.DataFrame:
    """计算收益率"""
    exprs = []
    for p in periods:
        exprs.append(
            (pl.col(col) / pl.col(col).shift(p) - 1).alias(f"ret_{p}")
        )
    return df.with_columns(exprs)


def calc_ma(df: pl.DataFrame, col: str = "close", windows: list[int] = [5, 10, 20, 60]) -> pl.DataFrame:
    """计算移动平均线"""
    exprs = []
    for w in windows:
        exprs.append(
            pl.col(col).rolling_mean(window_size=w).alias(f"ma_{w}")
        )
    # 计算价格相对MA的位置
    for w in windows:
        exprs.append(
            (pl.col(col) / pl.col(col).rolling_mean(window_size=w) - 1).alias(f"ma_{w}_ratio")
        )
    return df.with_columns(exprs)


def calc_volatility(df: pl.DataFrame, windows: list[int] = [5, 10, 20]) -> pl.DataFrame:
    """计算波动率"""
    # 先计算收益率
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("_ret")
    )
    
    exprs = []
    for w in windows:
        exprs.append(
            pl.col("_ret").rolling_std(window_size=w).alias(f"volatility_{w}")
        )
    
    return df.with_columns(exprs).drop("_ret")


def calc_rsi(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
    """计算RSI指标"""
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("_change")
    )
    
    df = df.with_columns([
        pl.when(pl.col("_change") > 0).then(pl.col("_change")).otherwise(0).alias("_gain"),
        pl.when(pl.col("_change") < 0).then(-pl.col("_change")).otherwise(0).alias("_loss"),
    ])
    
    df = df.with_columns([
        pl.col("_gain").rolling_mean(window_size=window).alias("_avg_gain"),
        pl.col("_loss").rolling_mean(window_size=window).alias("_avg_loss"),
    ])
    
    df = df.with_columns(
        (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10))).alias(f"rsi_{window}")
    )
    
    return df.drop(["_change", "_gain", "_loss", "_avg_gain", "_avg_loss"])


def calc_macd(df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
    """计算MACD指标"""
    df = df.with_columns([
        pl.col("close").ewm_mean(span=fast).alias("_ema_fast"),
        pl.col("close").ewm_mean(span=slow).alias("_ema_slow"),
    ])
    
    df = df.with_columns(
        (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_dif")
    )
    
    df = df.with_columns(
        pl.col("macd_dif").ewm_mean(span=signal).alias("macd_dea")
    )
    
    df = df.with_columns(
        (2 * (pl.col("macd_dif") - pl.col("macd_dea"))).alias("macd_hist")
    )
    
    return df.drop(["_ema_fast", "_ema_slow"])


def calc_bollinger(df: pl.DataFrame, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
    """计算布林带"""
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=window).alias("bb_mid"),
        pl.col("close").rolling_std(window_size=window).alias("_bb_std"),
    ])
    
    df = df.with_columns([
        (pl.col("bb_mid") + num_std * pl.col("_bb_std")).alias("bb_upper"),
        (pl.col("bb_mid") - num_std * pl.col("_bb_std")).alias("bb_lower"),
    ])
    
    # 计算价格在布林带中的位置 (0-1)
    df = df.with_columns(
        ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-10)).alias("bb_position")
    )
    
    return df.drop("_bb_std")


def calc_volume_features(df: pl.DataFrame, windows: list[int] = [5, 10, 20]) -> pl.DataFrame:
    """计算量价特征"""
    exprs = []
    
    # 成交量MA
    for w in windows:
        exprs.append(
            pl.col("volume").rolling_mean(window_size=w).alias(f"vol_ma_{w}")
        )
    
    # 量比
    for w in windows:
        exprs.append(
            (pl.col("volume") / (pl.col("volume").rolling_mean(window_size=w) + 1)).alias(f"vol_ratio_{w}")
        )
    
    return df.with_columns(exprs)


def calc_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """计算价格特征"""
    return df.with_columns([
        # 振幅
        ((pl.col("high") - pl.col("low")) / (pl.col("close").shift(1) + 1e-10)).alias("amplitude"),
        # 上影线
        ((pl.col("high") - pl.col("close").clip(lower_bound=pl.col("open"))) / (pl.col("high") - pl.col("low") + 1e-10)).alias("upper_shadow"),
        # 下影线
        ((pl.col("close").clip(upper_bound=pl.col("open")) - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-10)).alias("lower_shadow"),
        # 实体
        ((pl.col("close") - pl.col("open")) / (pl.col("open") + 1e-10)).alias("body"),
    ])


def calc_all_features(df: pl.DataFrame) -> pl.DataFrame:
    """计算所有特征"""
    df = calc_returns(df)
    df = calc_ma(df)
    df = calc_volatility(df)
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_bollinger(df)
    df = calc_volume_features(df)
    df = calc_price_features(df)
    return df


# 特征列表（用于模型输入）
FEATURE_COLS = [
    # 收益率
    "ret_1", "ret_5", "ret_10", "ret_20",
    # MA比率
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_60_ratio",
    # 波动率
    "volatility_5", "volatility_10", "volatility_20",
    # 技术指标
    "rsi_14", "macd_dif", "macd_dea", "macd_hist", "bb_position",
    # 量价特征
    "vol_ratio_5", "vol_ratio_10", "vol_ratio_20",
    # 价格特征
    "amplitude", "upper_shadow", "lower_shadow", "body",
]
