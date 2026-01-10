import polars as pl


def add_mean(df: pl.DataFrame, size: int) -> pl.DataFrame:
    return df.with_columns(
        pl.col("close").rolling_mean(window_size=size).alias(f"{size}_MA")
    )


def add_boll(df: pl.DataFrame, N: int, K: int) -> pl.DataFrame:
    # 计算布林带
    df = df.with_columns(
        [
            # 计算中线（移动平均线）
            pl.col("close").rolling_mean(window_size=N).alias("Middle Band"),
            # 计算标准差
            pl.col("close").rolling_std(window_size=N).alias("stddev"),
        ]
    )

    # 计算上轨和下轨
    df = df.with_columns(
        [
            (pl.col("Middle Band") + K * pl.col("stddev")).alias("Upper Band"),
            (pl.col("Middle Band") - K * pl.col("stddev")).alias("Lower Band"),
        ]
    )
    return df
