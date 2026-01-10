import polars as pl
import os
import toml

gm_dir = f"/mnt/data/stock/gm"


def get_trading_days(start: str, end: str):
    trading_days = toml.load(f"{gm_dir}/cfg/trading_days.toml")["trading_days"]
    return [day for day in trading_days if start <= day <= end]


def get_mkline_by_date(code: str, date: str):
    df_path = f"{gm_dir}/mkline/{code}/{date}.parquet"
    if not os.path.exists(df_path):
        return pl.DataFrame()
    return pl.read_parquet(df_path)


def get_mkline(code: str, start: str, end: str):
    dates = get_trading_days(start, end)
    mklines = []
    for date in dates:
        df = get_mkline_by_date(code, date)
        if not df.is_empty():
            mklines.append(df)
    return pl.concat(mklines, how="vertical")


def get_dkline(code: str, start: str, end=""):
    if end == "":
        end = start
    df = get_mkline(code, start, end)
    df = df.group_by("date").agg(
        [
            pl.col("open").first().alias("open"),  # 每日开盘价
            pl.col("close").last().alias("close"),  # 每日收盘价
            pl.col("high").max().alias("high"),  # 每日最高价
            pl.col("low").min().alias("low"),  # 每日最低价
            pl.col("volume").sum().alias("volume"),  # 每日成交量总和
        ]
    )
    return df.sort("date")


def get_index_list(code: str):
    return toml.load(f"{gm_dir}/meta/index/{code}.toml")["sec_ids"]
