# main.py

import polars as pl
import os

print(os.getcwd())

from src.futuapi.futu_api import FutuApi
from src.calculator.mathmatics import add_mean, add_boll

# 设置 Polars 显示的最大行数和列数
# pl.Config.set_tbl_width(1000)  # 设置表格的最大宽度，单位为字符
pl.Config.set_tbl_cols(100)  # 设置显示的最大列数
pl.Config.set_tbl_rows(100)  # 设置显示的最大行数

if __name__ == "__main__":
    api = FutuApi()
    try:
        df = api.get_minute_kline("SH.512400", "2019-01-01", "2024-11-08")
        df = df.with_columns(
            pl.col("time_key").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S")
        )
        df = df.with_columns(
            [
                pl.col("time_key").dt.date().cast(pl.Utf8).alias("date"),
                pl.col("time_key").dt.time().cast(pl.Utf8).alias("time"),
            ]
        )
        df = df.select(["code", "close", "change_rate", "date", "time"])

        close_df = df.filter(pl.col("time") == "15:00:00")
        check_point_df = df.filter(pl.col("time") == "14:55:00")

        df = (
            close_df.join(check_point_df, on=["code", "date"])
            .with_columns(
                [
                    pl.col("close_right").cast(pl.Utf8).alias("check_price"),
                ]
            )
            .select(["code", "date", "check_price", "close"])
        )

        # data = add_mean(data, size=5)
        # data = add_boll(data, N=5, K=2)
        print(df)
    finally:
        api.close()
