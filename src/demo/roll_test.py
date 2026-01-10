# main.py

import polars as pl
import os

print(os.getcwd())

from src.futuapi.futu_api import FutuApi
from src.calculator.mathmatics import add_mean, add_boll

# 设置 Polars 显示的最大行数和列数
# pl.Config.set_tbl_width(1000)  # 设置表格的最大宽度，单位为字符
pl.Config.set_tbl_cols(100)  # 设置显示的最大列数
pl.Config.set_tbl_rows(10)  # 设置显示的最大行数


def trans(df):
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
                pl.col("close_right").cast(pl.Float64).alias("check_price"),
            ]
        )
        .select(["code", "date", "check_price", "close"])
    )

    df = df.with_columns(
        [
            (
                (pl.col("check_price") - pl.col("close").shift(5))
                / pl.col("close").shift(5)
            ).alias("del_5")
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col("check_price") - pl.col("close").shift(10))
                / pl.col("close").shift(5)
            ).alias("del_10")
        ]
    )

    return df


import time
from dataclasses import dataclass


@dataclass
class Record:
    code: str
    price: float
    del_5: float
    del_10: float


if __name__ == "__main__":
    api = FutuApi()
    try:
        secs = [
            "SH.512660",
            "SH.515880",
            "SZ.159996",
            "SH.515790",
            "SZ.159995",
            "SH.515700",
            "SZ.159647",
            "SH.512290",
            "SH.512170",
            "SH.515220",
            "SH.515210",
            "SH.512400",
            "SZ.159825",
            "SH.512800",
            "SH.516770",
            "SH.516950",
            "SZ.159745",
            "SH.512880",
            "SH.512200",
            "SH.512690",
        ]

        info = {}

        for sec_id in secs:
            df = api.get_minute_kline(sec_id, "2024-01-01", "2024-11-08")
            df = trans(df)
            for row in df.iter_rows(named=True):
                # print(row)
                if row["date"] not in info:
                    info[row["date"]] = []
                record = Record(
                    code=row["code"],
                    price=row["check_price"],
                    del_5=row["del_5"],
                    del_10=row["del_10"],
                )
                info[row["date"]].append(record)

        # print(info["2024-11-08"])

        for key, value in info.items():
            value = [x for x in value if x.del_10 is not None]
            sorted_value = sorted(value, key=lambda x: x.del_10, reverse=True)
            print(key)
            for ele in sorted_value:
                print(ele.code, ele.price, ele.del_10)
            time.sleep(1)
    finally:
        api.close()
