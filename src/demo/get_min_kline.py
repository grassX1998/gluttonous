# main.py

import polars as pl

from src.futuapi.futu_api import FutuApi
from src.calculator.mathmatics import add_mean, add_boll

# 设置 Polars 显示的最大行数和列数
# pl.Config.set_tbl_width(1000)  # 设置表格的最大宽度，单位为字符
pl.Config.set_tbl_cols(100)  # 设置显示的最大列数
pl.Config.set_tbl_rows(100)  # 设置显示的最大行数

if __name__ == "__main__":
    api = FutuApi()
    try:
        data = api.get_minute_kline("SH.000001", "2022-01-01", "2023-01-31")
        data = add_mean(data, size=5)
        data = add_boll(data, N=5, K=2)
        print(data)
    finally:
        api.close()
