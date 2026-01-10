import polars as pl
import os
import time
import datetime
from utils import qq_bot

from dataclasses import dataclass
from src.futuapi.futu_api import FutuApi
from src.calculator.mathmatics import add_mean, add_boll

# 设置 Polars 显示的最大行数和列数
# pl.Config.set_tbl_width(1000)  # 设置表格的最大宽度，单位为字符
pl.Config.set_tbl_cols(100)  # 设置显示的最大列数
pl.Config.set_tbl_rows(10)  # 设置显示的最大行数


def parse_df_time(df):
    df = df.with_columns(
        pl.col("time_key").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
        pl.col("close").cast(pl.Float64).alias("price"),
    )
    df = df.with_columns(
        [
            pl.col("time_key").dt.date().cast(pl.Utf8).alias("date"),
            pl.col("time_key").dt.time().cast(pl.Utf8).alias("time"),
        ]
    )
    return df


def trans(fund_df: pl.DataFrame, index_df: pl.DataFrame):
    fund_df = (
        parse_df_time(fund_df)
        .filter(pl.col("time") == "14:50:00")
        .select(["date", "price"])
    )

    index_df = parse_df_time(index_df)
    df_index_vol = (
        index_df.group_by("date")
        .agg([pl.col("volume").sum().alias("volume")])
        .sort("date")
    )
    df_index_check_vol = (
        index_df.filter(pl.col("time") <= "14:50:00")
        .group_by("date")
        .agg([pl.col("volume").sum().alias("check_volume")])
        .sort("date")
    )
    df_index_check_price = (
        index_df.filter(pl.col("time") == "14:50:00")
        .with_columns(pl.col("price").cast(pl.Float64).alias("check_index_price"))
        .select("date", "check_index_price")
    )

    df = (
        index_df.filter(pl.col("time") == "15:00:00")
        .with_columns(pl.col("price").cast(pl.Float64).alias("index_price"))
        .select("date", "index_price")
        .join(df_index_check_vol, on=["date"])
        .join(df_index_vol, on=["date"])
        .join(df_index_check_price, on=["date"])
        .join(fund_df, on=["date"])
    ).select(
        ["date", "check_volume", "volume", "check_index_price", "index_price", "price"]
    )
    df = df.sort("date")
    df = df.with_columns(
        pl.col("index_price").rolling_max(window_size=25).alias("max_25")
    )
    df = df.with_columns(
        pl.col("check_volume")
        .rolling_mean(window_size=10)
        .cast(pl.Int64)
        .alias("avg_vol_10")
    )
    return df


@dataclass
class SecRecord:
    date: str
    code: str
    price: float
    check_index_price: float
    index_price: float
    check_volume: int
    volume: int
    max_25: float
    avg_vol_10: int


class SecSignal:
    def __init__(self):
        self.on_position = False
        self.price = 0.0
        self.rate = 0.0
        self.max_price = 0.0
        self.max_25 = None
        self.avg_vol_10 = None

    def load(self, record: SecRecord):
        if record.max_25 == None:
            return "None"
        elif self.max_25 == None:
            self.max_25 = record.max_25
            self.avg_vol_10 = record.avg_vol_10
            return "None"

        result = "None"
        if self.on_position:
            if (
                record.check_index_price <= self.max_25 * 0.95
                or record.check_volume < self.avg_vol_10
            ):
                self.rate = record.price * 100 / self.price - 100
                self.on_position = False
                self.max_price = 0
                result = "Sale"
        else:
            if (
                record.check_index_price >= self.max_25
                and record.check_volume > self.avg_vol_10 * 1.2
            ):
                self.price = record.price
                self.on_position = True
                self.max_price = record.price
                result = "Buy"

        self.max_25 = record.max_25
        self.avg_vol_10 = record.avg_vol_10
        self.max_price = max(self.max_price, record.price)
        return result

    def check(self, record: SecRecord):
        result = "None"
        print(record, self.avg_vol_10, self.max_25)
        if self.on_position:
            if (
                record.check_index_price <= self.max_25 * 0.95
                or record.check_volume < self.avg_vol_10
            ):
                result = "Sale"
        else:
            if (
                record.check_index_price >= self.max_25
                and record.check_volume > self.avg_vol_10 * 1.2
            ):
                result = "Buy"
        return result


class StrFundMax1:
    def __init__(self):
        self.sub_info = [
            {"code": "SH.512880", "index": "SZ.399975"},
            {"code": "SH.510300", "index": "SH.000300"},
            {"code": "SH.512100", "index": "SH.000852"},
        ]
        self.sub_list = [
            *[ele["code"] for ele in self.sub_info],
            *[ele["index"] for ele in self.sub_info],
        ]
        self.date = datetime.datetime.today().strftime("%Y-%m-%d")
        self.signals = dict([(ele["code"], SecSignal()) for ele in self.sub_info])

    def load(self):
        raw_data = {}

        # load raw data
        api = FutuApi()
        try:
            start_date = "2024-01-01"
            end_date = api.get_trading_days(start_date, self.date)[-2]
            for sub_dict in self.sub_info:
                fund_df = api.get_minute_kline(sub_dict["code"], start_date, end_date)
                index_df = api.get_minute_kline(sub_dict["index"], start_date, end_date)
                df = trans(fund_df, index_df)
                # print(df)
                for row in df.iter_rows(named=True):
                    date = row["date"]
                    if date not in raw_data:
                        raw_data[date] = []
                    record = SecRecord(
                        code=sub_dict["code"],
                        date=row["date"],
                        price=row["price"],
                        check_index_price=row["check_index_price"],
                        index_price=row["index_price"],
                        check_volume=row["check_volume"],
                        volume=row["volume"],
                        max_25=row["max_25"],
                        avg_vol_10=row["avg_vol_10"],
                    )
                    raw_data[date].append(record)
        finally:
            api.close()
        for date, records in raw_data.items():
            buy_list = []
            sale_list = []
            for record in records:
                sig = self.signals[record.code].load(record)
                if sig == "Sale":
                    sale_list.append([record, self.signals[record.code].rate])
                elif sig == "Buy":
                    buy_list.append([record])
            # print(
            #     date,
            #     "sale: ",
            #     [[ele[0].code, ele[1]] for ele in sale_list],
            #     "buy: ",
            #     [[ele[0].code] for ele in buy_list],
            # )

        self.index2fund = {
            "SZ.399975": "SH.512880",
            "SH.000300": "SH.510300",
            "SH.000852": "SH.512100",
        }
        self.check_records = []

    def handle_records(self):
        buy_list = []
        sale_list = []
        for record in self.check_records:
            sig = self.signals[record.code].check(record)
            if sig == "Sale":
                sale_list.append([record, self.signals[record.code].rate])
            elif sig == "Buy":
                buy_list.append([record])

        message = f"[fund_on_recv] {self.date}, sale: {[[ele[0].code, ele[1]] for ele in sale_list]}, buy: {[[ele[0].code] for ele in buy_list]}"
        qq_bot.push_message(message)
        # print(message)
        # print(
        #     self.date,
        #     "sale: ",
        #     [[ele[0].code, ele[1]] for ele in sale_list],
        #     "buy: ",
        #     [[ele[0].code] for ele in buy_list],
        # )

    def check(self, code: str, price: float, vol: int, data_time: str):
        if len(self.index2fund) == 0:
            return
        # print(data_time)
        if data_time >= "14:50:00.000" and code in self.index2fund:
            self.check_records.append(
                SecRecord(
                    code=self.index2fund[code],
                    date=self.date,
                    price=0,
                    check_index_price=price,
                    index_price=0,
                    check_volume=vol,
                    volume=0,
                    max_25=0,
                    avg_vol_10=0,
                )
            )
            del self.index2fund[code]

        if len(self.index2fund) == 0:
            self.handle_records()
