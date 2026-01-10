import polars as pl
import os
import toml
import time
import pytz

from futu import *
from utils.gm import *
from utils import qq_bot
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from dateutil.relativedelta import relativedelta

pl.Config.set_tbl_cols(100)  # 设置显示的最大列数
pl.Config.set_tbl_rows(100)  # 设置显示的最大行数

# sub_list = ["SHSE.603530", "SZSE.002351", "SZSE.300319", "SZSE.301308", "SZSE.301371"]
sub_list = get_index_list("SHSE.000852")


quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)


@dataclass
class KlineRecord:
    code: str
    date: str
    time: str
    open: float
    close: float
    volume: float
    high: float
    low: float


class SecSignal:
    def __init__(self, code: str):
        self.code = code
        self.day_10ma = 0.0
        self.day_20ma = 0.0
        self.day_2max = 0.0
        self.day_2delta = 0.0
        self.day_open = -1.0
        self.last_date = ""
        self.pre_day_close = 0.0
        self.minmap = {}
        self.buy_flag = False
        self.date = ""

    def shouldFocus(self) -> bool:
        # print(self.code, self.day_2delta)
        if (
            self.day_10ma <= self.day_20ma
            or self.day_2delta > 5.0
            or self.day_2delta < -10.0
            or self.pre_day_close < self.day_10ma
        ):
            return False
        return True

    def check(self, record: KlineRecord) -> bool:
        if self.buy_flag == True:
            return False
        if self.day_open < 0 and record.time > "09:30:00":
            self.load_day_open(record.code)
            print(f"{record.code} open {self.day_open}")
        if self.day_open > self.day_2max:
            return False

        if record.time < "09:35:00" or record.time > "10:00:00":
            return False
        if (record.close - record.open) * 100 / record.open < 0.5:
            return False
        if (
            record.time not in self.minmap
            or self.minmap[record.time]["volume_avg_10"] == 0
        ):
            return False
        vol_rate = record.volume / self.minmap[record.time]["volume_avg_10"]
        if vol_rate <= 10:
            return False

        self.buy_flag = True

        message = f"[stock_on_recv] buy: {record.code}, vol_rate: {vol_rate}, day_open: {self.day_open}, cur: {record.close}"
        qq_bot.push_message(message)
        return True

    def load_day_open(self, code):
        ret, data = quote_ctx.get_cur_kline(
            f"{code[0:2]}{code[4:]}", 240, KLType.K_1M, AuType.NONE
        )
        if ret == RET_OK:
            df = pl.DataFrame(data).filter(
                pl.col("time_key") >= f"{self.date} 09:30:00"
            )
            self.day_open = df["open"].to_list()[0]
        else:
            print("error:", data)

        data = pl.DataFrame(data)[
            "code",
            "time_key",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover",
            "last_close",
        ].to_dicts()
        for sec_info in data:
            sec_id = f'{sec_info["code"][0:2]}SE{sec_info["code"][2:]}'
            sec_date = sec_info["time_key"][0:10]
            sec_time = sec_info["time_key"][11:19]
            record = KlineRecord(
                code=sec_id,
                date=sec_date,
                time=sec_time,
                open=sec_info["open"],
                close=sec_info["close"],
                volume=sec_info["volume"],
                high=sec_info["high"],
                low=sec_info["low"],
            )
            print("test", record.code, record.time)
            if self.check(record):
                print("buy", record.code, record.time)
                qq_bot.push_message(f"buy, {record.code}, {record.time}")
        # time.sleep(1)

    def get_parsed_mk_info(self):
        end_time = datetime.strptime(self.date, "%Y-%m-%d") - relativedelta(days=1)
        start_time = end_time - relativedelta(months=2)
        end_date = end_time.date().strftime("%Y-%m-%d")
        start_date = start_time.date().strftime("%Y-%m-%d")

        cache_path = "/home/grasszhang/workspace/projects/gluttonous/data/sor1"
        parquet_path = f"{cache_path}/{self.code}/{self.date}_mk.parquet"
        if os.path.exists(parquet_path):
            df_mkline = pl.read_parquet(parquet_path)
        else:
            df_mkline = get_mkline(self.code, start_date, end_date)
            # print(df_mkline)
            mkline_list = []
            for key, group in df_mkline.group_by(["time"]):
                group = group.with_columns(
                    pl.col("volume")
                    .rolling_mean(window_size=10)
                    .alias("volume_avg_10"),
                )
                mkline_list.append(group)
            df_mkline = pl.concat(mkline_list)
            os.makedirs(f"{cache_path}/{self.code}", exist_ok=True)
            df_mkline.write_parquet(parquet_path)

        return (
            df_mkline.sort("date", "time")
            .filter(pl.col("date").dt.strftime("%Y-%m-%d") == self.last_date)
            .to_dicts()
        )

    def load(self, date: str):
        self.date = date
        end_time = datetime.strptime(self.date, "%Y-%m-%d") - relativedelta(days=1)
        start_time = end_time - relativedelta(months=2)
        end_date = end_time.date().strftime("%Y-%m-%d")
        start_date = start_time.date().strftime("%Y-%m-%d")

        dkline = (
            get_dkline(self.code, start_date, end_date)
            .with_columns(
                pl.col("open").shift(1).alias("pre_open"),
                pl.col("close").rolling_mean(window_size=10).alias("day_10ma"),
                pl.col("close").rolling_mean(window_size=20).alias("day_20ma"),
                pl.col("close").rolling_max(window_size=2).alias("day_2max"),
            )
            .with_columns(
                (
                    (pl.col("close") - pl.col("pre_open")) * 100 / pl.col("pre_open")
                ).alias("day_2delta")
            )
        )
        day_info = dkline.to_dicts()[-1]
        self.pre_day_close = day_info["close"]
        self.day_10ma = day_info["day_10ma"]
        self.day_20ma = day_info["day_20ma"]
        self.day_2max = day_info["day_2max"]
        self.day_2delta = day_info["day_2delta"]
        self.last_date = day_info["date"].strftime("%Y-%m-%d")
        # print(day_info)

        # df_mkline = get_mkline(self.code, start_date, end_date)
        # # print(df_mkline)
        # mkline_list = []
        # for key, group in df_mkline.group_by(["time"]):
        #     group = group.with_columns(
        #         pl.col("volume").rolling_mean(window_size=10).alias("volume_avg_10"),
        #     )
        #     mkline_list.append(group)
        # mk_infos = (
        #     pl.concat(mkline_list)
        #     .sort("date", "time")
        #     .filter(pl.col("date").dt.strftime("%Y-%m-%d") == self.last_date)
        # ).to_dicts()

        mk_infos = self.get_parsed_mk_info()
        for mk_info in mk_infos:
            self.minmap[mk_info["time"].strftime("%H:%M:%S")] = {
                "volume_avg_10": mk_info["volume_avg_10"]
            }


timezone = pytz.timezone("Asia/Shanghai")

cur_date = datetime.now(timezone).date().strftime("%Y-%m-%d")
signals = {}
filter_list = []
for sec_id in tqdm(sub_list):
    signal = SecSignal(sec_id)
    signal.load(cur_date)
    # print(sec_id, signal.shouldFocus())
    if signal.shouldFocus():
        signals[sec_id] = signal

futu_sub_list = [f"{sec[0:2]}{sec[4:]}" for sec in signals.keys()]
print("filter:", len(futu_sub_list))
qq_bot.push_message(f"[stock_on_recv] load sub_list filter len: {len(futu_sub_list)}")
futu_sub_list = futu_sub_list[0:290]
print("sub:", len(futu_sub_list))
qq_bot.push_message(f"[stock_on_recv] load sub_list len: {len(futu_sub_list)}")


class CurKlineTest(CurKlineHandlerBase):
    def on_recv_rsp(self, rsp_pb):
        ret_code, data = super(CurKlineTest, self).on_recv_rsp(rsp_pb)
        if ret_code != RET_OK:
            print("CurKlineTest: error, msg: %s" % data)
            return RET_ERROR, data

        sec_info = pl.DataFrame(data)[
            "code",
            "time_key",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "turnover",
            "last_close",
        ].to_dicts()[0]

        sec_id = f'{sec_info["code"][0:2]}SE{sec_info["code"][2:]}'
        sec_date = sec_info["time_key"][0:10]
        sec_time = sec_info["time_key"][11:19]
        record = KlineRecord(
            code=sec_id,
            date=sec_date,
            time=sec_time,
            open=sec_info["open"],
            close=sec_info["close"],
            volume=sec_info["volume"],
            high=sec_info["high"],
            low=sec_info["low"],
        )
        # print(f"check: {sec_id}")
        signals[sec_id].check(record)
        return RET_OK, data


# 实例化回调处理类
handler = CurKlineTest()
quote_ctx.set_handler(handler)  # 设置实时K线回调
ret, data = quote_ctx.subscribe(futu_sub_list, [SubType.K_1M])
if ret == RET_OK:
    print(data)
else:
    print("error:", data)


while True:
    cur_time = datetime.now(timezone).time().strftime("%H:%M:%S")
    time.sleep(10)
    if cur_time > "16:00:00":
        break

quote_ctx.close()
