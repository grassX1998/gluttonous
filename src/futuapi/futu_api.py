import os
import polars as pl
from datetime import datetime, timedelta
from futu import TrdMarket, OpenQuoteContext, RET_OK
from tqdm import tqdm
from .utils import rate_limited, load_existing_trading_days, save_trading_days_to_toml


class FutuApi:
    def __init__(self, host="127.0.0.1", port=11111, ktype="K_1M"):
        self.quote_ctx = OpenQuoteContext(host=host, port=port)
        self.max_count = 1000  # 每次请求的最大记录数
        self.ktype = ktype
        self.dir_path = "/data/futu/A-shares"
        self.cfg_dir = "/data/futu/A-shares/cfg"
        os.makedirs(self.cfg_dir, exist_ok=True)  # 确保目录存在
        self.save_trading_days_to_toml()

    def close(self):
        self.quote_ctx.close()

    def get_minute_kline(self, code, start_date, end_date):
        # 将 start_date 和 end_date 转换为 datetime 对象
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # 加载所有的交易日
        trading_days = self.get_trading_days(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        # print(trading_days)

        all_data = pl.DataFrame()
        for trading_day in tqdm(trading_days, desc="Processing trading days"):
            store_dir_path = f"{self.dir_path}/{code}/{trading_day[0:4]}/"
            store_path = f"{store_dir_path}/{trading_day}.parquet"
            if os.path.exists(store_path):
                df = pl.read_parquet(store_path)
                # print(trading_day)
                all_data = pl.concat([all_data, df], how="vertical")
            else:
                df = self.get_minute_kline_by_api(code, trading_day)
                os.makedirs(store_dir_path, exist_ok=True)
                df.write_parquet(store_path)
                all_data = pl.concat([all_data, df], how="vertical")

        return all_data

    def calculate_missing_intervals(self, start_date, end_date, min_time, max_time):
        missing_intervals = []
        if min_time > datetime.strptime(start_date, "%Y-%m-%d"):
            missing_intervals.append(
                (
                    start_date,
                    (min_time - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
                )
            )
        if max_time < datetime.strptime(end_date, "%Y-%m-%d"):
            missing_intervals.append(
                (
                    (max_time + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    end_date,
                )
            )
        return missing_intervals

    @rate_limited()
    def get_minute_kline_by_api(self, code, date):
        ret, data, page_req_key = self.quote_ctx.request_history_kline(
            code,
            start=date,
            end=date,
            ktype=self.ktype,
            max_count=self.max_count,
        )
        all_data = pl.DataFrame()
        if ret == RET_OK:
            df = pl.DataFrame(data)
            all_data = pl.concat([all_data, df], how="vertical")
        else:
            print("Error:", data)

        return all_data

    @rate_limited()
    def get_trading_days_by_api(
        self, market=TrdMarket.CN, start_date=None, end_date=None
    ):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=8 * 365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        ret, data = self.quote_ctx.request_trading_days(
            market=market, start=start_date, end=end_date
        )

        # print(start_date, end_date)

        if ret == RET_OK:
            trading_days = [item["time"] for item in data if "time" in item]
            return trading_days
        else:
            print(f"Error retrieving trading days: {data}")
            return []

    def get_trading_days(self, start_date=None, end_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=8 * 365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        file_path = os.path.join(self.cfg_dir, "trading_days.toml")
        all_trading_days = load_existing_trading_days(file_path)
        # print(all_trading_days)
        # Filter trading days within the specified range
        trading_days_within_range = [
            day for day in all_trading_days if start_date <= day <= end_date
        ]

        return trading_days_within_range

    def save_trading_days_to_toml(self):
        file_path = os.path.join(self.cfg_dir, "trading_days.toml")
        existing_trading_days = self.get_trading_days()

        if existing_trading_days:
            last_trading_day = existing_trading_days[-1]
            start_date = (
                datetime.strptime(last_trading_day, "%Y-%m-%d")  # + timedelta(days=1)
            ).strftime("%Y-%m-%d")
        else:
            start_date = (datetime.now() - timedelta(days=8 * 365)).strftime("%Y-%m-%d")

        end_date = datetime.now().strftime("%Y-%m-%d")
        # print(start_date)

        new_trading_days = self.get_trading_days_by_api(
            start_date=start_date, end_date=end_date
        )

        all_trading_days = sorted(set(existing_trading_days + new_trading_days))

        save_trading_days_to_toml(file_path, all_trading_days)
