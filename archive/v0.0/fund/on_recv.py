import polars as pl
import pytz

from utils.qq_bot import *

from futu import *
from str_fund_max_1 import StrFundMax1

strs = []
strs.append(StrFundMax1())
sub_list = []

timezone = pytz.timezone("Asia/Shanghai")

for strategy in strs:
    strategy.load()
    sub_list += strategy.sub_list

sub_list = list(set(sub_list))
push_message(f"[fund_on_recv] subs: {list(set(sub_list))}")


# 创建行情客户端
quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)

# all_data = pl.DataFrame()
# i = 0


# 定义回调函数，用于处理实时行情数据
class StockQuoteHandler(StockQuoteHandlerBase):
    def on_recv_rsp(self, rsp_str):
        ret_code, data = super().on_recv_rsp(rsp_str)
        if ret_code != RET_OK:
            print(f"Error: {ret_code}")
            return
        # 输出实时行情数据
        df = pl.DataFrame(data)["code", "last_price", "volume", "data_time"]
        [code, price, volume, data_time] = df.to_numpy().tolist()[0]
        for strategy in strs:
            strategy.check(code, price, volume, data_time)
        # print(data["data_time"])
        # print(data["code"], data["last_price"], data["volume"])
        # all_data = pl.concat([all_data, pl.DataFrame(data)], how="vertical")
        # i = i + 1


# 实例化回调处理类
handler = StockQuoteHandler()

# 订阅股票的实时行情

# sub list
sub_type = SubType.QUOTE
for sec_id in sub_list:
    ret_sub, err_message = quote_ctx.subscribe(sec_id, sub_type)
    if ret_sub == RET_OK:
        # 注册回调函数
        quote_ctx.set_handler(handler)
    else:
        print(f"订阅失败: {err_message}")

while True:
    cur_time = datetime.now(timezone).time().strftime("%H:%M:%S")
    time.sleep(10)
    # print(cur_time)
    if cur_time > "16:00:00":
        break

# 关闭行情客户端
quote_ctx.close()
