"""
策略回测脚本 - 基于 /data/stock/gm 数据
回测时间范围: 2025-01-01 至 2025-12-31
"""
import polars as pl
import toml
from dataclasses import dataclass
from typing import List, Tuple
import sys
sys.path.insert(0, '/home/grasszhang/workspace/projects/gluttonous')

from utils.gm import get_mkline, get_dkline, get_trading_days, get_index_list

pl.Config.set_tbl_cols(100)
pl.Config.set_tbl_rows(50)

# ============================================
# 策略1: 基金突破策略 (Fund Max Strategy)
# ============================================

@dataclass
class FundRecord:
    date: str
    code: str
    price: float
    check_index_price: float
    index_price: float
    check_volume: int
    volume: int
    max_25: float
    avg_vol_10: int


class FundSignal:
    """基金策略信号"""
    def __init__(self):
        self.on_position = False
        self.price = 0.0
        self.rate = 0.0
        self.max_25 = None
        self.avg_vol_10 = None

    def process(self, record: FundRecord) -> str:
        if record.max_25 is None:
            return "None"
        elif self.max_25 is None:
            self.max_25 = record.max_25
            self.avg_vol_10 = record.avg_vol_10
            return "None"

        result = "None"
        if self.on_position:
            if (record.check_index_price <= self.max_25 * 0.95 
                or record.check_volume < self.avg_vol_10):
                self.rate = record.price * 100 / self.price - 100
                self.on_position = False
                result = "Sell"
        else:
            if (record.check_index_price >= self.max_25 
                and record.check_volume > self.avg_vol_10 * 1.2):
                self.price = record.price
                self.on_position = True
                result = "Buy"

        self.max_25 = record.max_25
        self.avg_vol_10 = record.avg_vol_10
        return result


def prepare_fund_data(index_code: str, start_date: str, end_date: str) -> pl.DataFrame:
    """准备基金策略数据 - 使用指数数据模拟"""
    df = get_mkline(index_code, start_date, end_date)
    if df.is_empty():
        return pl.DataFrame()
    
    # 转换时间格式
    df = df.with_columns(
        pl.col("time").cast(pl.Utf8).alias("time_str")
    )
    
    # 计算每日成交量
    df_vol = df.group_by("date").agg([
        pl.col("volume").sum().alias("volume")
    ]).sort("date")
    
    # 计算14:50前累计成交量
    df_check_vol = df.filter(
        pl.col("time_str") <= "14:50:00"
    ).group_by("date").agg([
        pl.col("volume").sum().alias("check_volume")
    ]).sort("date")
    
    # 获取14:50价格
    df_check_price = df.filter(
        pl.col("time_str") == "14:50:00"
    ).select([
        "date", 
        pl.col("close").alias("check_index_price")
    ])
    
    # 获取15:00收盘价
    df_close = df.filter(
        pl.col("time_str") == "15:00:00"
    ).select([
        "date",
        pl.col("close").alias("index_price"),
        pl.col("close").alias("price")  # 模拟ETF价格
    ])
    
    # 合并数据
    result = df_close.join(df_check_vol, on="date").join(df_vol, on="date").join(df_check_price, on="date")
    result = result.sort("date")
    
    # 计算25日最高价和10日均量
    result = result.with_columns([
        pl.col("index_price").rolling_max(window_size=25).alias("max_25"),
        pl.col("check_volume").rolling_mean(window_size=10).cast(pl.Int64).alias("avg_vol_10")
    ])
    
    return result


def backtest_fund_strategy(index_code: str, start_date: str, end_date: str) -> List[dict]:
    """回测基金策略"""
    # 需要额外25天数据计算指标
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    warmup_start = (start_dt - timedelta(days=60)).strftime("%Y-%m-%d")
    
    df = prepare_fund_data(index_code, warmup_start, end_date)
    if df.is_empty():
        return []
    
    signal = FundSignal()
    trades = []
    
    for row in df.iter_rows(named=True):
        if row["date"].strftime("%Y-%m-%d") < start_date:
            # 预热期，只更新状态
            record = FundRecord(
                date=row["date"].strftime("%Y-%m-%d"),
                code=index_code,
                price=row["price"],
                check_index_price=row["check_index_price"],
                index_price=row["index_price"],
                check_volume=row["check_volume"],
                volume=row["volume"],
                max_25=row["max_25"],
                avg_vol_10=row["avg_vol_10"]
            )
            signal.process(record)
            continue
            
        record = FundRecord(
            date=row["date"].strftime("%Y-%m-%d"),
            code=index_code,
            price=row["price"],
            check_index_price=row["check_index_price"],
            index_price=row["index_price"],
            check_volume=row["check_volume"],
            volume=row["volume"],
            max_25=row["max_25"],
            avg_vol_10=row["avg_vol_10"]
        )
        
        action = signal.process(record)
        if action in ["Buy", "Sell"]:
            trades.append({
                "date": record.date,
                "code": index_code,
                "action": action,
                "price": record.price,
                "rate": signal.rate if action == "Sell" else 0
            })
    
    return trades


# ============================================
# 策略2: 个股放量策略 (Stock Volume Strategy)
# ============================================

@dataclass
class StockRecord:
    code: str
    date: str
    time: str
    open: float
    close: float
    volume: float
    high: float
    low: float


class StockSignal:
    """个股放量策略信号"""
    def __init__(self, code: str):
        self.code = code
        self.day_10ma = 0.0
        self.day_20ma = 0.0
        self.day_2max = 0.0
        self.day_2delta = 0.0
        self.day_open = -1.0
        self.pre_day_close = 0.0
        self.minmap = {}
        self.buy_flag = False

    def should_focus(self) -> bool:
        """是否符合筛选条件"""
        if (self.day_10ma <= self.day_20ma 
            or self.day_2delta > 5.0 
            or self.day_2delta < -10.0 
            or self.pre_day_close < self.day_10ma):
            return False
        return True

    def check(self, record: StockRecord) -> bool:
        """检查是否触发买入"""
        if self.buy_flag:
            return False
        if self.day_open > self.day_2max:
            return False
        if record.time < "09:35:00" or record.time > "10:00:00":
            return False
        if (record.close - record.open) * 100 / record.open < 0.5:
            return False
        if record.time not in self.minmap or self.minmap[record.time].get("volume_avg_10", 0) == 0:
            return False
        
        vol_rate = record.volume / self.minmap[record.time]["volume_avg_10"]
        if vol_rate <= 10:
            return False
        
        self.buy_flag = True
        return True


def backtest_stock_strategy(stock_list: List[str], start_date: str, end_date: str) -> List[dict]:
    """回测个股放量策略"""
    from datetime import datetime, timedelta
    
    trades = []
    trading_days = get_trading_days(start_date, end_date)
    
    for date in trading_days:
        # 为每个交易日初始化信号
        date_str = date
        prev_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=90)).strftime("%Y-%m-%d")
        
        buy_count = 0
        for code in stock_list[:100]:  # 限制股票数量加快测试
            try:
                # 获取日线数据计算指标
                dkline = get_dkline(code, prev_date, date)
                if dkline.is_empty() or len(dkline) < 25:
                    continue
                
                dkline = dkline.with_columns([
                    pl.col("open").shift(1).alias("pre_open"),
                    pl.col("close").rolling_mean(window_size=10).alias("day_10ma"),
                    pl.col("close").rolling_mean(window_size=20).alias("day_20ma"),
                    pl.col("close").rolling_max(window_size=2).alias("day_2max"),
                ]).with_columns([
                    ((pl.col("close") - pl.col("pre_open")) * 100 / pl.col("pre_open")).alias("day_2delta")
                ])
                
                last_row = dkline.to_dicts()[-1]
                
                signal = StockSignal(code)
                signal.day_10ma = last_row.get("day_10ma", 0) or 0
                signal.day_20ma = last_row.get("day_20ma", 0) or 0
                signal.day_2max = last_row.get("day_2max", 0) or 0
                signal.day_2delta = last_row.get("day_2delta", 0) or 0
                signal.pre_day_close = last_row.get("close", 0) or 0
                
                if not signal.should_focus():
                    continue
                
                # 获取当日分钟数据
                mkline = get_mkline(code, date, date)
                if mkline.is_empty():
                    continue
                
                mkline = mkline.with_columns(
                    pl.col("time").cast(pl.Utf8).alias("time_str")
                )
                
                # 简化：使用固定倍数检查放量
                for row in mkline.iter_rows(named=True):
                    time_str = row["time_str"]
                    if time_str < "09:35:00" or time_str > "10:00:00":
                        continue
                    
                    signal.minmap[time_str] = {"volume_avg_10": row["volume"] / 15}  # 简化假设
                    signal.day_open = signal.day_open if signal.day_open > 0 else row["open"]
                    
                    record = StockRecord(
                        code=code,
                        date=date,
                        time=time_str,
                        open=row["open"],
                        close=row["close"],
                        volume=row["volume"],
                        high=row["high"],
                        low=row["low"]
                    )
                    
                    if signal.check(record):
                        trades.append({
                            "date": date,
                            "code": code,
                            "action": "Buy",
                            "price": row["close"],
                            "time": time_str
                        })
                        buy_count += 1
                        break
                        
            except Exception as e:
                continue
        
        if buy_count > 0:
            print(f"{date}: {buy_count} 个买入信号")
    
    return trades


def calculate_returns(trades: List[dict]) -> dict:
    """计算收益统计"""
    if not trades:
        return {"total_trades": 0}
    
    buy_trades = [t for t in trades if t["action"] == "Buy"]
    sell_trades = [t for t in trades if t["action"] == "Sell"]
    
    total_return = sum(t.get("rate", 0) for t in sell_trades)
    win_trades = [t for t in sell_trades if t.get("rate", 0) > 0]
    
    return {
        "total_trades": len(buy_trades),
        "completed_trades": len(sell_trades),
        "total_return": round(total_return, 2),
        "avg_return": round(total_return / len(sell_trades), 2) if sell_trades else 0,
        "win_rate": round(len(win_trades) / len(sell_trades) * 100, 2) if sell_trades else 0,
        "win_trades": len(win_trades),
        "lose_trades": len(sell_trades) - len(win_trades)
    }


if __name__ == "__main__":
    print("=" * 60)
    print("策略回测 (2025-01-01 至 2025-12-31)")
    print("=" * 60)
    
    START_DATE = "2025-01-01"
    END_DATE = "2025-12-31"
    
    # ========== 策略1: 基金突破策略 ==========
    print("\n【策略1: 基金突破策略】")
    print("-" * 40)
    
    index_targets = [
        ("SZSE.000852", "中证1000"),
        ("SZSE.000905", "中证500"),
    ]
    
    all_fund_trades = []
    for index_code, index_name in index_targets:
        print(f"\n回测 {index_name} ({index_code})...")
        trades = backtest_fund_strategy(index_code, START_DATE, END_DATE)
        all_fund_trades.extend(trades)
        
        stats = calculate_returns(trades)
        print(f"  交易次数: {stats['total_trades']}")
        print(f"  完成交易: {stats['completed_trades']}")
        print(f"  总收益率: {stats['total_return']}%")
        print(f"  胜率: {stats['win_rate']}%")
        
        if trades:
            print(f"  交易明细:")
            for t in trades[:10]:
                print(f"    {t['date']} {t['action']} @ {t['price']:.2f}" + 
                      (f" 收益: {t['rate']:.2f}%" if t['action'] == 'Sell' else ""))
    
    # ========== 策略2: 个股放量策略 ==========
    print("\n" + "=" * 60)
    print("【策略2: 个股放量策略】")
    print("-" * 40)
    
    # 获取中证1000成分股
    try:
        stock_list = get_index_list("SHSE.000852")
        print(f"股票池: 中证1000 ({len(stock_list)} 只)")
    except:
        stock_list = []
        print("无法获取股票列表")
    
    if stock_list:
        print(f"\n回测中 (仅测试前100只股票)...")
        trades = backtest_stock_strategy(stock_list, START_DATE, END_DATE)
        
        stats = calculate_returns(trades)
        print(f"\n个股策略统计:")
        print(f"  买入信号数: {len(trades)}")
        
        if trades:
            print(f"\n  部分交易明细:")
            for t in trades[:20]:
                print(f"    {t['date']} {t['time']} {t['code']} Buy @ {t['price']:.2f}")
    
    print("\n" + "=" * 60)
    print("回测完成")
