"""
特征工程模块

从清洗后的数据计算技术指标特征，准备训练数据
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    CLEANED_DATA_DIR, FEATURE_DATA_DIR, TRAIN_DATA_DIR, FEATURE_CONFIG
)
from pipeline.shared.utils import (
    setup_logger, timer, save_parquet_optimized
)


logger = setup_logger("feature_engineering")


# 特征列表
FEATURE_COLS = [
    # 收益率特征
    "ret_1", "ret_5", "ret_10", "ret_20",
    # 移动平均特征
    "ma_5_ratio", "ma_10_ratio", "ma_20_ratio", "ma_60_ratio",
    # 波动率特征
    "volatility_5", "volatility_10", "volatility_20",
    # 技术指标
    "rsi_14",
    "macd_dif", "macd_dea", "macd_hist",
    "bb_position",
    # 成交量特征
    "volume_ratio_5", "volume_ratio_20",
    "turnover_ma_ratio",
    # ===== 策略相关特征 (fund_max_1) =====
    "open_gap",              # 开盘涨跌幅（今开/昨收-1）
    "price_vs_max_25",       # 当前价/25日最高价
    "volume_ratio_10",       # 成交量/10日均量
    "price_breakout",        # 是否突破25日最高价 (0/1)
    "stop_loss_signal",      # 是否触及止损线 (0/1)
    # ===== stock_volume_1 策略特征 =====
    "ma_10_vs_20",           # 10日均线/20日均线-1（多头排列强度）
    "price_vs_ma_10",        # 收盘价/10日均线-1（站上均线强度）
    "ret_2d",                # 近2日涨幅
    "price_vs_2d_max",       # 收盘价/近2日最高价（开盘跳空检测）
    "intraday_vol_spike",    # 日内最大成交量/均量（分钟放量信号聚合）
    "morning_vol_ratio",     # 早盘成交量占比（09:30-10:00）
    "morning_price_change",  # 早盘涨幅
    # ===== 市场状态特征 =====
    "market_ret_1",          # 大盘1日收益
    "market_ret_5",          # 大盘5日收益
    "market_ma_10_ratio",    # 大盘相对10日均线位置
    "market_volatility",     # 大盘波动率
    "stock_vs_market",       # 个股相对大盘强弱
    # ===== 涨跌停标记 =====
    "is_limit_up",           # 涨停标记 (1=涨停, 0=未涨停)
    "is_limit_down",         # 跌停标记 (1=跌停, 0=未跌停)
]

# 策略参数
STRATEGY_PARAMS = {
    "max_window": 25,        # 计算最高价的滚动窗口（天）
    "vol_avg_window": 10,    # 计算成交量均值的滚动窗口（天）
    "vol_ratio": 1.2,        # 买入时成交量需达到均量的倍数
    "stop_loss_ratio": 0.95, # 止损线：跌破最高价的 95%
    "holding_days": 5,       # 持仓天数（计算收益期望）
}

# stock_volume_1 策略参数
STOCK_VOL_PARAMS = {
    "ma_short": 10,          # 短期均线周期
    "ma_long": 20,           # 长期均线周期
    "vol_ratio_threshold": 10,  # 分钟成交量倍数阈值
    "price_change_threshold": 0.005,  # 分钟涨幅阈值 (0.5%)
    "check_start_time": "09:35:00",   # 信号检查开始时间
    "check_end_time": "10:00:00",     # 信号检查结束时间
}


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        self.config = FEATURE_CONFIG
        self.market_data = None  # 缓存市场数据
    
    def load_market_data(self) -> pl.DataFrame:
        """加载市场指数数据（中证500 SZSE.000905）"""
        if self.market_data is not None:
            return self.market_data
        
        from pipeline.shared.config import RAW_DATA_ROOT
        
        index_symbol = "SZSE.000905"  # 中证500指数
        index_path = RAW_DATA_ROOT / "mkline" / index_symbol
        
        if not index_path.exists():
            logger.warning(f"Market index path not found: {index_path}")
            return None
        
        # 读取所有日期的数据
        all_data = []
        for f in sorted(index_path.glob("*.parquet")):
            try:
                df = pl.read_parquet(f)
                all_data.append(df)
            except:
                pass
        
        if not all_data:
            logger.warning("No market data found")
            return None
        
        df = pl.concat(all_data)
        
        # 聚合为日线
        if "time" in df.columns:
            df = df.group_by("date").agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
            ]).sort("date")
        
        # 计算市场指标
        df = df.with_columns([
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("market_ret_1"),
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("market_ret_5"),
            (pl.col("close") / pl.col("close").rolling_mean(window_size=10) - 1).alias("market_ma_10_ratio"),
            (pl.col("close") / pl.col("close").shift(1) - 1).rolling_std(window_size=10).alias("market_volatility"),
        ])
        
        self.market_data = df.select(["date", "market_ret_1", "market_ret_5", 
                                       "market_ma_10_ratio", "market_volatility"])
        logger.info(f"Loaded market data: {self.market_data.height} days")
        return self.market_data
    
    @staticmethod
    def calc_returns(df: pl.DataFrame, periods: list[int] = [1, 5, 10, 20]) -> pl.DataFrame:
        """计算收益率"""
        exprs = []
        for p in periods:
            exprs.append(
                (pl.col("close") / pl.col("close").shift(p) - 1).alias(f"ret_{p}")
            )
        return df.with_columns(exprs)
    
    @staticmethod
    def calc_ma(df: pl.DataFrame, windows: list[int] = [5, 10, 20, 60]) -> pl.DataFrame:
        """计算移动平均线相对位置"""
        exprs = []
        for w in windows:
            exprs.append(
                (pl.col("close") / pl.col("close").rolling_mean(window_size=w) - 1).alias(f"ma_{w}_ratio")
            )
        return df.with_columns(exprs)
    
    @staticmethod
    def calc_volatility(df: pl.DataFrame, windows: list[int] = [5, 10, 20]) -> pl.DataFrame:
        """计算波动率"""
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("_ret")
        )
        
        exprs = []
        for w in windows:
            exprs.append(
                pl.col("_ret").rolling_std(window_size=w).alias(f"volatility_{w}")
            )
        
        return df.with_columns(exprs).drop("_ret")
    
    @staticmethod
    def calc_rsi(df: pl.DataFrame, window: int = 14) -> pl.DataFrame:
        """计算RSI指标"""
        df = df.with_columns(
            (pl.col("close") - pl.col("close").shift(1)).alias("_change")
        )
        
        df = df.with_columns([
            pl.when(pl.col("_change") > 0).then(pl.col("_change")).otherwise(0).alias("_gain"),
            pl.when(pl.col("_change") < 0).then(-pl.col("_change")).otherwise(0).alias("_loss"),
        ])
        
        df = df.with_columns([
            pl.col("_gain").rolling_mean(window_size=window).alias("_avg_gain"),
            pl.col("_loss").rolling_mean(window_size=window).alias("_avg_loss"),
        ])
        
        df = df.with_columns(
            (100 - 100 / (1 + pl.col("_avg_gain") / (pl.col("_avg_loss") + 1e-10))).alias(f"rsi_{window}")
        )
        
        return df.drop(["_change", "_gain", "_loss", "_avg_gain", "_avg_loss"])
    
    @staticmethod
    def calc_macd(df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
        """计算MACD指标"""
        df = df.with_columns([
            pl.col("close").ewm_mean(span=fast).alias("_ema_fast"),
            pl.col("close").ewm_mean(span=slow).alias("_ema_slow"),
        ])
        
        df = df.with_columns(
            (pl.col("_ema_fast") - pl.col("_ema_slow")).alias("macd_dif")
        )
        
        df = df.with_columns(
            pl.col("macd_dif").ewm_mean(span=signal).alias("macd_dea")
        )
        
        df = df.with_columns(
            (2 * (pl.col("macd_dif") - pl.col("macd_dea"))).alias("macd_hist")
        )
        
        return df.drop(["_ema_fast", "_ema_slow"])
    
    @staticmethod
    def calc_bollinger(df: pl.DataFrame, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
        """计算布林带位置"""
        df = df.with_columns([
            pl.col("close").rolling_mean(window_size=window).alias("_bb_mid"),
            pl.col("close").rolling_std(window_size=window).alias("_bb_std"),
        ])
        
        df = df.with_columns([
            (pl.col("_bb_mid") + num_std * pl.col("_bb_std")).alias("_bb_upper"),
            (pl.col("_bb_mid") - num_std * pl.col("_bb_std")).alias("_bb_lower"),
        ])
        
        # 价格在布林带中的相对位置 (0-1)
        df = df.with_columns(
            ((pl.col("close") - pl.col("_bb_lower")) / 
             (pl.col("_bb_upper") - pl.col("_bb_lower") + 1e-10)).alias("bb_position")
        )
        
        return df.drop(["_bb_mid", "_bb_std", "_bb_upper", "_bb_lower"])
    
    @staticmethod
    def calc_volume_features(df: pl.DataFrame) -> pl.DataFrame:
        """计算成交量特征"""
        df = df.with_columns([
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=5)).alias("volume_ratio_5"),
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20)).alias("volume_ratio_20"),
            (pl.col("turnover") / pl.col("turnover").rolling_mean(window_size=20)).alias("turnover_ma_ratio"),
        ])
        return df
    
    @staticmethod
    def calc_strategy_features(df: pl.DataFrame) -> pl.DataFrame:
        """计算策略相关特征 (fund_max_1)"""
        max_window = STRATEGY_PARAMS["max_window"]
        vol_avg_window = STRATEGY_PARAMS["vol_avg_window"]
        vol_ratio = STRATEGY_PARAMS["vol_ratio"]
        stop_loss_ratio = STRATEGY_PARAMS["stop_loss_ratio"]
        
        df = df.with_columns([
            # 开盘涨跌幅：今日开盘价 / 昨日收盘价 - 1
            (pl.col("open") / pl.col("close").shift(1) - 1).alias("open_gap"),
            
            # 25日滚动最高价
            pl.col("high").rolling_max(window_size=max_window).alias("_max_25"),
            
            # 10日成交量均值
            pl.col("volume").rolling_mean(window_size=vol_avg_window).alias("_vol_avg_10"),
        ])
        
        df = df.with_columns([
            # 当前价/25日最高价
            (pl.col("close") / pl.col("_max_25")).alias("price_vs_max_25"),
            
            # 成交量/10日均量
            (pl.col("volume") / pl.col("_vol_avg_10")).alias("volume_ratio_10"),
            
            # 是否突破25日最高价 (收盘价 >= 25日最高价)
            pl.when(pl.col("close") >= pl.col("_max_25")).then(1.0).otherwise(0.0).alias("price_breakout"),
            
            # 是否触及止损线 (收盘价 <= 25日最高价 * 0.95)
            pl.when(pl.col("close") <= pl.col("_max_25") * stop_loss_ratio).then(1.0).otherwise(0.0).alias("stop_loss_signal"),
        ])
        
        # 删除临时列
        df = df.drop(["_max_25", "_vol_avg_10"])
        
        return df
    
    @staticmethod
    def calc_limit_features(df: pl.DataFrame) -> pl.DataFrame:
        """计算涨跌停标记
        
        A股涨跌停规则:
        - 普通股票: ±10%
        - ST股票: ±5%
        - 科创板/创业板(部分): ±20%
        
        这里使用通用的阈值判断:
        - 涨停: 涨幅 >= 9.5% (考虑四舍五入)
        - 跌停: 跌幅 <= -9.5%
        """
        df = df.with_columns([
            # 计算日涨跌幅
            (pl.col("close") / pl.col("close").shift(1) - 1).alias("_daily_ret"),
        ])
        
        df = df.with_columns([
            # 涨停标记: 涨幅 >= 9.5%
            pl.when(pl.col("_daily_ret") >= 0.095).then(1.0).otherwise(0.0).alias("is_limit_up"),
            # 跌停标记: 跌幅 <= -9.5%
            pl.when(pl.col("_daily_ret") <= -0.095).then(1.0).otherwise(0.0).alias("is_limit_down"),
        ])
        
        # 删除临时列
        df = df.drop(["_daily_ret"])
        
        return df
    
    @staticmethod
    def calc_stock_volume_features(df: pl.DataFrame) -> pl.DataFrame:
        """计算 stock_volume_1 策略特征（日线级别）"""
        ma_short = STOCK_VOL_PARAMS["ma_short"]
        ma_long = STOCK_VOL_PARAMS["ma_long"]
        
        df = df.with_columns([
            # 10日均线
            pl.col("close").rolling_mean(window_size=ma_short).alias("_ma_10"),
            # 20日均线
            pl.col("close").rolling_mean(window_size=ma_long).alias("_ma_20"),
            # 近2日最高价
            pl.col("high").rolling_max(window_size=2).alias("_high_2d"),
            # 前日开盘价
            pl.col("open").shift(1).alias("_pre_open"),
        ])
        
        df = df.with_columns([
            # 10日均线 / 20日均线 - 1（多头排列强度，>0 为多头）
            (pl.col("_ma_10") / pl.col("_ma_20") - 1).alias("ma_10_vs_20"),
            
            # 收盘价 / 10日均线 - 1（站上均线强度）
            (pl.col("close") / pl.col("_ma_10") - 1).alias("price_vs_ma_10"),
            
            # 近2日涨幅 = (close - pre_open) / pre_open
            ((pl.col("close") - pl.col("_pre_open")) / pl.col("_pre_open")).alias("ret_2d"),
            
            # 收盘价 / 近2日最高价（用于检测开盘跳空）
            (pl.col("close") / pl.col("_high_2d")).alias("price_vs_2d_max"),
        ])
        
        # 删除临时列
        df = df.drop(["_ma_10", "_ma_20", "_high_2d", "_pre_open"])
        
        return df
    
    @staticmethod
    def calc_minute_features(df: pl.DataFrame) -> pl.DataFrame | None:
        """从分钟数据计算日内特征聚合（优化版本）
        
        捕捉 stock_volume_1 策略的分钟级别放量信号：
        - 早盘(09:30-10:00)成交量异常放大
        - 分钟涨幅突破
        """
        if "time" not in df.columns:
            return None
        
        # 确保time是字符串格式便于比较
        df = df.with_columns(
            pl.col("time").cast(pl.Utf8).alias("_time_str")
        )
        
        # 标记早盘时段
        df = df.with_columns(
            ((pl.col("_time_str") >= "09:30:00") & 
             (pl.col("_time_str") <= "10:00:00")).alias("_is_morning")
        )
        
        # 使用 Polars 原生 group_by 聚合（比 Python for 循环快很多）
        daily_features = df.group_by("date").agg([
            # 日内最大成交量
            pl.col("volume").max().alias("_max_vol"),
            # 日内平均成交量
            pl.col("volume").mean().alias("_mean_vol"),
            # 总成交量
            pl.col("volume").sum().alias("_total_vol"),
            # 早盘成交量
            pl.col("volume").filter(pl.col("_is_morning")).sum().alias("_morning_vol"),
            # 早盘第一根K线开盘价
            pl.col("open").filter(pl.col("_is_morning")).first().alias("_morning_open"),
            # 早盘最后一根K线收盘价
            pl.col("close").filter(pl.col("_is_morning")).last().alias("_morning_close"),
        ])
        
        # 计算特征
        daily_features = daily_features.with_columns([
            # 日内最大成交量 / 均量
            (pl.col("_max_vol") / (pl.col("_mean_vol") + 1e-10)).alias("intraday_vol_spike"),
            # 早盘成交量占比
            (pl.col("_morning_vol") / (pl.col("_total_vol") + 1e-10)).alias("morning_vol_ratio"),
            # 早盘涨幅
            ((pl.col("_morning_close") - pl.col("_morning_open")) / (pl.col("_morning_open") + 1e-10)).alias("morning_price_change"),
        ])
        
        # 只保留需要的列
        daily_features = daily_features.select([
            "date", "intraday_vol_spike", "morning_vol_ratio", "morning_price_change"
        ])
        
        # 处理 NaN 和 Inf
        daily_features = daily_features.with_columns([
            pl.col("intraday_vol_spike").fill_nan(0).fill_null(0),
            pl.col("morning_vol_ratio").fill_nan(0).fill_null(0),
            pl.col("morning_price_change").fill_nan(0).fill_null(0),
        ])
        
        return daily_features
    
    def calc_all_features(self, df: pl.DataFrame, add_market: bool = True) -> pl.DataFrame:
        """计算所有特征"""
        df = self.calc_returns(df)
        df = self.calc_ma(df)
        df = self.calc_volatility(df)
        df = self.calc_rsi(df)
        df = self.calc_macd(df)
        df = self.calc_bollinger(df)
        df = self.calc_volume_features(df)
        df = self.calc_strategy_features(df)  # fund_max_1 策略特征
        df = self.calc_stock_volume_features(df)  # stock_volume_1 策略特征
        df = self.calc_limit_features(df)  # 涨跌停标记
        
        # 添加市场状态特征
        if add_market:
            market_df = self.load_market_data()
            if market_df is not None:
                df = df.join(market_df, on="date", how="left")
                # 计算个股相对大盘强弱
                df = df.with_columns(
                    (pl.col("ret_5") - pl.col("market_ret_5")).alias("stock_vs_market")
                )
        
        return df
    
    @timer
    def process_symbol(self, symbol: str) -> pl.DataFrame | None:
        """处理单只股票的特征"""
        cleaned_path = CLEANED_DATA_DIR / f"{symbol}.parquet"
        
        if not cleaned_path.exists():
            return None
        
        df = pl.read_parquet(cleaned_path)
        
        # 按日期聚合为日线数据（如果是分钟数据）
        minute_features = None
        if "time" in df.columns:
            # 先计算分钟级别的聚合特征
            minute_features = self.calc_minute_features(df)
            
            # 聚合为日线
            df = df.group_by("date").agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum(),
                pl.col("turnover").sum(),
            ]).sort("date")
            
            # 合并分钟级别特征
            if minute_features is not None:
                df = df.join(minute_features, on="date", how="left")
        
        # 计算特征
        df = self.calc_all_features(df)
        
        # 确保分钟级别特征存在（填充默认值）
        minute_feature_cols = ["intraday_vol_spike", "morning_vol_ratio", "morning_price_change"]
        for col in minute_feature_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))
            else:
                df = df.with_columns(pl.col(col).fill_null(0.0))
        
        # ===== 新标签逻辑：相对收益（跑赢大盘） =====
        holding_days = STRATEGY_PARAMS["holding_days"]
        
        # 计算次日开盘价买入后，持有N天的收益
        # 假设：今日收盘决策 -> 次日开盘买入 -> 持有N天后开盘卖出
        df = df.with_columns([
            # 次日开盘价
            pl.col("open").shift(-1).alias("_next_open"),
            # N天后开盘价
            pl.col("open").shift(-1 - holding_days).alias("_future_open"),
        ])
        
        # 买入收益 = (N天后开盘价 / 次日开盘价) - 1
        df = df.with_columns(
            ((pl.col("_future_open") / pl.col("_next_open")) - 1).alias("future_ret")
        )
        
        # 加入大盘同期收益
        market_df = self.load_market_data()
        if market_df is not None:
            # 计算大盘未来N天收益
            market_future = market_df.with_columns([
                (pl.col("market_ret_1").shift(-holding_days).rolling_sum(window_size=holding_days)).alias("market_future_ret")
            ]).select(["date", "market_future_ret"])
            df = df.join(market_future, on="date", how="left")
            
            # 相对收益 = 个股收益 - 大盘收益
            df = df.with_columns(
                (pl.col("future_ret") - pl.col("market_future_ret").fill_null(0)).alias("relative_ret")
            )
            
            # 标签：相对收益 > 0（跑赢大盘）为1，否则为0
            df = df.with_columns(
                pl.when(pl.col("relative_ret") > 0).then(1).otherwise(0).alias("label")
            )
            
            df = df.drop(["market_future_ret"])
        else:
            # 没有市场数据时，使用绝对收益
            df = df.with_columns(
                pl.when(pl.col("future_ret") > 0).then(1).otherwise(0).alias("label")
            )
        
        # 删除临时列
        df = df.drop(["_next_open", "_future_open"])
        
        # 移除NaN
        df = df.drop_nulls()
        
        # 添加symbol列
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        
        return df


def _process_symbol_worker(args):
    """并行处理单只股票的worker函数"""
    symbol, market_df = args
    try:
        # 读取清洗后的数据
        cleaned_path = CLEANED_DATA_DIR / f"{symbol}.parquet"
        if not cleaned_path.exists():
            return None
        
        df = pl.read_parquet(cleaned_path)
        
        # 创建特征工程实例并处理
        engineer = FeatureEngineer()
        engineer.market_df = market_df
        
        result = engineer.process_symbol(symbol)
        
        if result is not None and result.height > 0:
            # 保存特征文件
            save_parquet_optimized(result, FEATURE_DATA_DIR / f"{symbol}.parquet")
            return result
        return None
    except Exception as e:
        return None


class FeatureEngineerRunner:
    """特征工程运行器（包含并行处理逻辑）"""
    
    def __init__(self):
        self.engineer = FeatureEngineer()
    
    @timer
    def run(self, symbol_limit: int | None = None, num_workers: int = 8):
        """执行特征工程（并行处理）"""
        logger.info("="*60)
        logger.info("Starting Feature Engineering")
        logger.info("="*60)
        
        cleaned_files = list(CLEANED_DATA_DIR.glob("*.parquet"))
        
        if symbol_limit:
            cleaned_files = cleaned_files[:symbol_limit]
        
        symbols = [f.stem for f in cleaned_files]
        logger.info(f"Processing {len(symbols)} symbols with {num_workers} workers...")
        
        # 预先加载市场数据供所有worker使用
        self.engineer.load_market_data()
        
        processed = 0
        success = 0
        
        # 使用线程池并行处理（I/O密集型）
        from concurrent.futures import ThreadPoolExecutor
        
        def process_one(symbol):
            try:
                df = self.engineer.process_symbol(symbol)
                if df is not None and df.height > 0:
                    save_parquet_optimized(df, FEATURE_DATA_DIR / f"{symbol}.parquet")
                    return symbol  # 只返回symbol名，不返回数据
            except Exception as e:
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one, s): s for s in symbols}
            
            for future in as_completed(futures):
                processed += 1
                result = future.result()
                if result is not None:
                    success += 1
                
                if processed % 100 == 0:
                    logger.info(f"Progress: {processed}/{len(symbols)} ({success} success)")
        
        logger.info(f"Completed: {success}/{len(symbols)} success")
        
        if success == 0:
            logger.error("No features generated!")
            return
        
        # 流式合并特征文件（避免内存溢出）
        logger.info("Merging features (streaming mode)...")
        self.merge_and_prepare_train_data()
    
    @timer
    def merge_and_prepare_train_data(self):
        """流式合并特征并准备训练数据（内存优化版 - 使用memory-mapped array）"""
        config = self.engineer.config
        lookback = config["lookback_days"]
        n_features = len(FEATURE_COLS)
        
        # 获取所有特征文件
        feature_files = sorted(FEATURE_DATA_DIR.glob("*.parquet"))
        feature_files = [f for f in feature_files if f.stem != "features_all"]
        
        if not feature_files:
            logger.error("No feature files found!")
            return
        
        logger.info(f"Found {len(feature_files)} feature files")
        
        # 第一遍：计算总样本数
        logger.info("Pass 1: Counting samples...")
        total_samples = 0
        file_sample_counts = []
        
        for f in feature_files:
            try:
                df = pl.read_parquet(f)
                n = max(0, df.height - lookback)
                file_sample_counts.append(n)
                total_samples += n
            except:
                file_sample_counts.append(0)
        
        logger.info(f"Total samples to generate: {total_samples:,}")
        
        if total_samples == 0:
            logger.error("No samples!")
            return
        
        # 使用临时文件存储中间结果，避免内存峰值
        temp_dir = TRAIN_DATA_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 第二遍：分批处理并保存到临时文件
        logger.info("Pass 2: Creating sequences in batches...")
        batch_size = 150  # 每批文件数
        batch_idx = 0
        
        for i in range(0, len(feature_files), batch_size):
            batch_files = feature_files[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(feature_files) - 1) // batch_size + 1
            
            batch_X, batch_y, batch_dates = [], [], []
            
            for j, f in enumerate(batch_files):
                try:
                    df = pl.read_parquet(f)
                    sequences, labels, dates = self._create_sequences_from_single_df(df, lookback)
                    if len(sequences) > 0:
                        batch_X.extend(sequences)
                        batch_y.extend(labels)
                        batch_dates.extend(dates)
                except:
                    pass
            
            if batch_X:
                # 立即转成numpy并保存到临时文件（不压缩，加快速度）
                np.savez(
                    temp_dir / f"batch_{batch_idx}.npz",
                    X=np.array(batch_X, dtype=np.float32),
                    y=np.array(batch_y, dtype=np.int8),
                    dates=np.array(batch_dates)
                )
                logger.info(f"Batch {batch_num}/{total_batches}: saved {len(batch_X):,} samples")
                batch_idx += 1
            
            # 释放内存
            del batch_X, batch_y, batch_dates
        
        # 第三遍：增量合并临时文件（避免内存峰值）
        logger.info("Pass 3: Incremental merge...")
        temp_files = sorted(temp_dir.glob("batch_*.npz"))
        
        # 先统计每个批次的样本数
        batch_sizes = []
        for tf in temp_files:
            data = np.load(tf, mmap_mode='r')
            batch_sizes.append(len(data["X"]))
            data.close()
        
        total_n = sum(batch_sizes)
        n_features = len(FEATURE_COLS)
        lookback = config["lookback_days"]
        
        logger.info(f"Total samples: {total_n:,}")
        
        # 预分配最终数组（使用memory-mapped临时文件）
        temp_X_path = temp_dir / "merged_X.npy"
        temp_y_path = temp_dir / "merged_y.npy"
        temp_dates_path = temp_dir / "merged_dates.npy"
        
        # 创建memmap文件
        X = np.memmap(temp_X_path, dtype=np.float32, mode='w+', shape=(total_n, lookback, n_features))
        y = np.memmap(temp_y_path, dtype=np.int8, mode='w+', shape=(total_n,))
        
        # 日期需要特殊处理（字符串无法memmap）
        all_dates = []
        
        # 逐批复制
        offset = 0
        for i, tf in enumerate(temp_files):
            data = np.load(tf, allow_pickle=True)  # dates是字符串数组，需要allow_pickle
            n = len(data["X"])
            X[offset:offset+n] = data["X"]
            y[offset:offset+n] = data["y"]
            all_dates.extend(data["dates"].tolist())
            offset += n
            data.close()
            logger.info(f"  Merged batch {i+1}/{len(temp_files)}")
        
        # 将日期转为numpy数组
        dates = np.array(all_dates)
        del all_dates
        
        logger.info(f"Final shape: {X.shape}")
        
        # 清理临时npz文件
        for tf in temp_files:
            tf.unlink()
        
        # 继续处理（排序、划分、标准化、保存）
        self._finalize_train_data(X, y, dates, config, temp_dir)
    
    def _create_sequences_from_single_df(self, df: pl.DataFrame, lookback: int):
        """从单个股票DataFrame创建序列（不返回numpy数组，减少内存）"""
        df = df.sort("date")
        
        if df.height < lookback + 1:
            return [], [], []
        
        features = df.select(FEATURE_COLS).to_numpy()
        label_arr = df["label"].to_numpy()
        date_arr = df["date"].to_list()
        
        # 处理NaN
        features = np.nan_to_num(features, 0)
        
        sequences = []
        labels = []
        dates = []
        
        for i in range(lookback, len(features)):
            seq = features[i-lookback:i]
            sequences.append(seq)
            labels.append(label_arr[i])
            dates.append(date_arr[i])
        
        return sequences, labels, dates
    
    def _finalize_train_data(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray, config: dict, temp_dir=None):
        """完成训练数据处理：排序、划分、标准化、保存（内存优化版）"""
        n = len(X)
        
        # 按日期排序（只排序索引，不复制数据）
        logger.info("Sorting by date...")
        sort_idx = np.argsort(dates)
        dates_sorted = dates[sort_idx]
        
        logger.info(f"Date range: {dates_sorted[0]} to {dates_sorted[-1]}")
        
        # 划分数据集（严格按时间顺序）
        train_end = int(n * config.get("train_ratio", 0.7))
        val_end = int(n * (config.get("train_ratio", 0.7) + config.get("val_ratio", 0.15)))
        
        train_idx = sort_idx[:train_end]
        val_idx = sort_idx[train_end:val_end]
        test_idx = sort_idx[val_end:]
        
        train_dates = dates_sorted[:train_end]
        val_dates = dates_sorted[train_end:val_end]
        test_dates = dates_sorted[val_end:]
        
        logger.info(f"Train period: {train_dates[0]} to {train_dates[-1]}")
        logger.info(f"Val period:   {val_dates[0]} to {val_dates[-1]}")
        logger.info(f"Test period:  {test_dates[0]} to {test_dates[-1]}")
        
        # 检查时间不重叠
        assert train_dates[-1] <= val_dates[0], "时间泄露！训练集与验证集重叠"
        assert val_dates[-1] <= test_dates[0], "时间泄露！验证集与测试集重叠"
        
        # 只用训练集计算标准化参数（分批计算避免内存峰值）
        logger.info("Computing mean/std from training set...")
        batch_size = 50000
        sum_x = np.zeros(X.shape[-1], dtype=np.float64)
        sum_x2 = np.zeros(X.shape[-1], dtype=np.float64)
        count = np.zeros(X.shape[-1], dtype=np.float64)  # 每个特征的有效计数
        
        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[i:i+batch_size]
            batch_X = X[batch_idx].reshape(-1, X.shape[-1])
            # 处理NaN：使用nansum计算
            valid_mask = ~np.isnan(batch_X)
            sum_x += np.nansum(batch_X, axis=0)
            sum_x2 += np.nansum(batch_X ** 2, axis=0)
            count += np.sum(valid_mask, axis=0)
        
        # 计算均值和标准差
        X_mean = sum_x / np.maximum(count, 1)  # 避免除以0
        variance = sum_x2 / np.maximum(count, 1) - X_mean ** 2
        variance = np.maximum(variance, 0)  # 避免负数（数值误差）
        X_std = np.sqrt(variance)
        X_std[X_std == 0] = 1  # 标准差为0的特征不缩放
        X_std[np.isnan(X_std)] = 1  # NaN标准差设为1
        X_mean[np.isnan(X_mean)] = 0  # NaN均值设为0
        
        # 保存标准化参数
        from pipeline.shared.config import MODEL_CHECKPOINT_DIR
        np.save(MODEL_CHECKPOINT_DIR / "X_mean.npy", X_mean.astype(np.float32))
        np.save(MODEL_CHECKPOINT_DIR / "X_std.npy", X_std.astype(np.float32))
        
        # 分批标准化并保存各数据集
        def save_dataset(name: str, idx: np.ndarray):
            n_samples = len(idx)
            X_out = np.zeros((n_samples, X.shape[1], X.shape[2]), dtype=np.float32)
            y_out = np.zeros(n_samples, dtype=np.int8)
            
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch_idx = idx[i:end]
                batch_X = (X[batch_idx] - X_mean) / X_std
                # 将NaN替换为0
                batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
                X_out[i:end] = batch_X
                y_out[i:end] = y[batch_idx]
            
            np.savez(TRAIN_DATA_DIR / f"{name}.npz", X=X_out, y=y_out)
            return n_samples, np.mean(y_out)
        
        logger.info("Saving datasets...")
        n_train, pos_train = save_dataset("train", train_idx)
        n_val, pos_val = save_dataset("val", val_idx)
        n_test, pos_test = save_dataset("test", test_idx)
        
        logger.info(f"Train: {n_train:,}, Val: {n_val:,}, Test: {n_test:,}")
        logger.info(f"Positive ratio - Train: {pos_train:.3f}, Val: {pos_val:.3f}, Test: {pos_test:.3f}")
        
        # 清理临时memmap文件
        if temp_dir and temp_dir.exists():
            import shutil
            # 先关闭memmap（删除对X,y的引用）
            del X, y
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info("Done!")
        logger.info("Training data saved!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of symbols (for testing)")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    runner = FeatureEngineerRunner()
    runner.run(args.limit, args.workers)


if __name__ == "__main__":
    main()
