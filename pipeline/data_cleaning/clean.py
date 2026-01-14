"""
数据清洗模块 - Phase 1: Data Cleaning

负责从原始数据中提取、清洗和预处理数据
优化利用SSD和内存，使用流式处理
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import polars as pl
import tomllib

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.shared.config import (
    RAW_DATA_ROOT, CLEANED_DATA_DIR, CLEANING_CONFIG, PIPELINE_DATA_ROOT
)
from pipeline.shared.utils import (
    setup_logger, timer, save_parquet_optimized,
    check_dataframe_quality, get_file_size
)


logger = setup_logger("data_cleaning")


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.config = CLEANING_CONFIG
        self.stats = {
            "total_symbols": 0,
            "valid_symbols": 0,
            "total_records": 0,
            "cleaned_records": 0,
            "removed_outliers": 0,
            "removed_missing": 0,
        }
    
    @timer
    def load_index_constituents(self, index_codes: list[str] = None) -> list[str]:
        """加载指数成分股列表
        
        Args:
            index_codes: 指数代码列表，如 ["SHSE.000905", "SHSE.000852"]
                        默认使用中证500 + 中证1000
        """
        if index_codes is None:
            # 默认：中证500 + 中证1000 = 1500只股票
            index_codes = ["SHSE.000905", "SHSE.000852"]
        
        logger.info(f"Loading index constituents: {index_codes}")
        
        index_dir = RAW_DATA_ROOT / "meta" / "index"
        
        # 找到最新的日期目录
        date_dirs = sorted([d for d in index_dir.iterdir() if d.is_dir()], reverse=True)
        if not date_dirs:
            raise ValueError("No index data found")
        
        latest_date = date_dirs[0].name
        logger.info(f"Using index data from {latest_date}")
        
        all_symbols = set()
        for code in index_codes:
            toml_path = index_dir / latest_date / f"{code}.toml"
            if toml_path.exists():
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                    symbols = data.get("sec_ids", [])
                    all_symbols.update(symbols)
                    logger.info(f"  {code}: {len(symbols)} stocks")
            else:
                logger.warning(f"Index file not found: {toml_path}")
        
        symbols = sorted(list(all_symbols))
        logger.info(f"Total unique stocks: {len(symbols)}")
        self.stats["total_symbols"] = len(symbols)
        
        return symbols
    
    @timer
    def get_exclude_symbols(self, exclude_codes: list[str]) -> set[str]:
        """获取需要排除的指数成分股
        
        Args:
            exclude_codes: 需要排除的指数代码列表，如 ["SHSE.000300", "SHSE.000905"]
        """
        if not exclude_codes:
            return set()
        
        logger.info(f"Loading symbols to exclude: {exclude_codes}")
        
        index_dir = RAW_DATA_ROOT / "meta" / "index"
        date_dirs = sorted([d for d in index_dir.iterdir() if d.is_dir()], reverse=True)
        if not date_dirs:
            return set()
        
        latest_date = date_dirs[0].name
        
        exclude_symbols = set()
        for code in exclude_codes:
            toml_path = index_dir / latest_date / f"{code}.toml"
            if toml_path.exists():
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                    symbols = data.get("sec_ids", [])
                    exclude_symbols.update(symbols)
                    logger.info(f"  Exclude {code}: {len(symbols)} stocks")
            else:
                logger.warning(f"Index file not found: {toml_path}")
        
        logger.info(f"Total symbols to exclude: {len(exclude_symbols)}")
        return exclude_symbols
    
    @timer
    def load_instruments(self) -> pl.DataFrame:
        """加载证券列表（备用方法）"""
        logger.info("Loading instruments...")
        instruments_path = RAW_DATA_ROOT / "meta" / "instruments.parquet"
        
        df = pl.read_parquet(instruments_path)
        
        # 只保留股票（sec_type=1是股票）
        df = df.filter(
            (pl.col("exchange").is_in(["SHSE", "SZSE"])) &
            (pl.col("sec_type") == 1)
        )
        
        self.stats["total_symbols"] = df.height
        logger.info(f"Loaded {df.height} stock instruments")
        
        return df
    
    @timer
    def get_trading_days(self, start_date: str, end_date: str) -> list[str]:
        """获取交易日历"""
        logger.info(f"Loading trading days from {start_date} to {end_date}")
        
        trading_days_path = RAW_DATA_ROOT / "cfg" / "trading_days.toml"
        with open(trading_days_path, "rb") as f:
            data = tomllib.load(f)
        
        days = [d for d in data["trading_days"] 
                if start_date <= d <= end_date]
        
        logger.info(f"Found {len(days)} trading days")
        return days
    
    def load_mkline_data(self, symbol: str, date: str) -> pl.DataFrame | None:
        """加载单只股票单日的分钟K线数据"""
        mkline_path = RAW_DATA_ROOT / "mkline" / symbol / f"{date}.parquet"
        
        try:
            if not mkline_path.exists():
                return None
            df = pl.read_parquet(mkline_path)
            return df
        except Exception as e:
            # 网络存储可能出现临时错误，静默忽略
            return None
    
    def load_symbol_all_data(self, symbol: str, trading_days: set[str]) -> pl.DataFrame | None:
        """批量加载单只股票的所有数据（更高效）"""
        symbol_dir = RAW_DATA_ROOT / "mkline" / symbol
        
        try:
            if not symbol_dir.exists():
                return None
            
            # 获取该股票的所有parquet文件
            files = list(symbol_dir.glob("*.parquet"))
            if not files:
                return None
            
            # 过滤只读取在交易日范围内的文件
            valid_files = [f for f in files if f.stem in trading_days]
            if not valid_files:
                return None
            
            # 批量读取所有文件
            dfs = []
            for f in valid_files:
                try:
                    df = pl.read_parquet(f)
                    dfs.append(df)
                except:
                    continue
            
            if not dfs:
                return None
            
            return pl.concat(dfs)
        except Exception as e:
            return None
    
    def clean_single_day(self, symbol: str, date: str) -> pl.DataFrame | None:
        """清洗单日数据"""
        df = self.load_mkline_data(symbol, date)
        
        if df is None or df.height == 0:
            return None
        
        original_count = df.height
        
        # 1. 移除缺失值
        df = df.drop_nulls()
        self.stats["removed_missing"] += original_count - df.height
        
        # 2. 移除异常值（价格和成交量）
        # 价格异常：涨跌幅超过20%
        if df.height > 1:
            df = df.with_columns([
                (pl.col("close") / pl.col("close").shift(1) - 1).alias("_ret")
            ])
            
            outlier_mask = (
                (pl.col("_ret").abs() > 0.2) |
                (pl.col("volume") <= 0) |
                (pl.col("turnover") <= 0)
            )
            
            removed = df.filter(outlier_mask).height
            df = df.filter(~outlier_mask).drop("_ret")
            self.stats["removed_outliers"] += removed
        
        # 3. 确保数据完整性
        required_cols = ["symbol", "open", "high", "low", "close", 
                        "volume", "turnover", "date", "time"]
        
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Missing required columns for {symbol} on {date}")
            return None
        
        # 4. 确保价格逻辑正确
        df = df.filter(
            (pl.col("high") >= pl.col("low")) &
            (pl.col("high") >= pl.col("close")) &
            (pl.col("high") >= pl.col("open")) &
            (pl.col("low") <= pl.col("close")) &
            (pl.col("low") <= pl.col("open"))
        )
        
        if df.height == 0:
            return None
        
        self.stats["cleaned_records"] += df.height
        
        return df
    
    def clean_symbol(self, symbol: str, trading_days: list[str]) -> bool:
        """清洗单只股票的所有数据（优化版：批量加载）"""
        trading_days_set = set(trading_days)
        
        # 批量加载所有数据
        df = self.load_symbol_all_data(symbol, trading_days_set)
        if df is None or df.height == 0:
            return False
        
        original_count = df.height
        
        # 1. 移除缺失值
        df = df.drop_nulls()
        self.stats["removed_missing"] += original_count - df.height
        
        # 2. 确保数据完整性
        required_cols = ["symbol", "open", "high", "low", "close", 
                        "volume", "turnover", "date", "time"]
        
        if not all(col in df.columns for col in required_cols):
            return False
        
        # 3. 移除异常值（价格和成交量）
        df = df.filter(
            (pl.col("volume") > 0) &
            (pl.col("turnover") > 0)
        )
        
        # 4. 确保价格逻辑正确
        df = df.filter(
            (pl.col("high") >= pl.col("low")) &
            (pl.col("high") >= pl.col("close")) &
            (pl.col("high") >= pl.col("open")) &
            (pl.col("low") <= pl.col("close")) &
            (pl.col("low") <= pl.col("open"))
        )
        
        self.stats["removed_outliers"] += original_count - df.height - self.stats["removed_missing"]
        
        if df.height == 0:
            return False
        
        # 5. 按时间排序
        df = df.sort(["date", "time"])
        
        # 6. 检查数据充足性（至少有足够的交易日）
        n_days = df.select("date").unique().height
        if n_days < self.config["min_trading_days"]:
            return False
        
        self.stats["cleaned_records"] += df.height
        
        # 保存清洗后的数据
        output_path = CLEANED_DATA_DIR / f"{symbol}.parquet"
        save_parquet_optimized(df, output_path)
        
        self.stats["valid_symbols"] += 1
        return True
    
    @timer
    def run(self, start_date: str, end_date: str, 
            symbol_limit: int | None = None,
            use_index: bool = True,
            index_codes: list[str] | None = None,
            exclude_codes: list[str] | None = None):
        """执行完整的数据清洗流程
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbol_limit: 限制股票数量（测试用）
            use_index: 是否使用指数成分股（默认True）
            index_codes: 指数代码列表（默认中证500+中证1000）
            exclude_codes: 需要排除的指数代码列表
        """
        logger.info("="*60)
        logger.info("Starting Data Cleaning Pipeline")
        logger.info("="*60)
        
        # 1. 加载股票列表
        if use_index:
            symbols = self.load_index_constituents(index_codes)
        else:
            instruments = self.load_instruments()
            symbols = instruments["symbol"].to_list()
        
        # 排除指定指数的成分股
        if exclude_codes:
            exclude_symbols = self.get_exclude_symbols(exclude_codes)
            original_count = len(symbols)
            symbols = [s for s in symbols if s not in exclude_symbols]
            logger.info(f"After excluding: {original_count} -> {len(symbols)} stocks")
        
        if symbol_limit:
            symbols = symbols[:symbol_limit]
            logger.info(f"Limited to {symbol_limit} symbols for testing")
        
        # 检查已处理的股票，跳过它们（实现断点续传）
        cleaned_dir = PIPELINE_DATA_ROOT / "cleaned"
        already_cleaned = set()
        if cleaned_dir.exists():
            already_cleaned = {p.stem for p in cleaned_dir.glob("*.parquet")}
        
        symbols_to_process = [s for s in symbols if s not in already_cleaned]
        logger.info(f"Already cleaned: {len(already_cleaned)}, remaining: {len(symbols_to_process)}")
        
        if not symbols_to_process:
            logger.info("All symbols already cleaned!")
            self._print_summary()
            return
        
        # 2. 获取交易日历
        trading_days = self.get_trading_days(start_date, end_date)
        
        # 3. 使用多线程并行清洗数据
        num_workers = 8  # 网络I/O密集型，可以用较多线程
        logger.info(f"Processing {len(symbols_to_process)} symbols with {num_workers} workers...")
        
        # 线程安全的计数器
        success_count = 0
        processed_count = 0
        lock = threading.Lock()
        
        def process_symbol(symbol):
            nonlocal success_count, processed_count
            try:
                result = self.clean_symbol(symbol, trading_days)
                with lock:
                    processed_count += 1
                    if result:
                        success_count += 1
                    if processed_count % 50 == 0:
                        logger.info(f"Progress: {processed_count}/{len(symbols_to_process)} ({success_count} success)")
                return result
            except Exception as e:
                with lock:
                    processed_count += 1
                return False
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_symbol, s): s for s in symbols_to_process}
            for future in as_completed(futures):
                pass  # 结果已经在 process_symbol 中处理
        
        logger.info(f"Completed: {success_count}/{len(symbols_to_process)} success")
        
        # 4. 输出统计报告
        self._print_summary()
    
    def _print_summary(self):
        """打印清洗总结"""
        logger.info("\n" + "="*60)
        logger.info("Data Cleaning Summary")
        logger.info("="*60)
        logger.info(f"Total symbols scanned: {self.stats['total_symbols']}")
        logger.info(f"Valid symbols: {self.stats['valid_symbols']}")
        logger.info(f"Success rate: {self.stats['valid_symbols']/self.stats['total_symbols']*100:.1f}%")
        logger.info(f"Total records cleaned: {self.stats['cleaned_records']:,}")
        logger.info(f"Removed missing: {self.stats['removed_missing']:,}")
        logger.info(f"Removed outliers: {self.stats['removed_outliers']:,}")
        
        total_size = get_file_size(CLEANED_DATA_DIR)
        logger.info(f"Total cleaned data size: {total_size:.2f}MB")
        logger.info("="*60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Cleaning Pipeline")
    parser.add_argument("--start_date", type=str, default="2024-06-18",
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-01-13",
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of symbols (for testing)")
    parser.add_argument("--all_stocks", action="store_true",
                       help="Use all stocks instead of index constituents")
    parser.add_argument("--index", type=str, nargs="+", 
                       default=["SHSE.000905", "SHSE.000852"],
                       help="Index codes to use (default: CSI500 + CSI1000)")
    parser.add_argument("--exclude", type=str, nargs="+", default=None,
                       help="Index codes to exclude (e.g. SHSE.000300 SHSE.000905)")
    
    args = parser.parse_args()
    
    cleaner = DataCleaner()
    cleaner.run(
        args.start_date, 
        args.end_date, 
        args.limit,
        use_index=not args.all_stocks,
        index_codes=args.index,
        exclude_codes=args.exclude
    )


if __name__ == "__main__":
    main()
