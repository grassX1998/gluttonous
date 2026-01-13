"""
按日期预处理数据

将每天中证1000成分股的分钟K线数据合并到一个文件，同时计算日线指标。
输出目录: /data/backtest_1/{date}/
  - mkline.parquet: 当天所有成分股的分钟K线
  - dkline.parquet: 当天所有成分股的日线指标（前一日成交量等）
"""

import os
import sys
import tomllib
from pathlib import Path
from datetime import datetime, timedelta

import polars as pl

# 数据路径
DATA_ROOT = Path("/data/stock/gm")
OUTPUT_ROOT = Path("/data/backtest_1")
INDEX_CODE = "SHSE.000852"  # 中证1000


def get_trading_days(start_date: str, end_date: str) -> list[str]:
    """获取交易日列表"""
    cfg_path = DATA_ROOT / "cfg" / "trading_days.toml"
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    
    all_days = sorted(data["trading_days"])
    return [d for d in all_days if start_date <= d <= end_date]


def get_index_constituents(date: str) -> list[str]:
    """获取某一天的中证1000成分股"""
    index_path = DATA_ROOT / "meta" / "index" / date / f"{INDEX_CODE}.toml"
    if not index_path.exists():
        return []
    
    with open(index_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("sec_ids", [])


def get_prev_trading_day(date: str, trading_days: list[str]) -> str | None:
    """获取前一个交易日"""
    try:
        idx = trading_days.index(date)
        if idx > 0:
            return trading_days[idx - 1]
    except ValueError:
        pass
    return None


def process_date(date: str, constituents: list[str], prev_date: str | None) -> bool:
    """
    处理单个交易日的数据
    
    Args:
        date: 当前日期
        constituents: 当天的成分股列表
        prev_date: 前一交易日（用于计算前日成交量）
    
    Returns:
        是否处理成功
    """
    output_dir = OUTPUT_ROOT / date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mkline_path = output_dir / "mkline.parquet"
    dkline_path = output_dir / "dkline.parquet"
    
    # 如果已经处理过，跳过
    if mkline_path.exists() and dkline_path.exists():
        return True
    
    mkline_dfs = []
    dkline_records = []
    
    for symbol in constituents:
        # 读取当天分钟K线
        mkline_file = DATA_ROOT / "mkline" / symbol / f"{date}.parquet"
        if not mkline_file.exists():
            continue
        
        try:
            df = pl.read_parquet(mkline_file)
            mkline_dfs.append(df)
        except Exception as e:
            print(f"Error reading mkline {mkline_file}: {e}")
            continue
        
        # 计算前一交易日的成交量（用于量比计算）
        if prev_date:
            prev_mkline_file = DATA_ROOT / "mkline" / symbol / f"{prev_date}.parquet"
            if prev_mkline_file.exists():
                try:
                    prev_df = pl.read_parquet(prev_mkline_file)
                    # 计算全天成交量
                    prev_volume = prev_df["volume"].sum()
                    dkline_records.append({
                        "symbol": symbol,
                        "date_str": date,
                        "prev_date": prev_date,
                        "prev_volume": prev_volume,
                    })
                except Exception as e:
                    print(f"Error reading prev mkline {prev_mkline_file}: {e}")
    
    if not mkline_dfs:
        print(f"No data for {date}")
        return False
    
    # 合并分钟K线
    mkline_combined = pl.concat(mkline_dfs)
    # 添加 date_str 列（如果没有的话）
    if "date_str" not in mkline_combined.columns:
        mkline_combined = mkline_combined.with_columns(
            pl.col("date").cast(pl.Utf8).alias("date_str")
        )
    mkline_combined.write_parquet(mkline_path)
    
    # 保存日线指标
    if dkline_records:
        dkline_df = pl.DataFrame(dkline_records)
        dkline_df.write_parquet(dkline_path)
    else:
        # 创建空的 dkline
        dkline_df = pl.DataFrame({
            "symbol": pl.Series([], dtype=pl.Utf8),
            "date_str": pl.Series([], dtype=pl.Utf8),
            "prev_date": pl.Series([], dtype=pl.Utf8),
            "prev_volume": pl.Series([], dtype=pl.Int64),
        })
        dkline_df.write_parquet(dkline_path)
    
    return True


def main():
    start_date = "2025-01-01"
    end_date = "2026-01-10"
    
    print(f"Processing data from {start_date} to {end_date}")
    
    # 获取交易日列表
    trading_days = get_trading_days(start_date, end_date)
    print(f"Found {len(trading_days)} trading days")
    
    # 处理每个交易日
    success_count = 0
    total_count = len(trading_days)
    
    for i, date in enumerate(trading_days):
        # 获取当天成分股
        constituents = get_index_constituents(date)
        if not constituents:
            print(f"[{i+1}/{total_count}] {date}: No constituents found, skipping")
            continue
        
        # 获取前一交易日
        prev_date = get_prev_trading_day(date, trading_days)
        
        # 处理数据
        success = process_date(date, constituents, prev_date)
        if success:
            success_count += 1
            print(f"[{i+1}/{total_count}] {date}: Processed ({len(constituents)} stocks)")
        else:
            print(f"[{i+1}/{total_count}] {date}: Failed")
    
    print(f"\nDone! Processed {success_count}/{total_count} trading days")


if __name__ == "__main__":
    main()
