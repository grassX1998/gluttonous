"""
AKShare 数据采集模块

提供 AKShare 免费数据源的封装，作为掘金付费数据的替代方案。
本模块独立运行，不与现有 collector.py 每日任务耦合。

支持数据类型：
- 申万行业分类（一/二/三级）
- 行业成分股
- 财务报表（资产负债表、利润表、现金流量表）
- 财务关键指标
- 历史日K线（东财）- 支持按股票/按日期两种存储方式
- 分红历史

存储结构：
- 日K线按股票存储：hist/daily/by_symbol/{code}.parquet（推荐采集方式）
- 日K线按日期存储：hist/daily/{YYYYMMDD}.parquet（可从按股票转换）
- 其他数据按标的存储：{type}/{code}.parquet

使用方式：
    # 测试 AKShare 连接
    python -m pipeline.data_collection.akshare_api --test

    # 采集申万行业分类
    python -m pipeline.data_collection.akshare_api --industry

    # 采集指定股票财务数据
    python -m pipeline.data_collection.akshare_api --finance --symbol 600000

    # ⭐⭐⭐ 批量采集所有股票完整历史日K线（推荐，快100倍）
    python -m pipeline.data_collection.akshare_api --hist-batch
    python -m pipeline.data_collection.akshare_api --hist-batch --start 2015-01-01

    # 将按股票存储的数据转换为按日期存储
    python -m pipeline.data_collection.akshare_api --convert-daily

    # 采集单个日期所有股票（较慢，不推荐用于大范围采集）
    python -m pipeline.data_collection.akshare_api --hist-all --date 2026-01-17

    # 采集单只股票历史日K线
    python -m pipeline.data_collection.akshare_api --hist --symbol 600000 --start 2024-01-01 --end 2026-01-17
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl

try:
    import akshare as ak
except ImportError:
    ak = None

from pipeline.shared.config import (
    AKSHARE_DATA_ROOT,
    AKSHARE_INDUSTRY_DIR,
    AKSHARE_FINANCE_DIR,
    AKSHARE_DIVIDEND_DIR,
    AKSHARE_HIST_DIR,
    AKSHARE_INDEX_DIR,
)
from pipeline.shared.logging_config import get_akshare_logger

# 使用统一日志配置，输出到 logs/{date}/akshare_collection.log
logger = get_akshare_logger()


def _check_akshare() -> None:
    """检查 AKShare 是否已安装"""
    if ak is None:
        raise ImportError(
            "AKShare 未安装，请先安装: pip install akshare --upgrade"
        )


def _convert_symbol(symbol: str) -> str:
    """
    将标准股票代码转换为 AKShare 格式

    Args:
        symbol: 标准代码，如 SHSE.600000 或 600000

    Returns:
        AKShare 格式代码，如 600000
    """
    if "." in symbol:
        return symbol.split(".")[-1]
    return symbol


# =====================================================
# 申万行业分类采集函数
# =====================================================

def load_sw_industry() -> bool:
    """
    采集申万行业分类（一/二/三级）

    数据保存到:
        - {AKSHARE_INDUSTRY_DIR}/sw_level1.parquet
        - {AKSHARE_INDUSTRY_DIR}/sw_level2.parquet
        - {AKSHARE_INDUSTRY_DIR}/sw_level3.parquet

    Returns:
        是否成功
    """
    _check_akshare()

    AKSHARE_INDUSTRY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 采集申万一级行业
        logger.info("采集申万一级行业分类...")
        df_level1 = ak.sw_index_first_info()
        if df_level1 is not None and not df_level1.empty:
            pl_df = pl.from_pandas(df_level1)
            pl_df.write_parquet(AKSHARE_INDUSTRY_DIR / "sw_level1.parquet")
            logger.info(f"申万一级行业已保存，共 {len(pl_df)} 个行业")

        # 采集申万二级行业
        logger.info("采集申万二级行业分类...")
        df_level2 = ak.sw_index_second_info()
        if df_level2 is not None and not df_level2.empty:
            pl_df = pl.from_pandas(df_level2)
            pl_df.write_parquet(AKSHARE_INDUSTRY_DIR / "sw_level2.parquet")
            logger.info(f"申万二级行业已保存，共 {len(pl_df)} 个行业")

        # 采集申万三级行业
        logger.info("采集申万三级行业分类...")
        df_level3 = ak.sw_index_third_info()
        if df_level3 is not None and not df_level3.empty:
            pl_df = pl.from_pandas(df_level3)
            pl_df.write_parquet(AKSHARE_INDUSTRY_DIR / "sw_level3.parquet")
            logger.info(f"申万三级行业已保存，共 {len(pl_df)} 个行业")

        return True

    except Exception as e:
        logger.error(f"采集申万行业分类失败: {e}")
        return False


def load_sw_constituents(index_code: str) -> bool:
    """
    采集申万行业成分股

    Args:
        index_code: 行业代码，如 801010（农林牧渔）

    Returns:
        是否成功
    """
    _check_akshare()

    constituents_dir = AKSHARE_INDUSTRY_DIR / "sw_constituents"
    constituents_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = constituents_dir / f"{index_code}.parquet"

    try:
        df = ak.sw_index_cons(index_code)
        if df is not None and not df.empty:
            pl_df = pl.from_pandas(df)
            pl_df.write_parquet(parquet_file)
            logger.info(f"行业 {index_code} 成分股已保存，共 {len(pl_df)} 只")
            return True
        else:
            logger.warning(f"行业 {index_code} 成分股数据为空")
            return True

    except Exception as e:
        logger.error(f"采集行业 {index_code} 成分股失败: {e}")
        return False


def load_all_sw_constituents(delay: float = 0.5) -> Tuple[int, int]:
    """
    采集所有申万行业成分股

    Args:
        delay: 请求间隔（秒），避免限流

    Returns:
        (成功数, 失败数) 元组
    """
    _check_akshare()

    # 先确保行业分类已采集
    level1_file = AKSHARE_INDUSTRY_DIR / "sw_level1.parquet"
    if not level1_file.exists():
        load_sw_industry()

    # 读取行业列表
    try:
        df = pl.read_parquet(level1_file)
        # 假设行业代码列名为 "行业代码" 或 "index_code"
        code_col = None
        for col in ["行业代码", "index_code", "code"]:
            if col in df.columns:
                code_col = col
                break

        if code_col is None:
            logger.error(f"无法识别行业代码列，可用列: {df.columns}")
            return 0, 0

        index_codes = df[code_col].to_list()
    except Exception as e:
        logger.error(f"读取行业列表失败: {e}")
        return 0, 0

    success_count = 0
    fail_count = 0

    for code in index_codes:
        if load_sw_constituents(str(code)):
            success_count += 1
        else:
            fail_count += 1
        time.sleep(delay)

    logger.info(f"行业成分股采集完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


# =====================================================
# 财务数据采集函数
# =====================================================

def load_financial_report(symbol: str, report_type: str = "all") -> bool:
    """
    采集新浪财务报表

    Args:
        symbol: 股票代码，如 600000 或 SHSE.600000
        report_type: 报表类型，可选 balance/income/cashflow/all

    Returns:
        是否成功
    """
    _check_akshare()

    code = _convert_symbol(symbol)
    AKSHARE_FINANCE_DIR.mkdir(parents=True, exist_ok=True)

    report_types = ["balance", "income", "cashflow"] if report_type == "all" else [report_type]
    report_map = {
        "balance": ("资产负债表", "balance"),
        "income": ("利润表", "income"),
        "cashflow": ("现金流量表", "cashflow"),
    }

    success = True

    for rt in report_types:
        if rt not in report_map:
            logger.warning(f"未知报表类型: {rt}")
            continue

        sina_type, dir_name = report_map[rt]
        report_dir = AKSHARE_FINANCE_DIR / dir_name
        report_dir.mkdir(parents=True, exist_ok=True)

        parquet_file = report_dir / f"{code}.parquet"

        try:
            df = ak.stock_financial_report_sina(stock=code, symbol=sina_type)
            if df is not None and not df.empty:
                pl_df = pl.from_pandas(df)
                pl_df.write_parquet(parquet_file)
                logger.info(f"{code} {sina_type}已保存，共 {len(pl_df)} 条记录")
            else:
                logger.warning(f"{code} {sina_type}数据为空")

        except Exception as e:
            logger.error(f"采集 {code} {sina_type}失败: {e}")
            success = False

    return success


def load_financial_abstract(symbol: str) -> bool:
    """
    采集财务关键指标摘要

    Args:
        symbol: 股票代码，如 600000

    Returns:
        是否成功
    """
    _check_akshare()

    code = _convert_symbol(symbol)
    abstract_dir = AKSHARE_FINANCE_DIR / "abstract"
    abstract_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = abstract_dir / f"{code}.parquet"

    try:
        df = ak.stock_financial_abstract(stock=code)
        if df is not None and not df.empty:
            pl_df = pl.from_pandas(df)
            pl_df.write_parquet(parquet_file)
            logger.info(f"{code} 财务摘要已保存，共 {len(pl_df)} 条记录")
            return True
        else:
            logger.warning(f"{code} 财务摘要数据为空")
            return True

    except Exception as e:
        logger.error(f"采集 {code} 财务摘要失败: {e}")
        return False


def load_finance_eastmoney(symbol: str, report_type: str = "all") -> bool:
    """
    采集东方财富财务报表（另一数据源）

    Args:
        symbol: 股票代码，如 600000
        report_type: 报表类型，可选 balance/income/cashflow/all

    Returns:
        是否成功
    """
    _check_akshare()

    code = _convert_symbol(symbol)
    em_dir = AKSHARE_FINANCE_DIR / "eastmoney"
    em_dir.mkdir(parents=True, exist_ok=True)

    report_types = ["balance", "income", "cashflow"] if report_type == "all" else [report_type]
    func_map = {
        "balance": (ak.stock_zcfz_em, "zcfz"),
        "income": (ak.stock_lrb_em, "lrb"),
        "cashflow": (ak.stock_xjll_em, "xjll"),
    }

    success = True

    for rt in report_types:
        if rt not in func_map:
            continue

        func, suffix = func_map[rt]
        parquet_file = em_dir / f"{code}_{suffix}.parquet"

        try:
            df = func(symbol=code)
            if df is not None and not df.empty:
                pl_df = pl.from_pandas(df)
                pl_df.write_parquet(parquet_file)
                logger.info(f"{code} 东财{rt}已保存，共 {len(pl_df)} 条记录")
            else:
                logger.warning(f"{code} 东财{rt}数据为空")

        except Exception as e:
            logger.error(f"采集 {code} 东财{rt}失败: {e}")
            success = False

    return success


# =====================================================
# 历史K线数据采集函数
# =====================================================

def get_all_a_stock_codes(include_delisted: bool = True) -> List[str]:
    """
    获取所有 A 股股票代码列表（包含已退市股票）

    Args:
        include_delisted: 是否包含已退市股票，默认 True

    Returns:
        股票代码列表，如 ['000001', '000002', ...]
    """
    _check_akshare()

    codes = set()

    try:
        # 1. 获取当前上市的股票（使用实时行情接口，约5800只）
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            code_col = None
            for col in ["代码", "code"]:
                if col in df.columns:
                    code_col = col
                    break
            if code_col:
                current_codes = df[code_col].tolist()
                codes.update(current_codes)
                logger.info(f"当前上市股票: {len(current_codes)} 只")

        # 2. 获取已退市股票
        if include_delisted:
            # 上交所退市
            try:
                df_sh = ak.stock_info_sh_delist()
                if df_sh is not None and not df_sh.empty:
                    code_col = None
                    for col in ["公司代码", "code", "证券代码"]:
                        if col in df_sh.columns:
                            code_col = col
                            break
                    if code_col:
                        sh_codes = df_sh[code_col].astype(str).tolist()
                        codes.update(sh_codes)
                        logger.info(f"上交所退市股票: {len(sh_codes)} 只")
            except Exception as e:
                logger.warning(f"获取上交所退市股票失败: {e}")

            # 深交所退市
            try:
                df_sz = ak.stock_info_sz_delist()
                if df_sz is not None and not df_sz.empty:
                    code_col = None
                    for col in ["证券代码", "code", "公司代码"]:
                        if col in df_sz.columns:
                            code_col = col
                            break
                    if code_col:
                        sz_codes = df_sz[code_col].astype(str).tolist()
                        codes.update(sz_codes)
                        logger.info(f"深交所退市股票: {len(sz_codes)} 只")
            except Exception as e:
                logger.warning(f"获取深交所退市股票失败: {e}")

        codes_list = sorted(list(codes))
        logger.info(f"股票总数: {len(codes_list)} 只 (含退市: {include_delisted})")
        return codes_list

    except Exception as e:
        logger.error(f"获取 A 股列表失败: {e}")
        return []


def load_hist_daily_by_date(
    date: str,
    adjust: str = "qfq",
    batch_size: int = 100,
    delay: float = 0.1,
    include_delisted: bool = True,
) -> Tuple[int, int]:
    """
    采集指定日期所有股票的日K线（按日期存储）

    工作原理：
    1. 获取所有 A 股代码列表（约6100只，含已退市股票）
    2. 逐一查询每只股票在指定日期的数据
    3. 没有数据的自动跳过（停牌/未上市/当日不存在）
    4. 最终保存的文件只包含当天实际有交易的股票

    存储位置：{AKSHARE_HIST_DIR}/daily/{YYYYMMDD}.parquet

    Args:
        date: 日期，格式 YYYY-MM-DD
        adjust: 复权类型，qfq=前复权, hfq=后复权, 空=不复权
        batch_size: 每批处理股票数（用于进度显示）
        delay: 请求间隔（秒），避免限流
        include_delisted: 是否包含已退市股票，默认 True

    Returns:
        (成功数, 失败数) 元组
    """
    _check_akshare()

    daily_dir = AKSHARE_HIST_DIR / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    date_fmt = date.replace("-", "")
    date_str = date.replace("-", "")  # 用于文件名
    parquet_file = daily_dir / f"{date_str}.parquet"

    # 获取所有 A 股代码（含已退市股票）
    all_codes = get_all_a_stock_codes(include_delisted=include_delisted)
    if not all_codes:
        logger.error("无法获取股票列表")
        return 0, 0

    total = len(all_codes)
    success_count = 0
    fail_count = 0
    all_data = []

    logger.info(f"开始采集 {date} 日K线，共 {total} 只股票")

    for i, code in enumerate(all_codes):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=date_fmt,
                end_date=date_fmt,
                adjust=adjust,
            )

            if df is not None and not df.empty:
                # 添加股票代码列
                df["symbol"] = code
                all_data.append(df)
                success_count += 1
            # 空数据不算失败（可能停牌）

        except Exception as e:
            fail_count += 1
            if fail_count <= 10:  # 只记录前10个错误
                logger.warning(f"采集 {code} 失败: {e}")

        # 进度显示
        if (i + 1) % batch_size == 0:
            logger.info(
                f"进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) | "
                f"成功: {success_count} | 失败: {fail_count}"
            )

        # 限流
        if delay > 0:
            time.sleep(delay)

    # 合并并保存
    if all_data:
        import pandas as pd

        merged_df = pd.concat(all_data, ignore_index=True)
        pl_df = pl.from_pandas(merged_df)
        pl_df.write_parquet(parquet_file)
        logger.info(
            f"{date} 日K线采集完成: {len(pl_df)} 条记录, "
            f"成功 {success_count}, 失败 {fail_count}"
        )
    else:
        logger.warning(f"{date} 无有效数据")

    return success_count, fail_count


def load_hist_daily_range(
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    batch_size: int = 100,
    delay: float = 0.1,
) -> Tuple[int, int]:
    """
    采集日期范围内所有股票的日K线（按日期存储）

    Args:
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        adjust: 复权类型
        batch_size: 每批处理股票数
        delay: 请求间隔（秒）

    Returns:
        (总成功数, 总失败数) 元组
    """
    from datetime import datetime, timedelta

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total_success = 0
    total_fail = 0
    current = start

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        # 跳过周末
        if current.weekday() < 5:
            logger.info(f"========== 采集 {date_str} ==========")
            success, fail = load_hist_daily_by_date(
                date_str, adjust, batch_size, delay
            )
            total_success += success
            total_fail += fail

        current += timedelta(days=1)

    logger.info(
        f"日期范围采集完成: {start_date} 至 {end_date}, "
        f"总成功 {total_success}, 总失败 {total_fail}"
    )
    return total_success, total_fail


def load_hist_daily(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> bool:
    """
    采集单只股票的历史日K线（保留旧接口兼容性，存储到 by_symbol 目录）

    Args:
        symbol: 股票代码，如 600000
        start_date: 开始日期，格式 YYYYMMDD 或 YYYY-MM-DD
        end_date: 结束日期，格式 YYYYMMDD 或 YYYY-MM-DD
        adjust: 复权类型，qfq=前复权, hfq=后复权, 空=不复权

    Returns:
        是否成功
    """
    _check_akshare()

    code = _convert_symbol(symbol)
    # 按标的存储放到 by_symbol 子目录
    daily_dir = AKSHARE_HIST_DIR / "daily" / "by_symbol"
    daily_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = daily_dir / f"{code}.parquet"

    # 格式化日期
    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_fmt,
            end_date=end_fmt,
            adjust=adjust,
        )

        if df is not None and not df.empty:
            pl_df = pl.from_pandas(df)

            # 如果文件存在，合并数据
            if parquet_file.exists():
                existing_df = pl.read_parquet(parquet_file)
                # 假设有日期列
                date_col = None
                for col in ["日期", "date", "trade_date"]:
                    if col in pl_df.columns:
                        date_col = col
                        break

                if date_col:
                    pl_df = pl.concat([existing_df, pl_df]).unique(subset=[date_col])
                else:
                    pl_df = pl.concat([existing_df, pl_df]).unique()

            pl_df.write_parquet(parquet_file)
            logger.info(f"{code} 日K线已保存，共 {len(pl_df)} 条记录")
            return True
        else:
            logger.warning(f"{code} 日K线数据为空")
            return True

    except Exception as e:
        logger.error(f"采集 {code} 日K线失败: {e}")
        return False


def load_all_hist_daily(
    start_date: str = "19910101",
    end_date: str = None,
    adjust: str = "qfq",
    delay: float = 0.5,
    include_delisted: bool = True,
    batch_size: int = 100,
) -> Tuple[int, int]:
    """
    批量采集所有股票的完整历史日K线（按股票存储，高效方式）

    每只股票一次性拉取全部历史数据，比按日期采集快100倍以上。
    数据存储到 {AKSHARE_HIST_DIR}/daily/by_symbol/{code}.parquet

    采集完成后，可用 convert_hist_to_daily() 转换为按日期存储。

    Args:
        start_date: 开始日期，格式 YYYYMMDD，默认 19910101
        end_date: 结束日期，格式 YYYYMMDD，默认今天
        adjust: 复权类型，qfq=前复权, hfq=后复权, 空=不复权
        delay: 请求间隔（秒），避免限流，默认 0.5
        include_delisted: 是否包含已退市股票，默认 True
        batch_size: 进度显示批次大小

    Returns:
        (成功数, 失败数) 元组
    """
    _check_akshare()

    from datetime import datetime

    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")

    daily_dir = AKSHARE_HIST_DIR / "daily" / "by_symbol"
    daily_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有股票代码
    all_codes = get_all_a_stock_codes(include_delisted=include_delisted)
    if not all_codes:
        logger.error("无法获取股票列表")
        return 0, 0

    total = len(all_codes)
    success_count = 0
    fail_count = 0
    skip_count = 0

    start_fmt = start_date.replace("-", "")
    end_fmt = end_date.replace("-", "")

    logger.info(f"开始批量采集日K线: {start_fmt} ~ {end_fmt}")
    logger.info(f"共 {total} 只股票，预计需要 {total * delay / 3600:.1f} 小时")

    for i, code in enumerate(all_codes):
        parquet_file = daily_dir / f"{code}.parquet"

        # 如果已存在，检查是否需要更新
        if parquet_file.exists():
            try:
                existing_df = pl.read_parquet(parquet_file)
                # 找日期列
                date_col = None
                for col in ["日期", "date", "trade_date"]:
                    if col in existing_df.columns:
                        date_col = col
                        break
                if date_col:
                    # 获取已有数据的最新日期
                    max_date = existing_df[date_col].max()
                    if isinstance(max_date, str):
                        max_date_str = max_date.replace("-", "")
                    else:
                        max_date_str = max_date.strftime("%Y%m%d")

                    # 如果已有数据到今天，跳过
                    if max_date_str >= end_fmt:
                        skip_count += 1
                        if (i + 1) % batch_size == 0:
                            logger.info(
                                f"进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) | "
                                f"成功: {success_count} | 失败: {fail_count} | 跳过: {skip_count}"
                            )
                        continue
                    # 否则只采集增量
                    start_fmt = max_date_str
            except Exception:
                pass  # 文件损坏，重新采集

        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_fmt,
                end_date=end_fmt,
                adjust=adjust,
            )

            if df is not None and not df.empty:
                # 添加股票代码列
                df["symbol"] = code
                pl_df = pl.from_pandas(df)

                # 如果已有数据，合并
                if parquet_file.exists():
                    try:
                        existing_df = pl.read_parquet(parquet_file)
                        # 确保existing_df也有symbol列
                        if "symbol" not in existing_df.columns:
                            existing_df = existing_df.with_columns(pl.lit(code).alias("symbol"))
                        # 找日期列去重
                        date_col = None
                        for col in ["日期", "date", "trade_date"]:
                            if col in pl_df.columns:
                                date_col = col
                                break
                        if date_col:
                            pl_df = pl.concat([existing_df, pl_df]).unique(subset=[date_col, "symbol"])
                        else:
                            pl_df = pl.concat([existing_df, pl_df]).unique()
                    except Exception:
                        pass  # 合并失败，直接覆盖

                pl_df.write_parquet(parquet_file)
                success_count += 1
            else:
                # 无数据（可能是退市股票在该时间段无数据）
                skip_count += 1

        except Exception as e:
            fail_count += 1
            if fail_count <= 20:
                logger.warning(f"采集 {code} 失败: {e}")

        # 进度显示
        if (i + 1) % batch_size == 0:
            logger.info(
                f"进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) | "
                f"成功: {success_count} | 失败: {fail_count} | 跳过: {skip_count}"
            )

        # 重置start_fmt为原始值（因为增量采集可能修改了）
        start_fmt = start_date.replace("-", "")

        # 限流
        if delay > 0:
            time.sleep(delay)

    logger.info(
        f"批量采集完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skip_count}"
    )
    return success_count, fail_count


def convert_hist_to_daily(output_dir: Path = None) -> int:
    """
    将按股票存储的日K线转换为按日期存储

    读取 {AKSHARE_HIST_DIR}/daily/by_symbol/*.parquet
    输出到 {AKSHARE_HIST_DIR}/daily/{YYYYMMDD}.parquet

    Args:
        output_dir: 输出目录，默认 {AKSHARE_HIST_DIR}/daily/

    Returns:
        生成的日期文件数量
    """
    import gc

    by_symbol_dir = AKSHARE_HIST_DIR / "daily" / "by_symbol"
    if output_dir is None:
        output_dir = AKSHARE_HIST_DIR / "daily"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not by_symbol_dir.exists():
        logger.error(f"源目录不存在: {by_symbol_dir}")
        return 0

    # 读取所有股票数据
    logger.info("读取所有股票数据...")
    all_files = list(by_symbol_dir.glob("*.parquet"))
    logger.info(f"共 {len(all_files)} 个股票文件")

    # 分批读取并合并
    all_data = []
    batch_size = 500

    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i : i + batch_size]
        batch_dfs = []

        for f in batch_files:
            try:
                df = pl.read_parquet(f)
                # 确保有symbol列
                if "symbol" not in df.columns:
                    code = f.stem
                    df = df.with_columns(pl.lit(code).alias("symbol"))
                batch_dfs.append(df)
            except Exception as e:
                logger.warning(f"读取 {f} 失败: {e}")

        if batch_dfs:
            batch_merged = pl.concat(batch_dfs)
            all_data.append(batch_merged)
            logger.info(f"已读取 {min(i + batch_size, len(all_files))}/{len(all_files)} 个文件")

        # 释放内存
        del batch_dfs
        gc.collect()

    if not all_data:
        logger.error("无有效数据")
        return 0

    # 合并所有数据
    logger.info("合并数据...")
    merged_df = pl.concat(all_data)
    del all_data
    gc.collect()

    logger.info(f"总记录数: {len(merged_df)}")

    # 找日期列
    date_col = None
    for col in ["日期", "date", "trade_date"]:
        if col in merged_df.columns:
            date_col = col
            break

    if date_col is None:
        logger.error(f"无法找到日期列，可用列: {merged_df.columns}")
        return 0

    # 按日期分组保存
    logger.info("按日期分组保存...")

    # 获取所有日期
    dates = merged_df[date_col].unique().sort().to_list()
    logger.info(f"共 {len(dates)} 个交易日")

    count = 0
    for i, date_val in enumerate(dates):
        # 转换日期格式
        if isinstance(date_val, str):
            date_str = date_val.replace("-", "")
        else:
            date_str = date_val.strftime("%Y%m%d")

        # 筛选当日数据
        day_df = merged_df.filter(pl.col(date_col) == date_val)

        # 保存
        output_file = output_dir / f"{date_str}.parquet"
        day_df.write_parquet(output_file)
        count += 1

        if (i + 1) % 500 == 0:
            logger.info(f"已保存 {i + 1}/{len(dates)} 个日期文件")

    logger.info(f"转换完成: 共生成 {count} 个日期文件")
    return count


# =====================================================
# 分红数据采集函数
# =====================================================

def load_dividend_history(symbol: str) -> bool:
    """
    采集股票分红历史

    Args:
        symbol: 股票代码，如 600000

    Returns:
        是否成功
    """
    _check_akshare()

    code = _convert_symbol(symbol)
    AKSHARE_DIVIDEND_DIR.mkdir(parents=True, exist_ok=True)

    parquet_file = AKSHARE_DIVIDEND_DIR / f"{code}.parquet"

    try:
        # 使用东财分红数据接口
        df = ak.stock_fhps_detail_em(symbol=code)

        if df is not None and not df.empty:
            pl_df = pl.from_pandas(df)
            pl_df.write_parquet(parquet_file)
            logger.info(f"{code} 分红历史已保存，共 {len(pl_df)} 条记录")
            return True
        else:
            logger.warning(f"{code} 分红历史数据为空")
            return True

    except Exception as e:
        logger.error(f"采集 {code} 分红历史失败: {e}")
        return False


# =====================================================
# 测试函数
# =====================================================

def test_connection() -> bool:
    """
    测试 AKShare 连接

    Returns:
        是否成功
    """
    try:
        _check_akshare()

        # 测试获取申万一级行业
        df = ak.sw_index_first_info()
        if df is not None and not df.empty:
            logger.info(f"AKShare 连接成功，获取到 {len(df)} 个申万一级行业")
            print(f"\n申万一级行业（前5个）:\n{df.head()}")
            return True
        else:
            logger.warning("AKShare 连接成功，但数据为空")
            return True

    except ImportError as e:
        logger.error(f"AKShare 未安装: {e}")
        return False
    except Exception as e:
        logger.error(f"AKShare 连接失败: {e}")
        return False


# =====================================================
# 指数日线数据采集函数
# =====================================================

# 指数配置：AKShare代码 -> (名称, 掘金代码)
INDEX_CONFIG = {
    '000001': ('上证指数', 'SHSE.000001'),
    '000016': ('上证50', 'SHSE.000016'),
    '000300': ('沪深300', 'SHSE.000300'),
    '000905': ('中证500', 'SHSE.000905'),
    '000852': ('中证1000', 'SHSE.000852'),
    '399303': ('中证2000', 'SZSE.399303'),
    '399006': ('创业板指', 'SZSE.399006'),
    '000688': ('科创50', 'SHSE.000688'),
    '000010': ('上证180', 'SHSE.000010'),
    '399001': ('深证成指', 'SZSE.399001'),
    '399005': ('中小板指', 'SZSE.399005'),
    '399106': ('深证综指', 'SZSE.399106'),
    '000015': ('红利指数', 'SHSE.000015'),
}


def load_index_daily(
    index_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> bool:
    """
    采集单个指数的历史日线数据

    Args:
        index_code: 指数代码，如 000300
        start_date: 开始日期，格式 YYYY-MM-DD 或 YYYYMMDD
        end_date: 结束日期，格式 YYYY-MM-DD 或 YYYYMMDD

    Returns:
        是否成功
    """
    _check_akshare()

    index_name = INDEX_CONFIG.get(index_code, (index_code, ''))[0]
    logger.info(f"采集 {index_name} ({index_code}) 日线数据...")

    try:
        # 处理日期格式
        start = start_date.replace("-", "") if start_date else "19900101"
        end = end_date.replace("-", "") if end_date else datetime.now().strftime("%Y%m%d")

        df = ak.index_zh_a_hist(
            symbol=index_code,
            period="daily",
            start_date=start,
            end_date=end
        )

        if df is None or df.empty:
            logger.warning(f"{index_name} 无数据")
            return True

        # 转换为 Polars
        df_pl = pl.from_pandas(df)

        # 重命名列（中文 -> 英文）
        column_mapping = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover_rate',
        }
        for old_name, new_name in column_mapping.items():
            if old_name in df_pl.columns:
                df_pl = df_pl.rename({old_name: new_name})

        # 添加指数代码列
        df_pl = df_pl.with_columns(pl.lit(index_code).alias('symbol'))

        # 保存
        AKSHARE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        file_path = AKSHARE_INDEX_DIR / f"{index_code}.parquet"
        df_pl.write_parquet(file_path)

        logger.info(f"{index_name} 日线数据已保存: {file_path} ({len(df_pl)} 条)")
        return True

    except Exception as e:
        logger.error(f"{index_name} 采集失败: {e}")
        return False


def load_all_index_daily(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    delay: float = 0.5
) -> Tuple[int, int]:
    """
    采集所有指数的历史日线数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        delay: 请求间隔（秒）

    Returns:
        (成功数, 失败数)
    """
    _check_akshare()

    logger.info("=" * 60)
    logger.info("开始采集所有指数日线数据")
    logger.info(f"指数数量: {len(INDEX_CONFIG)}")
    logger.info("=" * 60)

    success_count = 0
    fail_count = 0

    for index_code, (index_name, _) in INDEX_CONFIG.items():
        try:
            if load_index_daily(index_code, start_date, end_date):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logger.error(f"{index_name} 采集异常: {e}")
            fail_count += 1

        time.sleep(delay)

    logger.info("=" * 60)
    logger.info(f"采集完成: 成功 {success_count}, 失败 {fail_count}")
    logger.info("=" * 60)

    return success_count, fail_count


def show_index_stats():
    """显示已采集的指数数据统计"""
    if not AKSHARE_INDEX_DIR.exists():
        print("指数数据目录不存在")
        return

    print("\n指数日线数据统计")
    print("=" * 70)
    print(f"{'指数名称':<12} {'代码':<8} {'最早日期':<12} {'最新日期':<12} {'记录数':>8}")
    print("-" * 70)

    for index_code, (index_name, _) in INDEX_CONFIG.items():
        file_path = AKSHARE_INDEX_DIR / f"{index_code}.parquet"

        if file_path.exists():
            df = pl.read_parquet(file_path)
            if 'date' in df.columns:
                dates = df['date'].sort()
                min_date = str(dates[0])[:10]
                max_date = str(dates[-1])[:10]
            else:
                min_date = max_date = "N/A"

            print(f"{index_name:<12} {index_code:<8} {min_date:<12} {max_date:<12} {len(df):>8,}")
        else:
            print(f"{index_name:<12} {index_code:<8} {'未采集':<12}")

    print("=" * 70)


# =====================================================
# 命令行入口
# =====================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="AKShare 数据采集器")
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试 AKShare 连接",
    )
    parser.add_argument(
        "--industry",
        action="store_true",
        help="采集申万行业分类",
    )
    parser.add_argument(
        "--constituents",
        action="store_true",
        help="采集所有行业成分股",
    )
    parser.add_argument(
        "--finance",
        action="store_true",
        help="采集财务数据",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="采集单只股票历史日K线（需 --symbol）",
    )
    parser.add_argument(
        "--hist-all",
        action="store_true",
        help="采集所有股票日K线（按日期逐天采集，较慢）",
    )
    parser.add_argument(
        "--hist-batch",
        action="store_true",
        help="⭐ 批量采集所有股票完整历史（按股票存储，推荐，快100倍）",
    )
    parser.add_argument(
        "--convert-daily",
        action="store_true",
        help="将按股票存储的数据转换为按日期存储",
    )
    parser.add_argument(
        "--dividend",
        action="store_true",
        help="采集分红历史",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="采集所有指数历史日线数据",
    )
    parser.add_argument(
        "--index-stats",
        action="store_true",
        help="显示指数数据统计",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="股票代码，如 600000（用于 --finance, --hist, --dividend）",
    )
    parser.add_argument(
        "--date",
        type=str,
        help="单个日期，格式 YYYY-MM-DD（用于 --hist-all）",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="开始日期，格式 YYYY-MM-DD（用于 --hist, --hist-all）",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="结束日期，格式 YYYY-MM-DD（用于 --hist, --hist-all）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="请求间隔（秒），默认 0.1",
    )

    args = parser.parse_args()

    # 测试连接
    if args.test:
        test_connection()
        return

    # 采集申万行业分类
    if args.industry:
        load_sw_industry()
        return

    # 采集行业成分股
    if args.constituents:
        load_all_sw_constituents()
        return

    # 采集财务数据
    if args.finance:
        if not args.symbol:
            parser.error("--finance 需要指定 --symbol")
        load_financial_report(args.symbol)
        load_financial_abstract(args.symbol)
        return

    # 采集单只股票历史K线
    if args.hist:
        if not args.symbol:
            parser.error("--hist 需要指定 --symbol")
        if not args.start or not args.end:
            parser.error("--hist 需要指定 --start 和 --end")
        load_hist_daily(args.symbol, args.start, args.end)
        return

    # 采集所有股票日K线（按日期存储，逐天采集）
    if args.hist_all:
        if args.date:
            # 单个日期
            load_hist_daily_by_date(args.date, delay=args.delay)
        elif args.start and args.end:
            # 日期范围
            load_hist_daily_range(args.start, args.end, delay=args.delay)
        else:
            parser.error("--hist-all 需要指定 --date 或 --start/--end")
        return

    # ⭐ 批量采集所有股票完整历史（按股票存储，推荐）
    if args.hist_batch:
        start = args.start.replace("-", "") if args.start else "19910101"
        end = args.end.replace("-", "") if args.end else None
        load_all_hist_daily(
            start_date=start,
            end_date=end,
            delay=args.delay if args.delay != 0.1 else 0.5,  # 默认0.5秒更安全
        )
        return

    # 转换为按日期存储
    if args.convert_daily:
        convert_hist_to_daily()
        return

    # 采集分红历史
    if args.dividend:
        if not args.symbol:
            parser.error("--dividend 需要指定 --symbol")
        load_dividend_history(args.symbol)
        return

    # 采集指数日线数据
    if args.index:
        load_all_index_daily(
            start_date=args.start,
            end_date=args.end,
            delay=args.delay
        )
        return

    # 显示指数数据统计
    if args.index_stats:
        show_index_stats()
        return

    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
