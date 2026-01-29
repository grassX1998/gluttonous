"""
掘金 API 封装模块

提供掘金量化数据 API 的封装，支持：
- 交易日历管理
- 分钟K线数据下载
- 股票元数据下载
- 指数成分股下载
"""

from __future__ import print_function, absolute_import

import os
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import toml

try:
    import gm.api
except ImportError:
    gm = None

from pipeline.shared.config import (
    RAW_DATA_ROOT, JUEJIN_CONFIG, EXTENDED_DATA_CONFIG,
    FINANCE_BALANCE_DIR, FINANCE_INCOME_DIR, FINANCE_CASHFLOW_DIR, FINANCE_DERIV_DIR,
    INDUSTRY_DATA_DIR, DIVIDEND_DATA_DIR, ADJ_FACTOR_DIR, SHARE_DATA_DIR,
    TICK_DATA_DIR,
)

# 模块级 logger，可通过 set_logger() 替换为统一配置的 logger
_logger = logging.getLogger(__name__)


def set_logger(logger: logging.Logger):
    """
    设置模块使用的 logger

    在 collector.py 中调用，确保日志写入统一的日志文件。

    Args:
        logger: 配置好的 Logger 实例
    """
    global _logger
    _logger = logger


def get_logger() -> logging.Logger:
    """获取当前使用的 logger"""
    return _logger


# 注意：下面所有的 logger 调用都使用 _logger 变量

# 数据目录（使用配置文件中的路径）
DATA_DIR = RAW_DATA_ROOT


def _init_api():
    """初始化掘金API"""
    if gm is None:
        raise ImportError("掘金SDK未安装，请先安装: pip install gm")
    gm.api.set_serv_addr(JUEJIN_CONFIG["server"])
    gm.api.set_token(JUEJIN_CONFIG["token"])


def test_connection() -> bool:
    """
    测试API连接

    Returns:
        bool: 连接是否成功
    """
    try:
        _init_api()
        dates = gm.api.get_trading_dates("SHSE", "2026-01-01", "2026-01-15")
        if dates:
            _logger.info(f"API连接成功，获取到 {len(dates)} 个交易日")
            return True
        else:
            _logger.warning("API连接成功，但未获取到交易日数据")
            return True
    except Exception as e:
        _logger.error(f"API连接失败: {e}")
        return False


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    """
    获取指定日期范围内的交易日

    Args:
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD

    Returns:
        交易日列表
    """
    _init_api()
    return gm.api.get_trading_dates("SHSE", start_date, end_date)


def load_trading_days() -> List[str]:
    """
    更新并保存交易日历

    从掘金API获取最新交易日，与本地文件合并后保存。

    Returns:
        更新后的交易日列表
    """
    _init_api()

    trading_days_file = DATA_DIR / "cfg" / "trading_days.toml"

    # 读取现有交易日
    existing_days = []
    if trading_days_file.exists():
        data = toml.load(str(trading_days_file))
        existing_days = data.get("trading_days", [])

    # 获取新交易日（从2024-12-13到当前年底）
    current_year = datetime.now().year
    end_date = f"{current_year}-12-31"
    new_dates = get_trading_dates("2024-12-13", end_date)

    # 合并并排序
    trading_days = sorted(set(existing_days + new_dates))

    # 保存
    trading_days_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trading_days_file, "w") as f:
        toml.dump({"trading_days": trading_days}, f)

    _logger.info(f"交易日历已更新，共 {len(trading_days)} 个交易日")
    return trading_days


def _get_mkline(code: str, date: str) -> pl.DataFrame:
    """
    从API获取分钟K线数据

    Args:
        code: 股票代码，如 SHSE.600000
        date: 日期，格式 YYYY-MM-DD

    Returns:
        分钟K线数据
    """
    _init_api()

    parquet_file = DATA_DIR / "mkline" / code / f"{date}.parquet"
    if parquet_file.exists():
        return pl.read_parquet(parquet_file)

    try:
        df = pl.DataFrame(
            gm.api.history(
                symbol=code,
                frequency="60s",
                start_time=f"{date} 09:00:00",
                end_time=f"{date} 16:00:00",
                fields="symbol, eob, open, close, high, low, volume, amount",
                adjust=0,
                fill_missing="Last",
                df=True,
            )
        )
        if df.is_empty():
            return df

        # 处理时间字段
        df = df.with_columns([
            pl.col("eob").dt.date().alias("date"),
            pl.col("eob").dt.time().alias("time"),
        ]).drop("eob")

        # 重命名 amount -> turnover
        df = df.with_columns([
            pl.col("amount").alias("turnover"),
        ]).drop("amount")

    except Exception as ex:
        _logger.error(f"获取 {code} {date} K线数据失败: {ex}")
        return pl.DataFrame()

    return df


def load_mkline(code: str, date: str, max_retries: int = 3) -> bool:
    """
    下载并保存分钟K线数据

    Args:
        code: 股票代码
        date: 日期
        max_retries: 最大重试次数

    Returns:
        是否成功
    """
    parquet_file = DATA_DIR / "mkline" / code / f"{date}.parquet"

    # 如果文件已存在，跳过
    if parquet_file.exists():
        return True

    for attempt in range(max_retries):
        try:
            df = _get_mkline(code, date)
            if not df.is_empty():
                parquet_file.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(parquet_file)
                return True
            return True  # 空数据也算成功（非交易日等情况）
        except Exception as e:
            if attempt < max_retries - 1:
                _logger.warning(f"获取 {code} {date} 失败，重试 {attempt + 2}/{max_retries}: {e}")
            else:
                _logger.error(f"获取 {code} {date} 失败，已达最大重试次数: {e}")
                return False

    return False


def load_instruments() -> pl.DataFrame:
    """
    下载并保存股票元数据

    Returns:
        股票元数据 DataFrame
    """
    _init_api()

    df = pl.DataFrame(
        gm.api.get_instruments(
            exchanges=["SZSE", "SHSE"],
            skip_suspended=False,
            skip_st=False,
            df=True,
        )
    )

    instruments_file = DATA_DIR / "meta" / "instruments.parquet"
    instruments_file.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(instruments_file)

    _logger.info(f"证券元数据已保存，共 {len(df)} 条（含股票、基金、可转债等）")
    return df


def load_indexs(date: Optional[str] = None) -> None:
    """
    下载并保存指数成分股数据

    Args:
        date: 日期字符串，格式为 YYYY-MM-DD，用于保存每日快照
              如果为空则只保存最新版本
    """
    _init_api()

    # 指数列表及名称
    index_list = [
        ("SHSE.000016", "上证50"),
        ("SHSE.000300", "沪深300"),
        ("SHSE.000905", "中证500"),
        ("SHSE.000852", "中证1000"),
        ("SZSE.399303", "中证2000"),
    ]

    index_dir = DATA_DIR / "meta" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # 如果指定日期，创建日期快照目录
    if date:
        date_dir = index_dir / date
        date_dir.mkdir(parents=True, exist_ok=True)

    _logger.info(f"开始采集 {len(index_list)} 个指数的成分股数据...")

    success_count = 0
    for i, (index_code, index_name) in enumerate(index_list, 1):
        try:
            _logger.info(f"  [{i}/{len(index_list)}] 采集 {index_name} ({index_code})...")
            symbols = pl.DataFrame(gm.api.stk_get_index_constituents(index_code)).to_dicts()
            toml_value = {"sec_ids": [s["symbol"] for s in symbols]}

            # 保存最新版本
            with open(index_dir / f"{index_code}.toml", "w") as f:
                toml.dump(toml_value, f)

            # 保存日期快照
            if date:
                with open(index_dir / date / f"{index_code}.toml", "w") as f:
                    toml.dump(toml_value, f)

            _logger.info(f"  [{i}/{len(index_list)}] {index_name} 完成，共 {len(symbols)} 只成分股")
            success_count += 1
        except Exception as e:
            _logger.error(f"  [{i}/{len(index_list)}] {index_name} 采集失败: {e}")

    _logger.info(f"指数成分股数据采集完成: {success_count}/{len(index_list)} 成功")


def get_stock_list() -> List[str]:
    """
    获取所有A股股票列表

    Returns:
        股票代码列表
    """
    instruments_file = DATA_DIR / "meta" / "instruments.parquet"

    if not instruments_file.exists():
        load_instruments()

    df = pl.read_parquet(instruments_file)
    # sec_type=1 表示股票, sec_level=1 表示主板
    stock_list = (
        df.filter((pl.col("sec_type") == 1) & (pl.col("sec_level") == 1))
    )["symbol"].to_list()

    return stock_list


# =====================================================
# 财务数据采集函数
# =====================================================

# 资产负债表关键字段（最多20个字段）
# 有效字段: fix_ast, lt_eqy_inv, mny_cptl, invt 等
BALANCE_FIELDS = "fix_ast,lt_eqy_inv,mny_cptl,invt"

# 利润表关键字段
# 有效字段: NET_PROF, oper_prof, inc_tax 等
INCOME_FIELDS = "NET_PROF,oper_prof,inc_tax"

# 现金流量表关键字段
# TODO: 需要确认正确的字段名，暂时禁用
CASHFLOW_FIELDS = None  # 暂未找到有效字段名

# 财务衍生指标关键字段
# 有效字段: ROE, ROA, eps_basic, BPS 等
DERIV_FIELDS = "ROE,ROA,eps_basic,BPS"


def load_finance_balance(date: str, stock_list: Optional[List[str]] = None) -> bool:
    """
    采集资产负债表数据

    注意：掘金API需要逐个股票查询，且fields参数必须指定具体字段。

    Args:
        date: 报告期日期，格式 YYYY-MM-DD（如 2024-09-30 表示三季报）
        stock_list: 股票列表，为None时自动获取

    Returns:
        是否成功
    """
    _init_api()

    parquet_file = FINANCE_BALANCE_DIR / f"{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"资产负债表 {date} 已存在，跳过")
        return True

    if stock_list is None:
        stock_list = get_stock_list()

    all_data = []
    success_count = 0
    fail_count = 0

    for symbol in stock_list:
        try:
            result = gm.api.stk_get_fundamentals_balance(
                symbol=symbol,
                fields=BALANCE_FIELDS,
                start_date=date,
                end_date=date,
                df=False,
            )
            if result:
                all_data.extend(result)
                success_count += 1
        except Exception:
            fail_count += 1

    if not all_data:
        _logger.warning(f"资产负债表 {date} 数据为空 (成功: {success_count}, 失败: {fail_count})")
        return True

    try:
        df = pl.DataFrame(all_data)
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"资产负债表 {date} 已保存，共 {len(df)} 条记录 (成功: {success_count}, 失败: {fail_count})")
        return True
    except Exception as e:
        _logger.error(f"保存资产负债表 {date} 失败: {e}")
        return False


def load_finance_income(date: str, stock_list: Optional[List[str]] = None) -> bool:
    """
    采集利润表数据

    Args:
        date: 报告期日期，格式 YYYY-MM-DD
        stock_list: 股票列表，为None时自动获取

    Returns:
        是否成功
    """
    _init_api()

    parquet_file = FINANCE_INCOME_DIR / f"{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"利润表 {date} 已存在，跳过")
        return True

    if stock_list is None:
        stock_list = get_stock_list()

    all_data = []
    success_count = 0
    fail_count = 0

    for symbol in stock_list:
        try:
            result = gm.api.stk_get_fundamentals_income(
                symbol=symbol,
                fields=INCOME_FIELDS,
                start_date=date,
                end_date=date,
                df=False,
            )
            if result:
                all_data.extend(result)
                success_count += 1
        except Exception:
            fail_count += 1

    if not all_data:
        _logger.warning(f"利润表 {date} 数据为空 (成功: {success_count}, 失败: {fail_count})")
        return True

    try:
        df = pl.DataFrame(all_data)
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"利润表 {date} 已保存，共 {len(df)} 条记录 (成功: {success_count}, 失败: {fail_count})")
        return True
    except Exception as e:
        _logger.error(f"保存利润表 {date} 失败: {e}")
        return False


def load_finance_cashflow(date: str, stock_list: Optional[List[str]] = None) -> bool:
    """
    采集现金流量表数据

    注意：现金流量表字段名暂未确认，此功能暂时禁用。

    Args:
        date: 报告期日期，格式 YYYY-MM-DD
        stock_list: 股票列表，为None时自动获取

    Returns:
        是否成功
    """
    # TODO: 确认正确的现金流量表字段名后启用
    if CASHFLOW_FIELDS is None:
        _logger.warning(f"现金流量表字段未配置，跳过 {date}")
        return True

    _init_api()

    parquet_file = FINANCE_CASHFLOW_DIR / f"{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"现金流量表 {date} 已存在，跳过")
        return True

    if stock_list is None:
        stock_list = get_stock_list()

    all_data = []
    success_count = 0
    fail_count = 0

    for symbol in stock_list:
        try:
            result = gm.api.stk_get_fundamentals_cashflow(
                symbol=symbol,
                fields=CASHFLOW_FIELDS,
                start_date=date,
                end_date=date,
                df=False,
            )
            if result:
                all_data.extend(result)
                success_count += 1
        except Exception:
            fail_count += 1

    if not all_data:
        _logger.warning(f"现金流量表 {date} 数据为空 (成功: {success_count}, 失败: {fail_count})")
        return True

    try:
        df = pl.DataFrame(all_data)
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"现金流量表 {date} 已保存，共 {len(df)} 条记录 (成功: {success_count}, 失败: {fail_count})")
        return True
    except Exception as e:
        _logger.error(f"保存现金流量表 {date} 失败: {e}")
        return False


def load_finance_deriv(date: str, stock_list: Optional[List[str]] = None) -> bool:
    """
    采集财务衍生指标数据

    包括：ROE, ROA, EPS, BPS, 毛利率等

    Args:
        date: 报告期日期，格式 YYYY-MM-DD
        stock_list: 股票列表，为None时自动获取

    Returns:
        是否成功
    """
    _init_api()

    parquet_file = FINANCE_DERIV_DIR / f"{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"财务衍生指标 {date} 已存在，跳过")
        return True

    if stock_list is None:
        stock_list = get_stock_list()

    all_data = []
    success_count = 0
    fail_count = 0

    for symbol in stock_list:
        try:
            # 使用 stk_get_finance_deriv (非 _pt 版本)
            result = gm.api.stk_get_finance_deriv(
                symbol=symbol,
                fields=DERIV_FIELDS,
                start_date=date,
                end_date=date,
                df=False,
            )
            if result:
                all_data.extend(result)
                success_count += 1
        except Exception:
            fail_count += 1

    if not all_data:
        _logger.warning(f"财务衍生指标 {date} 数据为空 (成功: {success_count}, 失败: {fail_count})")
        return True

    try:
        df = pl.DataFrame(all_data)
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"财务衍生指标 {date} 已保存，共 {len(df)} 条记录 (成功: {success_count}, 失败: {fail_count})")
        return True
    except Exception as e:
        _logger.error(f"保存财务衍生指标 {date} 失败: {e}")
        return False


def load_finance_data(date: str) -> bool:
    """
    采集所有财务数据（资产负债表、利润表、现金流量表、衍生指标）

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否全部成功
    """
    results = [
        load_finance_balance(date),
        load_finance_income(date),
        load_finance_cashflow(date),
        load_finance_deriv(date),
    ]
    return all(results)


def should_update_finance(date: str) -> bool:
    """
    判断是否需要更新财务数据

    财务数据在季报发布月份（1/4/8/10月）更新

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否需要更新
    """
    dt = datetime.strptime(date, "%Y-%m-%d")
    update_months = EXTENDED_DATA_CONFIG["finance_update_months"]
    update_day = EXTENDED_DATA_CONFIG["finance_update_day"]

    return dt.month in update_months and dt.day >= update_day


# =====================================================
# 行业分类采集函数
# =====================================================

def load_industry_category() -> bool:
    """
    采集行业分类体系（申万/证监会）

    Returns:
        是否成功
    """
    _init_api()

    INDUSTRY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # 采集申万行业分类（一级、二级、三级）
        for level in [1, 2, 3]:
            df = pl.DataFrame(
                gm.api.stk_get_industry_category(
                    source="sw",  # 申万分类
                    level=level,
                    df=True,
                )
            )
            if not df.is_empty():
                parquet_file = INDUSTRY_DATA_DIR / f"sw_level{level}.parquet"
                df.write_parquet(parquet_file)
                _logger.info(f"申万{level}级行业分类已保存，共 {len(df)} 个行业")

        # 采集证监会行业分类（一级、二级）
        for level in [1, 2]:
            df = pl.DataFrame(
                gm.api.stk_get_industry_category(
                    source="zjh",  # 证监会分类
                    level=level,
                    df=True,
                )
            )
            if not df.is_empty():
                parquet_file = INDUSTRY_DATA_DIR / f"zjh_level{level}.parquet"
                df.write_parquet(parquet_file)
                _logger.info(f"证监会{level}级行业分类已保存，共 {len(df)} 个行业")

        return True

    except Exception as e:
        _logger.error(f"采集行业分类体系失败: {e}")
        return False


def load_symbol_industry(date: str) -> bool:
    """
    采集股票所属行业映射

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否成功
    """
    _init_api()

    parquet_file = INDUSTRY_DATA_DIR / f"symbol_industry_{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"股票行业映射 {date} 已存在，跳过")
        return True

    try:
        # 获取所有股票列表
        stock_list = get_stock_list()

        # 采集申万三级行业分类
        df = pl.DataFrame(
            gm.api.stk_get_symbol_industry(
                symbols=stock_list,
                source="sw",    # 申万分类
                level=3,        # 三级行业
                date=date,
                df=True,
            )
        )

        if df.is_empty():
            _logger.warning(f"股票行业映射 {date} 数据为空")
            return True

        INDUSTRY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"股票行业映射 {date} 已保存，共 {len(df)} 条记录")
        return True

    except Exception as e:
        _logger.error(f"采集股票行业映射 {date} 失败: {e}")
        return False


def load_industry_data(date: str) -> bool:
    """
    采集行业分类数据（分类体系 + 股票映射）

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否成功
    """
    # 检查是否需要更新行业分类体系
    category_file = INDUSTRY_DATA_DIR / "sw_level1.parquet"
    should_update_category = not category_file.exists()

    if should_update_category:
        load_industry_category()

    return load_symbol_industry(date)


def should_update_industry(date: str) -> bool:
    """
    判断是否需要更新行业分类

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否需要更新
    """
    category_file = INDUSTRY_DATA_DIR / "sw_level1.parquet"
    if not category_file.exists():
        return True

    # 检查文件修改时间
    import os
    mtime = os.path.getmtime(category_file)
    file_date = datetime.fromtimestamp(mtime)
    current_date = datetime.strptime(date, "%Y-%m-%d")

    interval_days = EXTENDED_DATA_CONFIG["industry_update_interval_days"]
    return (current_date - file_date).days >= interval_days


# =====================================================
# 分红复权采集函数
# =====================================================

def load_dividend(symbols: List[str], start_date: str, end_date: str) -> bool:
    """
    采集分红送股数据

    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        是否成功
    """
    _init_api()

    DIVIDEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for symbol in symbols:
        parquet_file = DIVIDEND_DATA_DIR / f"{symbol}.parquet"

        try:
            # 获取分红数据
            df = pl.DataFrame(
                gm.api.stk_get_dividend(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    df=True,
                )
            )

            if not df.is_empty():
                # 如果文件存在，合并数据
                if parquet_file.exists():
                    existing_df = pl.read_parquet(parquet_file)
                    df = pl.concat([existing_df, df]).unique()

                df.write_parquet(parquet_file)
                success_count += 1
            else:
                success_count += 1  # 无分红数据也算成功

        except Exception as e:
            _logger.debug(f"获取 {symbol} 分红数据失败: {e}")
            fail_count += 1

    _logger.info(f"分红数据采集完成: 成功 {success_count}, 失败 {fail_count}")
    return fail_count == 0


def load_adj_factor(date: str) -> bool:
    """
    采集复权因子数据

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否成功
    """
    _init_api()

    parquet_file = ADJ_FACTOR_DIR / f"{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"复权因子 {date} 已存在，跳过")
        return True

    try:
        # 获取所有股票列表
        stock_list = get_stock_list()

        # 获取复权因子
        df = pl.DataFrame(
            gm.api.stk_get_adj_factor(
                symbols=stock_list,
                start_date=date,
                end_date=date,
                df=True,
            )
        )

        if df.is_empty():
            _logger.warning(f"复权因子 {date} 数据为空")
            return True

        ADJ_FACTOR_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"复权因子 {date} 已保存，共 {len(df)} 条记录")
        return True

    except Exception as e:
        _logger.error(f"采集复权因子 {date} 失败: {e}")
        return False


def load_dividend_data(date: str) -> bool:
    """
    采集分红复权数据（分红 + 复权因子）

    Args:
        date: 日期

    Returns:
        是否成功
    """
    # 获取股票列表
    stock_list = get_stock_list()

    # 计算开始日期（回溯一年）
    dt = datetime.strptime(date, "%Y-%m-%d")
    lookback_days = EXTENDED_DATA_CONFIG["dividend_lookback_days"]
    start_date = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    results = [
        load_dividend(stock_list, start_date, date),
        load_adj_factor(date),
    ]
    return all(results)


# =====================================================
# 股本数据采集函数
# =====================================================

def load_share_change(date: str) -> bool:
    """
    采集股本变动数据

    包含：总股本、流通股本、限售股本等

    Args:
        date: 日期，格式 YYYY-MM-DD

    Returns:
        是否成功
    """
    _init_api()

    parquet_file = SHARE_DATA_DIR / f"{date}.parquet"
    if parquet_file.exists():
        _logger.debug(f"股本数据 {date} 已存在，跳过")
        return True

    try:
        # 获取所有股票列表
        stock_list = get_stock_list()

        # 获取股本变动数据
        df = pl.DataFrame(
            gm.api.stk_get_share_change(
                symbols=stock_list,
                start_date=date,
                end_date=date,
                df=True,
            )
        )

        if df.is_empty():
            _logger.warning(f"股本数据 {date} 数据为空")
            return True

        SHARE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)
        _logger.info(f"股本数据 {date} 已保存，共 {len(df)} 条记录")
        return True

    except Exception as e:
        _logger.error(f"采集股本数据 {date} 失败: {e}")
        return False


def load_share_data(date: str) -> bool:
    """
    采集股本数据

    Args:
        date: 日期

    Returns:
        是否成功
    """
    return load_share_change(date)


# =====================================================
# Level 1 Tick 数据采集函数
# =====================================================

def _get_tick_l1(code: str, date: str) -> pl.DataFrame:
    """
    从API获取 Level 1 Tick 数据

    Args:
        code: 股票代码，如 SHSE.600000
        date: 日期，格式 YYYY-MM-DD

    Returns:
        Tick 数据 DataFrame
    """
    _init_api()

    try:
        # 使用 history 接口获取 tick 数据
        df = pl.DataFrame(
            gm.api.history(
                symbol=code,
                frequency="tick",
                start_time=f"{date} 09:15:00",
                end_time=f"{date} 15:01:00",
                fields="symbol, created_at, price, open, high, low, cum_volume, cum_amount, last_volume, last_amount, cum_position, last_close, quotes",
                adjust=0,
                df=True,
            )
        )
        if df.is_empty():
            return df

        # 处理时间字段
        if "created_at" in df.columns:
            df = df.with_columns([
                pl.col("created_at").dt.date().alias("date"),
                pl.col("created_at").dt.time().alias("time"),
            ]).drop("created_at")

    except Exception as ex:
        _logger.debug(f"获取 {code} {date} Tick 数据失败: {ex}")
        return pl.DataFrame()

    return df


def load_tick_l1(code: str, date: str, max_retries: int = 3) -> bool:
    """
    下载并保存 Level 1 Tick 数据

    Args:
        code: 股票代码，如 SHSE.600000
        date: 日期，格式 YYYY-MM-DD
        max_retries: 最大重试次数

    Returns:
        是否成功
    """
    import time

    parquet_file = TICK_DATA_DIR / code / f"{date}.parquet"

    # 如果文件已存在，跳过
    if parquet_file.exists():
        return True

    retry_delay = EXTENDED_DATA_CONFIG.get("tick_retry_delay", 1.0)

    for attempt in range(max_retries):
        try:
            df = _get_tick_l1(code, date)
            if not df.is_empty():
                parquet_file.parent.mkdir(parents=True, exist_ok=True)
                df.write_parquet(parquet_file)
                return True
            return True  # 空数据也算成功（非交易日、停牌等情况）
        except Exception as e:
            if attempt < max_retries - 1:
                _logger.warning(f"获取 {code} {date} Tick 失败，重试 {attempt + 2}/{max_retries}: {e}")
                time.sleep(retry_delay)
            else:
                _logger.error(f"获取 {code} {date} Tick 失败，已达最大重试次数: {e}")
                return False

    return False


def load_tick_data(date: str, stock_list: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    批量采集 Level 1 Tick 数据

    Args:
        date: 日期，格式 YYYY-MM-DD
        stock_list: 股票列表，为 None 时自动获取

    Returns:
        (成功数, 失败数) 元组
    """
    import time
    from typing import Tuple

    if stock_list is None:
        stock_list = get_stock_list()

    batch_size = EXTENDED_DATA_CONFIG.get("tick_batch_size", 50)
    retry_delay = EXTENDED_DATA_CONFIG.get("tick_retry_delay", 1.0)

    success_count = 0
    fail_count = 0

    for i, code in enumerate(stock_list):
        success = load_tick_l1(code, date)
        if success:
            success_count += 1
        else:
            fail_count += 1

        # 每批次暂停一下，避免 API 限流
        if (i + 1) % batch_size == 0:
            time.sleep(retry_delay)

    _logger.info(f"Tick 数据 {date} 采集完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


# =====================================================
# 指数成分股历史数据采集函数
# =====================================================

# 指数配置：代码 -> (名称, 最早可查日期)
INDEX_CONSTITUENTS_CONFIG = {
    # 宽基指数
    'SHSE.000016': ('上证50', '2004-01-02'),
    'SHSE.000300': ('沪深300', '2005-04-08'),
    'SHSE.000905': ('中证500', '2007-01-15'),
    'SHSE.000852': ('中证1000', '2014-10-17'),
    'SZSE.399303': ('中证2000', '2014-10-17'),

    # 创业板/科创板
    'SZSE.399006': ('创业板指', '2010-06-01'),
    'SZSE.399102': ('创业板综', '2010-06-01'),
    'SHSE.000688': ('科创50', '2020-07-23'),

    # 上证系列
    'SHSE.000001': ('上证指数', '1992-05-21'),
    'SHSE.000010': ('上证180', '2002-07-01'),

    # 深证系列
    'SZSE.399001': ('深证成指', '2005-01-04'),
    'SZSE.399005': ('中小板指', '2006-01-24'),
    'SZSE.399106': ('深证综指', '1991-04-03'),

    # 其他
    'SHSE.000015': ('红利指数', '2005-01-04'),
}

# 每日采集的核心指数（数据量适中）
DAILY_INDEX_LIST = [
    'SHSE.000016',  # 上证50
    'SHSE.000300',  # 沪深300
    'SHSE.000905',  # 中证500
    'SHSE.000852',  # 中证1000
    'SZSE.399303',  # 中证2000
    'SZSE.399006',  # 创业板指
    'SHSE.000688',  # 科创50
]


def load_index_constituents_daily(date: str, index_list: Optional[List[str]] = None) -> bool:
    """
    采集指定日期的指数成分股数据

    Args:
        date: 日期，格式 YYYY-MM-DD
        index_list: 指数列表，为 None 时使用默认的每日采集列表

    Returns:
        是否成功
    """
    import pandas as pd
    from pipeline.shared.config import INDEX_CONSTITUENTS_HISTORY_DIR

    _init_api()

    if index_list is None:
        index_list = DAILY_INDEX_LIST

    INDEX_CONSTITUENTS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for index_code in index_list:
        index_name = INDEX_CONSTITUENTS_CONFIG.get(index_code, ('未知', ''))[0]

        try:
            result = gm.api.stk_get_index_constituents(index=index_code, trade_date=date)

            if isinstance(result, pd.DataFrame) and not result.empty:
                # 添加日期列
                result['trade_date'] = date
                df_new = pl.from_pandas(result)

                # 文件路径
                file_path = INDEX_CONSTITUENTS_HISTORY_DIR / f"{index_code.replace('.', '_')}.parquet"

                # 追加到现有文件
                if file_path.exists():
                    df_existing = pl.read_parquet(file_path)
                    # 检查该日期是否已存在
                    if date in df_existing['trade_date'].unique().to_list():
                        _logger.debug(f"{index_name} {date} 数据已存在，跳过")
                        success_count += 1
                        continue
                    df_combined = pl.concat([df_existing, df_new])
                else:
                    df_combined = df_new

                # 排序并保存
                df_combined = df_combined.sort(['trade_date', 'symbol'])
                df_combined.write_parquet(file_path)
                _logger.debug(f"{index_name} {date} 成分股数据已保存: {len(df_new)} 条")
                success_count += 1
            else:
                _logger.debug(f"{index_name} {date} 无数据")
                success_count += 1  # 无数据也算成功

        except Exception as e:
            _logger.error(f"{index_name} {date} 成分股采集失败: {e}")
            fail_count += 1

    _logger.info(f"指数成分股 {date} 采集完成: 成功 {success_count}/{len(index_list)}, 失败 {fail_count}")
    return fail_count == 0


def get_index_constituents_symbols(date: str, index_list: Optional[List[str]] = None) -> List[str]:
    """
    获取指定日期多个指数的成分股并集

    Args:
        date: 日期，格式 YYYY-MM-DD
        index_list: 指数列表，为 None 时使用默认列表

    Returns:
        成分股代码列表（去重）
    """
    import pandas as pd

    _init_api()

    if index_list is None:
        index_list = DAILY_INDEX_LIST

    all_symbols = set()

    for index_code in index_list:
        try:
            result = gm.api.stk_get_index_constituents(index=index_code, trade_date=date)
            if isinstance(result, pd.DataFrame) and not result.empty:
                symbols = result['symbol'].tolist()
                all_symbols.update(symbols)
        except Exception as e:
            _logger.warning(f"获取 {index_code} {date} 成分股失败: {e}")

    return sorted(list(all_symbols))


def load_index_mkline(date: str, index_list: Optional[List[str]] = None, max_retries: int = 3) -> Tuple[int, int]:
    """
    采集指定日期指数成分股的分钟K线数据

    Args:
        date: 日期，格式 YYYY-MM-DD
        index_list: 指数列表
        max_retries: 最大重试次数

    Returns:
        (成功数, 失败数) 元组
    """
    # 获取成分股列表
    symbols = get_index_constituents_symbols(date, index_list)

    if not symbols:
        _logger.warning(f"指数成分股 {date} 为空，跳过 K 线采集")
        return 0, 0

    _logger.info(f"开始采集 {len(symbols)} 只指数成分股的 K 线数据")

    success_count = 0
    fail_count = 0

    total = len(symbols)
    for i, code in enumerate(symbols):
        success = load_mkline(code, date, max_retries=max_retries)
        if success:
            success_count += 1
        else:
            fail_count += 1

        # 每200只或最后一只时输出进度
        if (i + 1) % 200 == 0 or (i + 1) == total:
            _logger.info(f"  K线进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) | 成功: {success_count} | 失败: {fail_count}")
    return success_count, fail_count


def load_index_tick(date: str, index_list: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    采集指定日期指数成分股的 Tick 数据

    Args:
        date: 日期，格式 YYYY-MM-DD
        index_list: 指数列表

    Returns:
        (成功数, 失败数) 元组
    """
    import time

    # 获取成分股列表
    symbols = get_index_constituents_symbols(date, index_list)

    if not symbols:
        _logger.warning(f"指数成分股 {date} 为空，跳过 Tick 采集")
        return 0, 0

    _logger.info(f"开始采集 {len(symbols)} 只指数成分股的 Tick 数据")

    batch_size = EXTENDED_DATA_CONFIG.get("tick_batch_size", 50)
    retry_delay = EXTENDED_DATA_CONFIG.get("tick_retry_delay", 1.0)

    success_count = 0
    fail_count = 0

    total = len(symbols)
    for i, code in enumerate(symbols):
        success = load_tick_l1(code, date)
        if success:
            success_count += 1
        else:
            fail_count += 1

        # 每批次暂停一下，避免 API 限流
        if (i + 1) % batch_size == 0:
            time.sleep(retry_delay)

        # 每200只或最后一只时输出进度
        if (i + 1) % 200 == 0 or (i + 1) == total:
            _logger.info(f"  Tick进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%) | 成功: {success_count} | 失败: {fail_count}")
    return success_count, fail_count


# =====================================================
# 多线程并行采集函数
# =====================================================

# 全局限流器实例（模块级别，线程安全）
_rate_limiter = None
_limiter_lock = threading.Lock()


def _get_rate_limiter():
    """获取全局限流器实例（单例模式）"""
    global _rate_limiter
    if _rate_limiter is None:
        with _limiter_lock:
            if _rate_limiter is None:
                from .rate_limiter import AdaptiveRateLimiter
                rate = EXTENDED_DATA_CONFIG.get("api_rate_limit", 50)
                burst = EXTENDED_DATA_CONFIG.get("api_burst_limit", 100)
                _rate_limiter = AdaptiveRateLimiter(
                    initial_rate=rate,
                    capacity=burst,
                    min_rate=5,
                    max_rate=rate * 2,
                )
    return _rate_limiter


def reset_rate_limiter():
    """重置全局限流器"""
    global _rate_limiter
    with _limiter_lock:
        if _rate_limiter is not None:
            _rate_limiter.reset()


# 指数本身的代码列表（5个宽基指数）
INDEX_CODES = [
    "SHSE.000016",  # 上证50
    "SHSE.000300",  # 沪深300
    "SHSE.000905",  # 中证500
    "SHSE.000852",  # 中证1000
    "SZSE.399303",  # 中证2000
]


def load_index_self_kline(date: str, max_retries: int = 3) -> Tuple[int, int]:
    """
    采集指数本身的分钟 K 线数据

    采集 5 个宽基指数的 K 线数据（非成分股，是指数本身）。

    Args:
        date: 日期，格式 YYYY-MM-DD
        max_retries: 最大重试次数

    Returns:
        (成功数, 失败数) 元组
    """
    _logger.info(f"开始采集 {len(INDEX_CODES)} 个指数本身的 K 线数据")

    success_count = 0
    fail_count = 0

    for index_code in INDEX_CODES:
        index_name = INDEX_CONSTITUENTS_CONFIG.get(index_code, ('未知', ''))[0]
        success = load_mkline(index_code, date, max_retries=max_retries)
        if success:
            success_count += 1
            _logger.debug(f"  {index_name} ({index_code}) K 线采集成功")
        else:
            fail_count += 1
            _logger.warning(f"  {index_name} ({index_code}) K 线采集失败")

    _logger.info(f"指数 K 线采集完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


def load_index_self_tick(date: str, max_retries: int = 3) -> Tuple[int, int]:
    """
    采集指数本身的 Tick 数据

    采集 5 个宽基指数的 Tick 数据（非成分股，是指数本身）。

    Args:
        date: 日期，格式 YYYY-MM-DD
        max_retries: 最大重试次数

    Returns:
        (成功数, 失败数) 元组
    """
    _logger.info(f"开始采集 {len(INDEX_CODES)} 个指数本身的 Tick 数据")

    success_count = 0
    fail_count = 0

    for index_code in INDEX_CODES:
        index_name = INDEX_CONSTITUENTS_CONFIG.get(index_code, ('未知', ''))[0]
        success = load_tick_l1(index_code, date, max_retries=max_retries)
        if success:
            success_count += 1
            _logger.debug(f"  {index_name} ({index_code}) Tick 采集成功")
        else:
            fail_count += 1
            _logger.warning(f"  {index_name} ({index_code}) Tick 采集失败")

    _logger.info(f"指数 Tick 采集完成: 成功 {success_count}, 失败 {fail_count}")
    return success_count, fail_count


def _download_mkline_single(args: tuple) -> bool:
    """
    下载单只股票的 K 线数据（供多线程调用）

    Args:
        args: (code, date, max_retries) 元组

    Returns:
        是否成功
    """
    code, date, max_retries = args

    # 限流
    limiter = _get_rate_limiter()
    limiter.acquire()

    try:
        success = load_mkline(code, date, max_retries=max_retries)
        if success:
            limiter.report_success()
        else:
            limiter.report_error()
        return success
    except Exception as e:
        limiter.report_error()
        _logger.debug(f"下载 {code} {date} K 线失败: {e}")
        return False


def _download_tick_single(args: tuple) -> bool:
    """
    下载单只股票的 Tick 数据（供多线程调用）

    Args:
        args: (code, date, max_retries) 元组

    Returns:
        是否成功
    """
    code, date, max_retries = args

    # 限流
    limiter = _get_rate_limiter()
    limiter.acquire()

    try:
        success = load_tick_l1(code, date, max_retries=max_retries)
        if success:
            limiter.report_success()
        else:
            limiter.report_error()
        return success
    except Exception as e:
        limiter.report_error()
        _logger.debug(f"下载 {code} {date} Tick 失败: {e}")
        return False


def load_all_mkline_parallel(
    date: str,
    stock_list: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
    max_retries: int = 3,
) -> Tuple[int, int]:
    """
    多线程并行采集全市场分钟 K 线数据

    Args:
        date: 日期，格式 YYYY-MM-DD
        stock_list: 股票列表，为 None 时自动获取全市场股票
        num_workers: 工作线程数，为 None 时使用配置默认值
        max_retries: 最大重试次数

    Returns:
        (成功数, 失败数) 元组
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .thread_safe import ProgressTracker

    if stock_list is None:
        stock_list = get_stock_list()

    if not stock_list:
        _logger.warning(f"股票列表为空，跳过 K 线采集")
        return 0, 0

    if num_workers is None:
        num_workers = EXTENDED_DATA_CONFIG.get("collection_workers", 12)

    log_interval = EXTENDED_DATA_CONFIG.get("progress_log_interval", 100)

    total = len(stock_list)
    _logger.info(f"开始多线程采集 {total} 只股票的 K 线数据 (线程数: {num_workers})")

    # 准备任务参数
    tasks = [(code, date, max_retries) for code in stock_list]

    # 创建进度追踪器
    tracker = ProgressTracker(
        total=total,
        logger=_logger,
        task_name="K线",
        log_interval=log_interval,
    )

    # 重置限流器
    reset_rate_limiter()

    # 多线程执行
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_download_mkline_single, task): task[0] for task in tasks}

        for future in as_completed(futures):
            try:
                success = future.result()
                if success:
                    tracker.report_success()
                else:
                    tracker.report_fail()
            except Exception as e:
                tracker.report_fail()
                code = futures[future]
                _logger.debug(f"任务执行异常 {code}: {e}")

    # 输出汇总
    tracker.log_summary()

    return tracker.success_count, tracker.fail_count


def load_all_tick_parallel(
    date: str,
    stock_list: Optional[List[str]] = None,
    num_workers: Optional[int] = None,
    max_retries: int = 3,
) -> Tuple[int, int]:
    """
    多线程并行采集全市场 Level 1 Tick 数据

    Args:
        date: 日期，格式 YYYY-MM-DD
        stock_list: 股票列表，为 None 时自动获取全市场股票
        num_workers: 工作线程数，为 None 时使用配置默认值
        max_retries: 最大重试次数

    Returns:
        (成功数, 失败数) 元组
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .thread_safe import ProgressTracker

    if stock_list is None:
        stock_list = get_stock_list()

    if not stock_list:
        _logger.warning(f"股票列表为空，跳过 Tick 采集")
        return 0, 0

    if num_workers is None:
        num_workers = EXTENDED_DATA_CONFIG.get("collection_workers", 12)

    log_interval = EXTENDED_DATA_CONFIG.get("progress_log_interval", 100)

    total = len(stock_list)
    _logger.info(f"开始多线程采集 {total} 只股票的 Tick 数据 (线程数: {num_workers})")

    # 准备任务参数
    tasks = [(code, date, max_retries) for code in stock_list]

    # 创建进度追踪器
    tracker = ProgressTracker(
        total=total,
        logger=_logger,
        task_name="Tick",
        log_interval=log_interval,
    )

    # 重置限流器
    reset_rate_limiter()

    # 多线程执行
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_download_tick_single, task): task[0] for task in tasks}

        for future in as_completed(futures):
            try:
                success = future.result()
                if success:
                    tracker.report_success()
                else:
                    tracker.report_fail()
            except Exception as e:
                tracker.report_fail()
                code = futures[future]
                _logger.debug(f"任务执行异常 {code}: {e}")

    # 输出汇总
    tracker.log_summary()

    return tracker.success_count, tracker.fail_count
