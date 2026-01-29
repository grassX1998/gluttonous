"""
数据采集器

支持：
- 单日数据采集
- 日期范围批量采集
- 多线程并行采集（默认启用）
- 错误重试机制
- 进度条显示

使用方式：
    # 采集指定日期（多线程，默认12线程）
    python -m pipeline.data_collection.collector --date 2026-01-17

    # 指定线程数
    python -m pipeline.data_collection.collector --date 2026-01-17 --workers 8

    # 串行模式（禁用多线程）
    python -m pipeline.data_collection.collector --date 2026-01-17 --no-parallel

    # 采集日期范围
    python -m pipeline.data_collection.collector --start 2026-01-01 --end 2026-01-17
"""

import argparse
import time
from datetime import datetime
from typing import Optional, Tuple

from . import juejin_api
from pipeline.shared.logging_config import get_collection_logger
from pipeline.shared.config import EXTENDED_DATA_CONFIG


def collect_date(
    target_date: str,
    max_retries: int = 3,
    collect_extended: bool = True,
    collect_tick: bool = True,
    num_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Tuple[int, int]:
    """
    采集指定日期的数据

    简化后的采集流程：
    1. 采集指数本身的 K 线 + Tick（5个宽基指数）
    2. 采集全市场 K 线（多线程）
    3. 采集全市场 Tick（多线程，可选）

    Args:
        target_date: 目标日期，格式 YYYY-MM-DD
        max_retries: 最大重试次数
        collect_extended: 是否采集扩展数据（财务、行业、分红、股本）
        collect_tick: 是否采集 Level 1 Tick 数据
        num_workers: 工作线程数，None 使用配置默认值
        use_parallel: 是否使用多线程并行采集

    Returns:
        (成功数, 失败数) 元组
    """
    # 使用目标日期创建 Logger
    logger = get_collection_logger(target_date)

    # 设置 juejin_api 模块的 logger
    juejin_api.set_logger(logger)

    logger.info(f"{'=' * 60}")
    logger.info(f"开始采集 {target_date} 数据")
    logger.info(f"模式: {'多线程并行' if use_parallel else '串行'}")
    if use_parallel:
        workers = num_workers or EXTENDED_DATA_CONFIG.get("collection_workers", 12)
        logger.info(f"线程数: {workers}")
    logger.info(f"{'=' * 60}")

    start_time = time.time()

    # 更新交易日历
    juejin_api.load_trading_days()

    # 检查是否为交易日
    dates = juejin_api.get_trading_dates(target_date, target_date)
    if not dates:
        logger.warning(f"{target_date} 非交易日，跳过")
        return 0, 0

    # 更新元数据
    juejin_api.load_instruments()
    juejin_api.load_indexs(target_date)
    logger.info("元数据更新完成")

    total_success = 0
    total_fail = 0

    # ========== 阶段1: 采集指数本身 ==========
    logger.info("")
    logger.info("=" * 50)
    logger.info("阶段1: 采集指数本身的行情数据（5个宽基指数）")
    logger.info("=" * 50)

    phase1_start = time.time()

    # 指数 K 线
    idx_k_success, idx_k_fail = juejin_api.load_index_self_kline(target_date, max_retries=max_retries)
    total_success += idx_k_success
    total_fail += idx_k_fail

    # 指数 Tick
    if collect_tick:
        idx_t_success, idx_t_fail = juejin_api.load_index_self_tick(target_date, max_retries=max_retries)
        total_success += idx_t_success
        total_fail += idx_t_fail

    phase1_elapsed = time.time() - phase1_start
    logger.info(f"阶段1完成: 耗时 {phase1_elapsed:.1f}s")

    # ========== 阶段2: 采集全市场 K 线 ==========
    logger.info("")
    logger.info("=" * 50)
    logger.info("阶段2: 采集全市场 K 线数据")
    logger.info("=" * 50)

    phase2_start = time.time()

    if use_parallel:
        mkline_success, mkline_fail = juejin_api.load_all_mkline_parallel(
            target_date,
            num_workers=num_workers,
            max_retries=max_retries,
        )
    else:
        mkline_success, mkline_fail = _collect_mkline_serial(
            target_date,
            max_retries=max_retries,
            logger=logger,
        )

    total_success += mkline_success
    total_fail += mkline_fail

    phase2_elapsed = time.time() - phase2_start
    logger.info(f"阶段2完成: 成功 {mkline_success}, 失败 {mkline_fail}, 耗时 {phase2_elapsed:.1f}s")

    # ========== 阶段3: 采集全市场 Tick ==========
    if collect_tick:
        logger.info("")
        logger.info("=" * 50)
        logger.info("阶段3: 采集全市场 Tick 数据")
        logger.info("=" * 50)

        phase3_start = time.time()

        if use_parallel:
            tick_success, tick_fail = juejin_api.load_all_tick_parallel(
                target_date,
                num_workers=num_workers,
                max_retries=max_retries,
            )
        else:
            tick_success, tick_fail = _collect_tick_serial(
                target_date,
                max_retries=max_retries,
                logger=logger,
            )

        total_success += tick_success
        total_fail += tick_fail

        phase3_elapsed = time.time() - phase3_start
        logger.info(f"阶段3完成: 成功 {tick_success}, 失败 {tick_fail}, 耗时 {phase3_elapsed:.1f}s")

    # ========== 阶段4: 采集扩展数据 ==========
    if collect_extended:
        logger.info("")
        logger.info("=" * 50)
        logger.info("阶段4: 采集扩展数据（财务、行业、分红、股本）")
        logger.info("=" * 50)
        _collect_extended_data(target_date, logger)

    # 汇总
    total_elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"采集完成: {target_date}")
    logger.info(f"总成功: {total_success}, 总失败: {total_fail}")
    logger.info(f"总耗时: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} 分钟)")
    logger.info("=" * 60)

    return total_success, total_fail


def _collect_mkline_serial(target_date: str, max_retries: int, logger) -> Tuple[int, int]:
    """
    串行采集全市场 K 线数据

    Args:
        target_date: 目标日期
        max_retries: 最大重试次数
        logger: 日志记录器

    Returns:
        (成功数, 失败数) 元组
    """
    stock_list = juejin_api.get_stock_list()
    total = len(stock_list)
    logger.info(f"开始串行采集 {total} 只股票的 K 线数据")

    success_count = 0
    fail_count = 0

    for i, code in enumerate(stock_list, 1):
        success = juejin_api.load_mkline(code, target_date, max_retries=max_retries)
        if success:
            success_count += 1
        else:
            fail_count += 1

        # 每100只或完成时输出进度
        if i % 100 == 0 or i == total:
            progress = i / total * 100
            logger.info(f"K线进度: {i}/{total} ({progress:.1f}%) | 成功: {success_count} | 失败: {fail_count}")

    return success_count, fail_count


def _collect_tick_serial(target_date: str, max_retries: int, logger) -> Tuple[int, int]:
    """
    串行采集全市场 Tick 数据

    Args:
        target_date: 目标日期
        max_retries: 最大重试次数
        logger: 日志记录器

    Returns:
        (成功数, 失败数) 元组
    """
    stock_list = juejin_api.get_stock_list()
    total = len(stock_list)
    logger.info(f"开始串行采集 {total} 只股票的 Tick 数据")

    batch_size = EXTENDED_DATA_CONFIG.get("tick_batch_size", 50)
    retry_delay = EXTENDED_DATA_CONFIG.get("tick_retry_delay", 1.0)

    success_count = 0
    fail_count = 0

    for i, code in enumerate(stock_list, 1):
        success = juejin_api.load_tick_l1(code, target_date, max_retries=max_retries)
        if success:
            success_count += 1
        else:
            fail_count += 1

        # 每批次暂停一下，避免 API 限流
        if i % batch_size == 0:
            time.sleep(retry_delay)

        # 每100只或完成时输出进度
        if i % 100 == 0 or i == total:
            progress = i / total * 100
            logger.info(f"Tick进度: {i}/{total} ({progress:.1f}%) | 成功: {success_count} | 失败: {fail_count}")

    return success_count, fail_count


def _collect_extended_data(target_date: str, logger) -> None:
    """
    采集扩展数据（财务、行业、分红、股本）

    Args:
        target_date: 目标日期
        logger: 日志记录器
    """
    # 1. 财务数据（每季度更新）
    if juejin_api.should_update_finance(target_date):
        logger.info("采集财务数据...")
        if juejin_api.load_finance_data(target_date):
            logger.info("财务数据采集完成")
        else:
            logger.warning("财务数据采集部分失败")

    # 2. 行业分类数据（每月更新）
    if juejin_api.should_update_industry(target_date):
        logger.info("采集行业分类数据...")
        if juejin_api.load_industry_data(target_date):
            logger.info("行业分类数据采集完成")
        else:
            logger.warning("行业分类数据采集部分失败")

    # 3. 复权因子（每日更新）
    logger.info("采集复权因子...")
    if juejin_api.load_adj_factor(target_date):
        logger.info("复权因子采集完成")
    else:
        logger.warning("复权因子采集失败")

    # 4. 股本数据（每日更新）
    logger.info("采集股本数据...")
    if juejin_api.load_share_data(target_date):
        logger.info("股本数据采集完成")
    else:
        logger.warning("股本数据采集失败")

    logger.info("扩展数据采集完成")


def collect_date_range(
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    collect_extended: bool = True,
    collect_tick: bool = True,
    num_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Tuple[int, int]:
    """
    采集日期范围内的数据

    Args:
        start_date: 开始日期，格式 YYYY-MM-DD
        end_date: 结束日期，格式 YYYY-MM-DD
        max_retries: 最大重试次数
        collect_extended: 是否采集扩展数据
        collect_tick: 是否采集 Level 1 Tick 数据
        num_workers: 工作线程数
        use_parallel: 是否使用多线程并行采集

    Returns:
        (总成功数, 总失败数) 元组
    """
    # 使用当天日期创建 Logger（范围采集的汇总日志）
    today = datetime.now().strftime("%Y-%m-%d")
    logger = get_collection_logger(today)

    # 设置 juejin_api 模块的 logger
    juejin_api.set_logger(logger)

    logger.info(f"采集日期范围: {start_date} 至 {end_date}")

    # 获取日期范围内的交易日
    trading_dates = juejin_api.get_trading_dates(start_date, end_date)
    if not trading_dates:
        logger.warning(f"日期范围 {start_date} 至 {end_date} 内无交易日")
        return 0, 0

    logger.info(f"共 {len(trading_dates)} 个交易日待采集")

    total_success = 0
    total_fail = 0

    for date in trading_dates:
        success, fail = collect_date(
            date,
            max_retries=max_retries,
            collect_extended=collect_extended,
            collect_tick=collect_tick,
            num_workers=num_workers,
            use_parallel=use_parallel,
        )
        total_success += success
        total_fail += fail

    logger.info(f"========== 范围采集完成: 总成功 {total_success}, 总失败 {total_fail} ==========")
    return total_success, total_fail


def collect_today(
    max_retries: int = 3,
    collect_extended: bool = True,
    collect_tick: bool = True,
    num_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Tuple[int, int]:
    """
    采集当天数据

    Args:
        max_retries: 最大重试次数
        collect_extended: 是否采集扩展数据
        collect_tick: 是否采集 Level 1 Tick 数据
        num_workers: 工作线程数
        use_parallel: 是否使用多线程并行采集

    Returns:
        (成功数, 失败数) 元组
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return collect_date(
        today,
        max_retries=max_retries,
        collect_extended=collect_extended,
        collect_tick=collect_tick,
        num_workers=num_workers,
        use_parallel=use_parallel,
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="掘金数据采集器（支持多线程并行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 采集指定日期（多线程，默认12线程）
    python -m pipeline.data_collection.collector --date 2026-01-17

    # 指定线程数
    python -m pipeline.data_collection.collector --date 2026-01-17 --workers 8

    # 串行模式（禁用多线程）
    python -m pipeline.data_collection.collector --date 2026-01-17 --no-parallel

    # 采集日期范围
    python -m pipeline.data_collection.collector --start 2026-01-01 --end 2026-01-17
        """,
    )
    parser.add_argument(
        "--date",
        type=str,
        help="采集指定日期，格式 YYYY-MM-DD",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="开始日期，格式 YYYY-MM-DD（与 --end 配合使用）",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="结束日期，格式 YYYY-MM-DD（与 --start 配合使用）",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="最大重试次数（默认: 3）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="工作线程数（默认: 12）",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="禁用多线程，使用串行模式",
    )
    parser.add_argument(
        "--no-extended",
        action="store_true",
        help="不采集扩展数据（财务、行业、分红、股本）",
    )
    parser.add_argument(
        "--no-tick",
        action="store_true",
        help="不采集 Level 1 Tick 数据",
    )

    args = parser.parse_args()

    # 参数校验
    if args.date and (args.start or args.end):
        parser.error("--date 不能与 --start/--end 同时使用")

    if (args.start and not args.end) or (args.end and not args.start):
        parser.error("--start 和 --end 必须同时指定")

    # 执行采集
    collect_extended = not args.no_extended
    collect_tick = not args.no_tick
    use_parallel = not args.no_parallel

    if args.date:
        collect_date(
            args.date,
            max_retries=args.retries,
            collect_extended=collect_extended,
            collect_tick=collect_tick,
            num_workers=args.workers,
            use_parallel=use_parallel,
        )
    elif args.start and args.end:
        collect_date_range(
            args.start,
            args.end,
            max_retries=args.retries,
            collect_extended=collect_extended,
            collect_tick=collect_tick,
            num_workers=args.workers,
            use_parallel=use_parallel,
        )
    else:
        # 默认采集当天
        collect_today(
            max_retries=args.retries,
            collect_extended=collect_extended,
            collect_tick=collect_tick,
            num_workers=args.workers,
            use_parallel=use_parallel,
        )


if __name__ == "__main__":
    main()
