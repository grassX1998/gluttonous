"""
数据采集模块

提供掘金API数据采集功能，支持：
- 单日数据采集
- 日期范围批量采集
- 每日定时任务

使用方式：
    # 手动采集指定日期
    python -m pipeline.data_collection.collector --date 2026-01-17

    # 手动采集日期范围
    python -m pipeline.data_collection.collector --start 2026-01-01 --end 2026-01-17

    # 启动每日任务
    python -m pipeline.data_collection.daily_task
"""

# 延迟导入以避免循环依赖警告
def __getattr__(name):
    """延迟导入模块成员"""
    if name in ("get_trading_dates", "load_trading_days", "load_mkline",
                "load_instruments", "load_indexs", "test_connection", "get_stock_list"):
        from . import juejin_api
        return getattr(juejin_api, name)
    elif name in ("collect_date", "collect_date_range", "collect_today"):
        from . import collector
        return getattr(collector, name)
    elif name == "DailyTaskRunner":
        from .daily_task import DailyTaskRunner
        return DailyTaskRunner
    elif name in ("send_email", "send_collection_report", "test_email"):
        from . import notifier
        return getattr(notifier, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # juejin_api
    "get_trading_dates",
    "load_trading_days",
    "load_mkline",
    "load_instruments",
    "load_indexs",
    "test_connection",
    "get_stock_list",
    # collector
    "collect_date",
    "collect_date_range",
    "collect_today",
    # daily_task
    "DailyTaskRunner",
    # notifier
    "send_email",
    "send_collection_report",
    "test_email",
]
