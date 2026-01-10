# utils.py
import time
import toml
import functools
import os


def rate_limited(max_calls: int = 60, period: int = 30):
    """
    限制API调用频率的装饰器。
    :param max_calls: 最大允许的连续调用次数
    :param period: 达到最大调用次数后的休眠时间（秒）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.call_count += 1
            if wrapper.call_count > max_calls:
                print(
                    f"Reached {max_calls} API calls, sleeping for {period} seconds..."
                )
                time.sleep(period)
                wrapper.call_count = 1  # Reset the call count after sleeping
            return func(*args, **kwargs)

        wrapper.call_count = 0
        return wrapper

    return decorator


def load_existing_trading_days(file_path):
    """
    从已有的 TOML 文件中读取交易日期
    :param file_path: TOML 文件路径
    :return: 已存在的交易日期集合
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            existing_data = toml.load(file)
        return existing_data.get("trading_days", [])
    return []


def save_trading_days_to_toml(file_path, trading_days):
    """
    保存交易日期到 TOML 文件
    :param file_path: TOML 文件路径
    :param trading_days: 交易日期集合
    """
    data_to_save = {"trading_days": sorted(set(trading_days))}
    with open(file_path, "w") as file:
        toml.dump(data_to_save, file)
    print(f"Trading days saved to {file_path}")
