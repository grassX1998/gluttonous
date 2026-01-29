"""
每日任务调度器

在指定时间后（默认17:00）自动执行数据采集任务。

特性：
- 单实例运行保护（通过锁文件）
- 结果报告保存到项目根目录
- 详细日志记录

使用方式：
    # 启动每日任务（前台运行）
    python -m pipeline.data_collection.daily_task

    # 后台运行（Linux/Mac）
    nohup python -m pipeline.data_collection.daily_task &

    # 测试模式（立即执行一次）
    python -m pipeline.data_collection.daily_task --test
"""

import argparse
import atexit
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from pipeline.shared.config import PIPELINE_DATA_ROOT, DAILY_TASK_CONFIG, PROJECT_ROOT
from pipeline.shared.logging_config import get_collection_logger
from .collector import collect_today
from .notifier import send_collection_report

# 锁文件路径
LOCK_FILE = PIPELINE_DATA_ROOT / "daily_task.lock"

# 状态文件路径（放在项目根目录，方便查看）
STATUS_FILE = PROJECT_ROOT / "DATA_COLLECTION_STATUS.json"


class SingleInstanceLock:
    """
    单实例锁

    使用文件锁确保只有一个进程在运行。
    """

    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self.locked = False

    def acquire(self) -> bool:
        """
        尝试获取锁

        Returns:
            是否成功获取锁
        """
        if self.lock_file.exists():
            # 检查锁文件中的 PID 是否还在运行
            try:
                with open(self.lock_file, "r") as f:
                    old_pid = int(f.read().strip())
                # 检查进程是否存在
                if self._is_process_running(old_pid):
                    return False
                # 旧进程已不存在，删除锁文件
                self.lock_file.unlink()
            except (ValueError, FileNotFoundError):
                # 锁文件损坏或已被删除
                pass

        # 创建新锁文件
        try:
            with open(self.lock_file, "w") as f:
                f.write(str(os.getpid()))
            self.locked = True
            return True
        except Exception:
            return False

    def release(self):
        """释放锁"""
        if self.locked and self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except Exception:
                pass
            self.locked = False

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """检查进程是否在运行"""
        try:
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                os.kill(pid, 0)
                return True
        except (OSError, ProcessLookupError):
            return False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError("无法获取单实例锁，可能已有另一个实例在运行")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def save_status(
    status: str,
    date: str,
    success_count: int = 0,
    fail_count: int = 0,
    error: Optional[str] = None,
    last_run: Optional[str] = None,
    retry_count: int = 0,
    max_retries: int = 0,
):
    """
    保存状态到项目根目录的 JSON 文件

    Args:
        status: 状态（running/completed/failed/retrying）
        date: 采集日期
        success_count: 成功数
        fail_count: 失败数
        error: 错误信息
        last_run: 上次运行日期
        retry_count: 当前重试次数
        max_retries: 最大重试次数
    """
    data = {
        "status": status,
        "date": date,
        "updated_at": datetime.now().isoformat(),
        "success_count": success_count,
        "fail_count": fail_count,
        "last_successful_run": last_run,
    }
    if error:
        data["error"] = error
    if retry_count > 0 or max_retries > 0:
        data["retry_count"] = retry_count
        data["max_retries"] = max_retries

    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class DailyTaskRunner:
    """
    每日任务运行器

    在指定时间后自动执行数据采集任务。
    支持单实例运行保护和失败自动重试。
    """

    def __init__(
        self,
        run_after: str = None,
        check_interval: int = None,
        max_retries: int = None,
        retry_interval: int = None,
    ):
        """
        初始化运行器

        Args:
            run_after: 运行时间阈值，格式 HH:MM:SS（默认从配置读取）
            check_interval: 检查间隔（秒），默认从配置读取
            max_retries: 最大重试次数（默认从配置读取）
            retry_interval: 重试间隔（秒），默认从配置读取
        """
        self.run_after = run_after or DAILY_TASK_CONFIG["run_after"]
        self.check_interval = check_interval or DAILY_TASK_CONFIG["check_interval"]
        self.max_retries = max_retries if max_retries is not None else DAILY_TASK_CONFIG.get("max_retries", 3)
        self.retry_interval = retry_interval or DAILY_TASK_CONFIG.get("retry_interval", 1800)
        self.last_run_date = None
        self.lock = SingleInstanceLock(LOCK_FILE)
        # 重试状态
        self.retry_count = 0
        self.last_error = None
        # 缓存 logger，避免频繁创建
        self._logger = None
        self._logger_date = None

    def _get_logger(self):
        """获取当天的日志器（带缓存，避免文件句柄泄漏）"""
        today = datetime.now().strftime("%Y-%m-%d")
        # 如果日期变了或者还没创建，重新获取 logger
        if self._logger is None or self._logger_date != today:
            # 关闭旧的 handlers
            if self._logger is not None:
                for handler in self._logger.handlers[:]:
                    handler.close()
                    self._logger.removeHandler(handler)
            self._logger = get_collection_logger(today)
            self._logger_date = today
        return self._logger

    def should_run(self, current_date: str, current_time: str) -> bool:
        """
        判断是否应该运行任务

        Args:
            current_date: 当前日期
            current_time: 当前时间

        Returns:
            是否应该运行
        """
        # 今天还没运行过，且当前时间超过运行阈值
        return (
            current_date != self.last_run_date
            and current_time >= self.run_after
        )

    def run_once(self) -> bool:
        """
        执行一次数据采集

        Returns:
            是否成功
        """
        current_date = datetime.now().strftime("%Y-%m-%d")

        try:
            self._get_logger().info("开始执行数据采集任务")
            save_status("running", current_date, last_run=self.last_run_date)

            success, fail = collect_today()

            self._get_logger().info(f"数据采集完成: 成功 {success}, 失败 {fail}")

            if fail == 0:
                save_status("completed", current_date, success, fail, last_run=current_date)
                # 发送成功邮件
                send_collection_report(current_date, success, fail, "completed")
                return True
            else:
                save_status("completed_with_errors", current_date, success, fail, last_run=self.last_run_date)
                # 发送部分失败邮件
                send_collection_report(current_date, success, fail, "completed_with_errors")
                return True  # 部分失败仍标记为运行过

        except Exception as e:
            self._get_logger().error(f"数据采集失败: {e}")
            self.last_error = str(e)
            save_status("failed", current_date, error=str(e), last_run=self.last_run_date)
            # 发送失败邮件
            send_collection_report(current_date, 0, 0, "failed", str(e))
            return False

    def run_with_retry(self) -> bool:
        """
        执行数据采集，失败时自动重试

        Returns:
            是否最终成功
        """
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.retry_count = 0
        self.last_error = None

        while self.retry_count <= self.max_retries:
            if self.retry_count > 0:
                self._get_logger().info(f"第 {self.retry_count}/{self.max_retries} 次重试...")
                save_status(
                    "retrying", current_date,
                    error=self.last_error,
                    last_run=self.last_run_date,
                    retry_count=self.retry_count,
                    max_retries=self.max_retries
                )

            success = self.run_once()

            if success:
                self.retry_count = 0
                self.last_error = None
                return True

            self.retry_count += 1

            if self.retry_count <= self.max_retries:
                wait_minutes = self.retry_interval // 60
                self._get_logger().warning(
                    f"任务执行失败，{wait_minutes} 分钟后重试 "
                    f"({self.retry_count}/{self.max_retries})"
                )
                time.sleep(self.retry_interval)
            else:
                self._get_logger().error(f"任务失败，已达最大重试次数 ({self.max_retries})")
                save_status(
                    "failed_max_retries", current_date,
                    error=f"已重试 {self.max_retries} 次仍失败: {self.last_error}",
                    last_run=self.last_run_date,
                    retry_count=self.retry_count,
                    max_retries=self.max_retries
                )

        return False

    def run(self):
        """
        启动调度器（循环运行）
        """
        # 尝试获取单实例锁
        if not self.lock.acquire():
            self._get_logger().error("无法启动：已有另一个实例在运行")
            save_status("blocked", datetime.now().strftime("%Y-%m-%d"),
                       error="无法启动：已有另一个实例在运行")
            sys.exit(1)

        # 注册退出时释放锁
        atexit.register(self.lock.release)

        self._get_logger().info("=" * 50)
        self._get_logger().info("每日任务调度器启动")
        self._get_logger().info(f"运行时间: {self.run_after} 之后")
        self._get_logger().info(f"检查间隔: {self.check_interval} 秒")
        self._get_logger().info(f"最大重试: {self.max_retries} 次")
        self._get_logger().info(f"重试间隔: {self.retry_interval // 60} 分钟")
        self._get_logger().info(f"状态文件: {STATUS_FILE}")
        self._get_logger().info("=" * 50)

        # 初始化状态文件
        save_status("idle", datetime.now().strftime("%Y-%m-%d"), last_run=self.last_run_date)

        check_count = 0
        try:
            while True:
                try:
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    current_time = datetime.now().strftime("%H:%M:%S")

                    check_count += 1
                    # 每小时输出一次状态日志
                    if check_count % (3600 // self.check_interval) == 1:
                        self._get_logger().info(
                            f"状态检查 | 日期: {current_date} 时间: {current_time} "
                            f"上次运行: {self.last_run_date or '无'}"
                        )

                    if self.should_run(current_date, current_time):
                        success = self.run_with_retry()
                        if success:
                            self.last_run_date = current_date

                    time.sleep(self.check_interval)

                except KeyboardInterrupt:
                    raise  # 重新抛出，让外层处理
                except Exception as e:
                    # 捕获循环内的其他异常，记录但继续运行
                    try:
                        self._get_logger().error(f"循环内异常: {e}", exc_info=True)
                    except Exception:
                        # 如果连日志都写不了，写到状态文件
                        save_status("error", datetime.now().strftime("%Y-%m-%d"),
                                   error=f"循环内异常: {e}", last_run=self.last_run_date)
                    # 等待一段时间后继续
                    time.sleep(60)

        except KeyboardInterrupt:
            self._get_logger().info("收到中断信号，正在退出...")
        except Exception as e:
            # 捕获所有未处理的异常
            try:
                self._get_logger().error(f"调度器致命错误: {e}", exc_info=True)
            except Exception:
                pass
            save_status("crashed", datetime.now().strftime("%Y-%m-%d"),
                       error=f"致命错误: {e}", last_run=self.last_run_date)
        finally:
            self.lock.release()
            save_status("stopped", datetime.now().strftime("%Y-%m-%d"), last_run=self.last_run_date)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="每日数据采集任务调度器")
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式：立即执行一次采集",
    )
    parser.add_argument(
        "--run-after",
        type=str,
        default=None,
        help="运行时间阈值，格式 HH:MM:SS（默认: 17:00:00）",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="检查间隔（秒），默认: 600",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="显示当前状态",
    )

    args = parser.parse_args()

    # 显示状态
    if args.status:
        if STATUS_FILE.exists():
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status = json.load(f)
            print(json.dumps(status, ensure_ascii=False, indent=2))
        else:
            print("状态文件不存在，任务可能从未运行过")
        return

    runner = DailyTaskRunner(
        run_after=args.run_after,
        check_interval=args.interval,
    )

    if args.test:
        # 测试模式：立即执行一次（不检查单实例锁）
        runner.run_once()
    else:
        # 正常模式：启动调度器
        runner.run()


if __name__ == "__main__":
    main()
