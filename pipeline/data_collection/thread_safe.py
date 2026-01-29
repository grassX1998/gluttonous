"""
线程安全工具模块

提供多线程数据采集所需的工具类：
- ThreadSafeCounter: 线程安全计数器
- ProgressTracker: 进度追踪器（支持多线程更新和定期日志输出）
"""

import time
import threading
import logging
from typing import Optional, Callable


class ThreadSafeCounter:
    """
    线程安全计数器

    支持多线程环境下的安全计数操作。
    """

    def __init__(self, initial: int = 0):
        """
        初始化计数器

        Args:
            initial: 初始值
        """
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        """
        增加计数

        Args:
            amount: 增加量

        Returns:
            增加后的值
        """
        with self._lock:
            self._value += amount
            return self._value

    def decrement(self, amount: int = 1) -> int:
        """
        减少计数

        Args:
            amount: 减少量

        Returns:
            减少后的值
        """
        with self._lock:
            self._value -= amount
            return self._value

    @property
    def value(self) -> int:
        """获取当前值"""
        with self._lock:
            return self._value

    def reset(self, value: int = 0):
        """重置计数器"""
        with self._lock:
            self._value = value


class ProgressTracker:
    """
    进度追踪器

    用于多线程环境下追踪任务完成进度，并定期输出日志。

    Features:
        - 线程安全的成功/失败计数
        - 定期自动输出进度日志
        - 支持自定义日志格式
        - 支持回调函数
    """

    def __init__(
        self,
        total: int,
        logger: Optional[logging.Logger] = None,
        task_name: str = "任务",
        log_interval: int = 100,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ):
        """
        初始化进度追踪器

        Args:
            total: 总任务数
            logger: 日志记录器
            task_name: 任务名称（用于日志输出）
            log_interval: 日志输出间隔（每完成多少任务输出一次）
            on_progress: 进度回调函数，参数为 (completed, success, fail)
        """
        self.total = total
        self.logger = logger or logging.getLogger(__name__)
        self.task_name = task_name
        self.log_interval = log_interval
        self.on_progress = on_progress

        self._success = ThreadSafeCounter()
        self._fail = ThreadSafeCounter()
        self._completed = ThreadSafeCounter()
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._last_log_count = 0

    def report_success(self):
        """报告任务成功"""
        self._success.increment()
        completed = self._completed.increment()
        self._maybe_log(completed)

    def report_fail(self):
        """报告任务失败"""
        self._fail.increment()
        completed = self._completed.increment()
        self._maybe_log(completed)

    def _maybe_log(self, completed: int):
        """
        判断是否需要输出日志

        Args:
            completed: 当前完成数
        """
        with self._lock:
            # 检查是否达到输出间隔或全部完成
            should_log = (
                completed - self._last_log_count >= self.log_interval
                or completed == self.total
            )
            if should_log:
                self._last_log_count = completed
                self._do_log(completed)

    def _do_log(self, completed: int):
        """
        输出日志

        Args:
            completed: 当前完成数
        """
        success = self._success.value
        fail = self._fail.value
        elapsed = time.time() - self._start_time
        progress = completed / self.total * 100 if self.total > 0 else 0

        # 计算速度
        speed = completed / elapsed if elapsed > 0 else 0

        # 估算剩余时间
        remaining = (self.total - completed) / speed if speed > 0 else 0

        self.logger.info(
            f"{self.task_name}进度: {completed}/{self.total} ({progress:.1f}%) | "
            f"成功: {success} | 失败: {fail} | "
            f"速度: {speed:.1f}/s | 剩余: {remaining:.0f}s"
        )

        # 触发回调
        if self.on_progress:
            self.on_progress(completed, success, fail)

    @property
    def success_count(self) -> int:
        """获取成功数"""
        return self._success.value

    @property
    def fail_count(self) -> int:
        """获取失败数"""
        return self._fail.value

    @property
    def completed_count(self) -> int:
        """获取完成数"""
        return self._completed.value

    @property
    def elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        return time.time() - self._start_time

    def get_summary(self) -> dict:
        """
        获取进度摘要

        Returns:
            包含进度信息的字典
        """
        elapsed = self.elapsed_time
        completed = self._completed.value
        success = self._success.value
        fail = self._fail.value

        return {
            "total": self.total,
            "completed": completed,
            "success": success,
            "fail": fail,
            "elapsed_seconds": elapsed,
            "speed": completed / elapsed if elapsed > 0 else 0,
            "progress_percent": completed / self.total * 100 if self.total > 0 else 0,
        }

    def log_summary(self):
        """输出最终汇总日志"""
        summary = self.get_summary()
        self.logger.info(
            f"{self.task_name}完成: "
            f"总数 {summary['total']} | "
            f"成功 {summary['success']} | "
            f"失败 {summary['fail']} | "
            f"耗时 {summary['elapsed_seconds']:.1f}s | "
            f"平均速度 {summary['speed']:.1f}/s"
        )
