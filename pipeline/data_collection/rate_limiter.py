"""
限流器模块

提供 API 请求限流功能：
- TokenBucketRateLimiter: 令牌桶限流器
- AdaptiveRateLimiter: 自适应限流器（根据错误率动态调整速率）
"""

import time
import threading
from typing import Optional


class TokenBucketRateLimiter:
    """
    令牌桶限流器

    实现令牌桶算法，控制 API 请求速率。

    Attributes:
        rate: 每秒生成令牌数
        capacity: 令牌桶容量（突发上限）
    """

    def __init__(self, rate: float, capacity: float):
        """
        初始化令牌桶限流器

        Args:
            rate: 每秒生成令牌数（QPS）
            capacity: 令牌桶容量（突发请求上限）
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_time = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """
        获取令牌

        如果令牌不足，会阻塞等待直到有足够的令牌或超时。

        Args:
            tokens: 需要获取的令牌数
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            是否成功获取令牌
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                # 计算新增令牌
                now = time.monotonic()
                elapsed = now - self._last_time
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last_time = now

                # 尝试获取令牌
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # 计算等待时间
                wait_time = (tokens - self._tokens) / self.rate

            # 检查超时
            if timeout is not None:
                elapsed_total = time.monotonic() - start_time
                if elapsed_total + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed_total)

            # 等待
            time.sleep(min(wait_time, 0.1))  # 最多等待 0.1 秒后重试

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        尝试获取令牌（非阻塞）

        Args:
            tokens: 需要获取的令牌数

        Returns:
            是否成功获取令牌
        """
        with self._lock:
            # 计算新增令牌
            now = time.monotonic()
            elapsed = now - self._last_time
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last_time = now

            # 尝试获取令牌
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """获取当前可用令牌数"""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_time
            return min(self.capacity, self._tokens + elapsed * self.rate)


class AdaptiveRateLimiter:
    """
    自适应限流器

    根据错误率动态调整请求速率：
    - 错误率升高时降低速率
    - 错误率降低时逐步恢复速率

    Attributes:
        initial_rate: 初始每秒请求数
        min_rate: 最小每秒请求数
        max_rate: 最大每秒请求数
        error_threshold: 错误率阈值（超过则降速）
        recovery_factor: 恢复系数（每次成功后乘以此系数）
        backoff_factor: 退避系数（每次失败后除以此系数）
    """

    def __init__(
        self,
        initial_rate: float = 50,
        capacity: float = 100,
        min_rate: float = 5,
        max_rate: float = 100,
        error_threshold: float = 0.1,
        recovery_factor: float = 1.05,
        backoff_factor: float = 2.0,
        window_size: int = 100,
    ):
        """
        初始化自适应限流器

        Args:
            initial_rate: 初始每秒请求数
            capacity: 令牌桶容量
            min_rate: 最小每秒请求数
            max_rate: 最大每秒请求数
            error_threshold: 错误率阈值
            recovery_factor: 恢复系数
            backoff_factor: 退避系数
            window_size: 统计窗口大小
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.error_threshold = error_threshold
        self.recovery_factor = recovery_factor
        self.backoff_factor = backoff_factor
        self.window_size = window_size

        self._current_rate = initial_rate
        self._limiter = TokenBucketRateLimiter(initial_rate, capacity)
        self._lock = threading.Lock()

        # 滑动窗口统计
        self._success_count = 0
        self._error_count = 0
        self._total_count = 0

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取令牌

        Args:
            timeout: 超时时间

        Returns:
            是否成功获取令牌
        """
        return self._limiter.acquire(1.0, timeout)

    def report_success(self):
        """报告请求成功"""
        with self._lock:
            self._success_count += 1
            self._total_count += 1
            self._maybe_adjust_rate()

    def report_error(self):
        """报告请求失败"""
        with self._lock:
            self._error_count += 1
            self._total_count += 1
            self._maybe_adjust_rate()

    def _maybe_adjust_rate(self):
        """
        根据统计数据调整速率

        在窗口期满后计算错误率并调整。
        """
        if self._total_count < self.window_size:
            return

        # 计算错误率
        error_rate = self._error_count / self._total_count

        # 调整速率
        if error_rate > self.error_threshold:
            # 降速
            new_rate = max(self.min_rate, self._current_rate / self.backoff_factor)
        else:
            # 恢复速率
            new_rate = min(self.max_rate, self._current_rate * self.recovery_factor)

        # 更新限流器
        if new_rate != self._current_rate:
            self._current_rate = new_rate
            self._limiter = TokenBucketRateLimiter(new_rate, self._limiter.capacity)

        # 重置计数器
        self._success_count = 0
        self._error_count = 0
        self._total_count = 0

    @property
    def current_rate(self) -> float:
        """获取当前速率"""
        return self._current_rate

    @property
    def error_rate(self) -> float:
        """获取当前窗口的错误率"""
        with self._lock:
            if self._total_count == 0:
                return 0.0
            return self._error_count / self._total_count

    def reset(self):
        """重置限流器状态"""
        with self._lock:
            self._current_rate = self.initial_rate
            self._limiter = TokenBucketRateLimiter(self.initial_rate, self._limiter.capacity)
            self._success_count = 0
            self._error_count = 0
            self._total_count = 0
