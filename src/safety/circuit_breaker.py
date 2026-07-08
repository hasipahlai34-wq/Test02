"""
# ============================================================
# ★ 三态熔断器 (Closed → Open → Half-Open)
# ← 原项目 B 双重熔断 + 标准断路器模式
# ← WeKnora: 仅有超时配置，无熔断器 (GAP_ANALYSIS.md #4)
#
# 面试可讲:
# "我实现了标准的三态断路器模式，不是简单的 if-else 阈值判断。
# 质量熔断用滑动窗口统计答案质量分，频率熔断用令牌桶限流。
# 当熔断触发时自动降级到备选模型 (OpenAI → DeepSeek → 本地)。"
# ============================================================

本模块实现两种熔断器:

1. **质量熔断 (QualityCircuitBreaker)**:
   - 滑动窗口统计最近 N 次答案质量分
   - 失败率 ≥ 阈值 → OPEN (拒绝请求)
   - 超时后 → HALF_OPEN (探针模式)
   - 探针成功 → CLOSED (恢复正常)
   - 探针失败 → OPEN (重新熔断)

2. **频率熔断 (FrequencyCircuitBreaker)**:
   - 令牌桶算法限制 API 调用频率
   - 每分钟最多 N 次请求
   - 超限 → 拒绝请求，返回降级响应
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

from src.types import CircuitState

logger = logging.getLogger(__name__)


# ================================================================
# 质量熔断器 (★ 三态状态机)
# ================================================================


@dataclass
class QualityCircuitBreaker:
    """
    质量熔断器 — 基于答案质量分的滑动窗口熔断

    状态机: CLOSED → OPEN → HALF_OPEN → CLOSED (或 OPEN)

    用法:
        cb = QualityCircuitBreaker()
        cb.record_result(True)   # 记录一次成功
        cb.record_result(False)  # 记录一次失败
        if cb.allow_request():
            do_work()
    """

    window_size: int = 20
    failure_threshold: float = 0.3
    timeout_seconds: int = 60

    # 内部状态
    state: CircuitState = CircuitState.CLOSED
    _results: deque[bool] = field(default_factory=deque)  # 滑动窗口: True=成功, False=失败
    _failure_count: int = 0
    _opened_at: float = 0.0
    _half_open_probe_count: int = 0
    _half_open_success_count: int = 0
    _max_half_open_probes: int = 3  # Half-Open 最多放行 3 个请求

    # ----------------------------------------------------------------
    # 核心逻辑
    # ----------------------------------------------------------------

    def record_result(self, success: bool) -> None:
        """
        记录一次请求的结果
        ← 滑动窗口: 添加新结果，自动淘汰超出窗口的旧结果

        Args:
            success: 本次请求是否成功 (质量达标)
        """
        self._results.append(success)
        if not success:
            self._failure_count += 1

        # 滑动窗口: 淘汰旧结果
        while len(self._results) > self.window_size:
            old = self._results.popleft()
            if not old:
                self._failure_count -= 1

        self._update_state()

    def allow_request(self) -> bool:
        """
        判断是否允许请求通过

        Returns:
            True → 允许通过, False → 拒绝 (返回降级响应)
        """
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # 检查是否到了 Half-Open 探测时间
            if self._should_try_half_open():
                self._transition_to(CircuitState.HALF_OPEN)
                return self._allow_half_open_probe()
            return False

        if self.state == CircuitState.HALF_OPEN:
            return self._allow_half_open_probe()

        return True

    # ----------------------------------------------------------------
    # 状态转换 (★ 三态状态机核心)
    # ----------------------------------------------------------------

    def _update_state(self) -> None:
        """根据滑动窗口统计更新熔断器状态"""
        if len(self._results) < self.window_size // 2:
            return  # 样本量不够，不触发熔断

        failure_rate = self._calculate_failure_rate()

        if self.state == CircuitState.CLOSED:
            if failure_rate >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.warning(
                    "⛓️ 质量熔断: CLOSED → OPEN (失败率=%.1f%% >= %.0f%%, "
                    "失败=%d/%d)",
                    failure_rate * 100, self.failure_threshold * 100,
                    self._failure_count, self.window_size,
                )

        elif self.state == CircuitState.HALF_OPEN:
            if self._half_open_probe_count >= self._max_half_open_probes:
                success_rate = (
                    self._half_open_success_count / self._half_open_probe_count
                    if self._half_open_probe_count > 0 else 0
                )
                if success_rate >= 0.5:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info("⛓️ 质量熔断: HALF_OPEN → CLOSED (恢复)")
                else:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning("⛓️ 质量熔断: HALF_OPEN → OPEN (探测失败)")

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换 + 重置相关计数器"""
        old_state = self.state
        self.state = new_state

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            logger.info("质量熔断: %s → %s", old_state.value, new_state.value)

        if new_state == CircuitState.CLOSED:
            # 恢复后清空旧数据
            self._results.clear()
            self._failure_count = 0
            self._half_open_probe_count = 0
            self._half_open_success_count = 0
            logger.info("质量熔断: 状态恢复 → %s", new_state.value)

    def _should_try_half_open(self) -> bool:
        """检查是否到了 Half-Open 探测时间"""
        return (time.time() - self._opened_at) >= self.timeout_seconds

    def _allow_half_open_probe(self) -> bool:
        """Half-Open 模式下的探针请求放行判断"""
        if self._half_open_probe_count < self._max_half_open_probes:
            self._half_open_probe_count += 1
            logger.info(
                "质量熔断: HALF_OPEN 探针 %d/%d",
                self._half_open_probe_count, self._max_half_open_probes,
            )
            return True
        return False

    def _calculate_failure_rate(self) -> float:
        """计算滑动窗口内的失败率"""
        if len(self._results) == 0:
            return 0.0
        return self._failure_count / len(self._results)

    # ----------------------------------------------------------------
    # 状态查询
    # ----------------------------------------------------------------

    @property
    def window_failure_rate(self) -> float:
        return self._calculate_failure_rate()

    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "window_size": self.window_size,
            "window_failure_rate": f"{self.window_failure_rate:.1%}",
            "failure_count": self._failure_count,
            "total_in_window": len(self._results),
            "threshold": f"{self.failure_threshold:.0%}",
            "half_open_probes": self._half_open_probe_count,
        }


# ================================================================
# 频率熔断器 (★ 令牌桶算法)
# ================================================================


@dataclass
class FrequencyCircuitBreaker:
    """
    频率熔断器 — 基于令牌桶算法的 API 调用频率限制

    令牌桶算法:
    - 桶中有 max_tokens 个令牌
    - 每个请求需要 1 个令牌
    - 每 (60 / refill_rate) 秒补充 1 个令牌
    - 令牌耗尽 → 拒绝请求

    用法:
        cb = FrequencyCircuitBreaker(max_requests=20, refill_rate=20)
        if cb.allow_request():
            make_api_call()
    """

    max_requests: int = 20     # 每分钟最大请求数
    refill_rate: int = 20      # 每分钟令牌补充数

    # 内部状态
    state: CircuitState = CircuitState.CLOSED
    _tokens: float = 0.0       # 当前令牌数
    _last_refill: float = 0.0  # 上次补充时间

    def __post_init__(self):
        self._tokens = float(self.max_requests)
        self._last_refill = time.time()

    def allow_request(self) -> bool:
        """
        判断是否允许请求通过 (令牌桶)

        Returns:
            True → 有令牌可用，False → 令牌耗尽
        """
        self._refill()

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            self._transition_to(CircuitState.CLOSED)
            return True
        else:
            self._transition_to(CircuitState.OPEN)
            logger.warning("⛓️ 频率熔断: 令牌耗尽，拒绝请求")
            return False

    def _refill(self) -> None:
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_refill

        # 按速率补充令牌
        refill_amount = elapsed * (self.refill_rate / 60.0)
        self._tokens = min(self._tokens + refill_amount, float(self.max_requests))
        self._last_refill = now

    def _transition_to(self, new_state: CircuitState) -> None:
        """状态转换"""
        if self.state != new_state:
            self.state = new_state

    def stats(self) -> dict:
        self._refill()
        return {
            "state": self.state.value,
            "tokens_available": f"{self._tokens:.1f}",
            "max_tokens": self.max_requests,
            "refill_rate": f"{self.refill_rate}/min",
        }
