"""
# ============================================================
# 熔断器状态机测试
# ← test_circuit_breaker.py: 三态断路器单元测试
# ============================================================
"""

import pytest
from src.safety.circuit_breaker import QualityCircuitBreaker, FrequencyCircuitBreaker
from src.types import CircuitState


class TestQualityCircuitBreaker:
    """质量熔断器测试 — 三态状态机: Closed → Open → Half-Open → Closed"""

    def test_initial_state_closed(self):
        """初始状态应为 CLOSED"""
        cb = QualityCircuitBreaker(window_size=10, failure_threshold=0.3)
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_allow_request_when_closed(self):
        """CLOSED 状态应该放行所有请求"""
        cb = QualityCircuitBreaker()
        for _ in range(5):
            assert cb.allow_request() is True

    def test_transition_to_open_on_high_failure(self):
        """失败率超过阈值 → CLOSED → OPEN"""
        cb = QualityCircuitBreaker(
            window_size=10,
            failure_threshold=0.3,
            timeout_seconds=999,  # 防止自动恢复
        )
        # 记录 4 次失败 + 6 次成功 = 40% 失败率 > 30% 阈值
        for _ in range(4):
            cb.record_result(False)
        for _ in range(6):
            cb.record_result(True)

        assert cb.state == CircuitState.OPEN
        assert cb.window_failure_rate >= 0.3

    def test_stay_closed_on_low_failure(self):
        """失败率低于阈值 → 保持 CLOSED"""
        cb = QualityCircuitBreaker(window_size=10, failure_threshold=0.5)
        for _ in range(2):  # 20% 失败率
            cb.record_result(False)
        for _ in range(8):
            cb.record_result(True)

        assert cb.state == CircuitState.CLOSED

    def test_reject_request_when_open(self):
        """OPEN 状态应拒绝请求"""
        cb = QualityCircuitBreaker(
            window_size=4,
            failure_threshold=0.25,
            timeout_seconds=999,
        )
        # 触发熔断
        for _ in range(3):
            cb.record_result(False)
        cb.record_result(True)

        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_sliding_window_expires_old_entries(self):
        """滑动窗口应自动淘汰超出窗口大小的旧条目"""
        cb = QualityCircuitBreaker(window_size=5, failure_threshold=0.5)
        # 填满窗口: 5 个失败
        for _ in range(5):
            cb.record_result(False)
        assert cb.window_failure_rate == 1.0

        # 新增 5 个成功，应淘汰旧的 5 个失败
        for _ in range(5):
            cb.record_result(True)
        assert cb.window_failure_rate == 0.0


class TestFrequencyCircuitBreaker:
    """频率熔断器测试 — 令牌桶算法"""

    def test_initial_has_tokens(self):
        """初始状态应有足够令牌"""
        cb = FrequencyCircuitBreaker(max_requests=20, refill_rate=20)
        assert cb.allow_request() is True

    def test_token_consumed_on_request(self):
        """每次请求消耗一个令牌"""
        cb = FrequencyCircuitBreaker(max_requests=5, refill_rate=5)
        for _ in range(5):
            assert cb.allow_request() is True
        # 第 6 个请求应被拒绝 (令牌耗尽)
        assert cb.allow_request() is False

    def test_state_changes_on_token_exhaustion(self):
        """令牌耗尽 → 状态变为 OPEN"""
        cb = FrequencyCircuitBreaker(max_requests=3, refill_rate=3)
        for _ in range(3):
            cb.allow_request()
        cb.allow_request()  # 耗尽触发
        assert cb.state == CircuitState.OPEN
