"""
# ============================================================
# LangSmith 全链路可观测
# ← WeKnora: internal/tracing/langfuse/ — OpenTelemetry + Langfuse
#   我们使用 LangSmith 替代 Langfuse (原项目 B 要求)
#   功能: 全链路追踪 + 节点延迟分析 + Token 消耗统计
# ============================================================

本模块负责:
- LangSmith 追踪初始化
- 自动记录 LangGraph 每个节点的执行时间、输入输出
- Token 消耗统计 (从 LLM 回调中提取)
- 错误追踪

设计要点:
- LangSmith 是 LangChain 生态的标准可观测工具
- LangGraph 内置 LangSmith 集成，只需设置环境变量即可自动追踪
- 本模块提供额外的辅助功能: 手动 trace、自定义 metadata
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Optional

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def setup_langsmith(settings: Settings | None = None) -> bool:
    """
    初始化 LangSmith 追踪
    ← WeKnora: tracing/langfuse/ 初始化 → LangSmith

    Args:
        settings: 全局配置，默认自动获取

    Returns:
        是否成功初始化
    """
    if settings is None:
        settings = get_settings()

    api_key = settings.langsmith_api_key

    if not api_key or api_key.startswith("lsv2-pt-"):
        if not api_key:
            logger.info("LangSmith API Key 未配置，追踪功能已禁用")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logger.info("LangSmith 追踪已启用 (project=%s)", settings.langsmith_project)
    return True


def get_trace_url(run_id: str) -> str:
    """
    获取 LangSmith Trace 的浏览器 URL

    Args:
        run_id: LangSmith Run ID

    Returns:
        可点击的 URL 字符串
    """
    settings = get_settings()
    endpoint = settings.langsmith_endpoint.rstrip("/")
    return f"{endpoint}/o/{settings.langsmith_project}/r/{run_id}"


# ================================================================
# Performance 追踪 (独立于 LangSmith 的轻量级计时)
# ================================================================


class PerformanceTracker:
    """
    轻量级性能追踪器
    在 LangSmith 之外提供简单的计时功能，用于终端输出和调试

    用法:
        tracker = PerformanceTracker()
        with tracker.track("search"):
            results = do_search()
        # 打印: [search] 耗时 234ms
    """

    def __init__(self):
        self._metrics: dict[str, list[float]] = {}

    @contextmanager
    def track(self, name: str, log: bool = True):
        """
        追踪一段代码的执行时间

        Args:
            name: 操作名称
            log: 是否自动打印日志
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics.setdefault(name, []).append(elapsed_ms)
            if log:
                logger.debug("[%s] 耗时 %.0fms", name, elapsed_ms)

    def stats(self) -> dict[str, dict[str, float]]:
        """
        获取所有追踪到的性能统计

        Returns:
            {name: {"count": N, "total_ms": T, "avg_ms": A, "min_ms": Min, "max_ms": Max}}
        """
        result = {}
        for name, times in self._metrics.items():
            if not times:
                continue
            result[name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }
        return result

    def summary(self) -> str:
        """
        生成可读的性能摘要

        Returns:
            格式化的多行字符串
        """
        lines = ["📊 性能统计:"]
        for name, stat in sorted(self.stats().items()):
            lines.append(
                f"  [{name}] "
                f"次数={stat['count']} "
                f"平均={stat['avg_ms']:.0f}ms "
                f"总计={stat['total_ms']:.0f}ms"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """清空所有追踪数据"""
        self._metrics.clear()


# ================================================================
# Token 消耗追踪
# ================================================================


class TokenTracker:
    """
    Token 消耗追踪器
    ← WeKnora: Langfuse → 我们改用 LangSmith 回调 + 此手动追踪器

    用法:
        tracker = TokenTracker()
        tracker.record("classify", input_tokens=200, output_tokens=50, model="gpt-4o-mini")
        tracker.record("generate", input_tokens=3000, output_tokens=500, model="gpt-4o")
        print(tracker.summary())
    """

    def __init__(self):
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        step: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = "unknown",
        total_cost: float = 0.0,
    ) -> None:
        """
        记录一次 LLM 调用的 Token 消耗

        Args:
            step: 步骤名称 (如 "classify", "generate", "review")
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            model: 模型名称
        """
        self._records.append({
            "step": step,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "total_cost": total_cost,
        })
        logger.debug(
            "[Token] %s: input=%d output=%d total=%d model=%s",
            step, input_tokens, output_tokens,
            input_tokens + output_tokens, model,
        )

    def total(self) -> int:
        """获取总 Token 消耗"""
        return sum(r["total_tokens"] for r in self._records)

    def by_step(self) -> dict[str, int]:
        """按步骤统计 Token 消耗"""
        result: dict[str, int] = {}
        for r in self._records:
            result[r["step"]] = result.get(r["step"], 0) + r["total_tokens"]
        return result

    def by_model(self) -> dict[str, int]:
        """按模型统计 Token 消耗"""
        result: dict[str, int] = {}
        for r in self._records:
            result[r["model"]] = result.get(r["model"], 0) + r["total_tokens"]
        return result

    @property
    def stats(self) -> dict[str, float | int]:
        """Return aggregate token usage for UI and diagnostics."""
        prompt_tokens = sum(r["input_tokens"] for r in self._records)
        completion_tokens = sum(r["output_tokens"] for r in self._records)
        total_cost = sum(float(r.get("total_cost", 0.0)) for r in self._records)
        return {
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_cost": total_cost,
            "requests_count": len(self._records),
        }

    def summary(self) -> str:
        """生成可读的 Token 消耗摘要"""
        by_step = self.by_step()
        by_model = self.by_model()
        total = self.total()

        lines = [f"💰 Token 消耗: 总计 {total}"]
        lines.append("  按步骤:")
        for step, count in sorted(by_step.items()):
            lines.append(f"    {step}: {count}")
        lines.append("  按模型:")
        for model, count in sorted(by_model.items()):
            lines.append(f"    {model}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        """清空所有记录"""
        self._records.clear()


# 全局追踪器实例
_perf_tracker = PerformanceTracker()
_token_tracker = TokenTracker()


def get_perf_tracker() -> PerformanceTracker:
    return _perf_tracker


def get_token_tracker() -> TokenTracker:
    return _token_tracker
