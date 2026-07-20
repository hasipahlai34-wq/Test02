"""Observability helpers for LangSmith, Langfuse, timing, and token tracking."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_langfuse_callback_handler: Any | None = None
_langfuse_atexit_registered = False
_langfuse_setup_status: bool | None = None


def setup_langsmith(settings: Settings | None = None) -> bool:
    """Configure LangSmith environment variables when credentials exist."""
    if settings is None:
        settings = get_settings()

    api_key = settings.langsmith_api_key
    if not api_key or api_key.startswith("lsv2-pt-"):
        if not api_key:
            logger.info("LangSmith API key is not configured; tracing is disabled")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    return True


def setup_langfuse(settings: Settings | None = None) -> bool:
    """Configure Langfuse SDK environment variables when credentials exist."""
    global _langfuse_setup_status
    if settings is None and _langfuse_setup_status is not None:
        return _langfuse_setup_status

    if settings is None:
        settings = get_settings()

    if not settings.langfuse_enabled:
        logger.info("Langfuse tracing disabled by LANGFUSE_ENABLED=false")
        _langfuse_setup_status = False
        return False

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.info("Langfuse keys are not configured; tracing is disabled")
        _langfuse_setup_status = False
        return False

    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_base_url
    os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_base_url

    _register_langfuse_flush()
    logger.info("Langfuse tracing enabled (host=%s)", settings.langfuse_base_url)
    _langfuse_setup_status = True
    return True


def _register_langfuse_flush() -> None:
    global _langfuse_atexit_registered
    if _langfuse_atexit_registered:
        return

    import atexit

    atexit.register(flush_langfuse)
    _langfuse_atexit_registered = True


def _load_langfuse_callback_handler():
    try:
        from langfuse.langchain import CallbackHandler
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler
        except ImportError:
            return None
    return CallbackHandler


def get_langfuse_callback_handler(settings: Settings | None = None):
    """Return a reusable LangChain callback handler, or None when unavailable."""
    global _langfuse_callback_handler

    if _langfuse_callback_handler is not None:
        return _langfuse_callback_handler

    if settings is None:
        settings = get_settings()
    if not setup_langfuse(settings):
        return None

    callback_handler = _load_langfuse_callback_handler()
    if callback_handler is None:
        logger.warning("Langfuse package is not installed; run `pip install langfuse`")
        return None

    try:
        _langfuse_callback_handler = callback_handler()
    except TypeError:
        _langfuse_callback_handler = callback_handler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_base_url,
        )
    return _langfuse_callback_handler


def _sanitize_langfuse_metadata(metadata: dict[str, Any] | None = None) -> dict[str, str]:
    """Langfuse v4 trace metadata must be short string values."""
    sanitized: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        sanitized[str(key)] = str(value)[:200]
    return sanitized


def _compact_langfuse_value(value: Any, limit: int = 2000) -> Any:
    """Keep traced input/output readable and bounded."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
        return text if len(text) <= limit else text[: limit - 1] + "…"
    if isinstance(value, dict):
        return {
            str(key): _compact_langfuse_value(item, limit=limit // 2)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, (list, tuple)):
        return [_compact_langfuse_value(item, limit=limit // 2) for item in list(value)[:20]]
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dedupe_tags(tags: list[str] | None = None) -> list[str]:
    deduped: list[str] = []
    for tag in tags or []:
        if tag and tag not in deduped:
            deduped.append(tag)
    return deduped


@contextmanager
def langfuse_trace_context(
    *,
    trace_name: str = "adaptive-rag",
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[Any | None]:
    """Create a request-level Langfuse root chain and propagate trace attributes."""
    if not setup_langfuse():
        yield None
        return

    try:
        from langfuse import get_client, propagate_attributes
    except ImportError:
        logger.warning("Langfuse package is not installed; run `pip install langfuse`")
        yield None
        return

    sanitized_metadata = _sanitize_langfuse_metadata(metadata)
    input_payload = {
        "query": sanitized_metadata.get("query"),
        "session_id": session_id,
        "request_id": sanitized_metadata.get("request_id"),
        "entrypoint": sanitized_metadata.get("entrypoint"),
    }

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        tags=_dedupe_tags([*(tags or []), "adaptive-rag"]),
        metadata=sanitized_metadata,
    ):
        with get_client().start_as_current_observation(
            name=trace_name,
            as_type="chain",
            input=_compact_langfuse_value(input_payload),
            metadata=sanitized_metadata,
        ) as observation:
            try:
                yield observation
            except Exception as e:
                observation.update(level="ERROR", status_message=str(e)[:500])
                raise


@contextmanager
def langfuse_observation(
    *,
    name: str,
    as_type: str = "span",
    input: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Create a nested Langfuse observation for custom Python business logic."""
    if not setup_langfuse():
        yield None
        return

    try:
        from langfuse import get_client
    except ImportError:
        logger.warning("Langfuse package is not installed; run `pip install langfuse`")
        yield None
        return

    with get_client().start_as_current_observation(
        name=name,
        as_type=as_type,
        input=_compact_langfuse_value(input),
        metadata=_sanitize_langfuse_metadata(metadata),
    ) as observation:
        try:
            yield observation
        except Exception as e:
            observation.update(level="ERROR", status_message=str(e)[:500])
            raise


def with_langfuse_config(
    config: dict | None = None,
    *,
    trace_name: str = "adaptive-rag",
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Attach Langfuse callback and trace metadata to a LangChain/LangGraph config."""
    merged = dict(config or {})
    handler = get_langfuse_callback_handler()
    if handler is None:
        return merged

    callbacks = list(merged.get("callbacks") or [])
    if handler not in callbacks:
        callbacks.append(handler)
    merged["callbacks"] = callbacks

    merged.setdefault("run_name", trace_name)

    merged_metadata = _sanitize_langfuse_metadata(merged.get("metadata") or {})
    merged_metadata.update(_sanitize_langfuse_metadata({
        "session_id": session_id,
        "langfuse_session_id": session_id,
        "user_id": user_id,
        "langfuse_user_id": user_id,
        "trace_name": trace_name,
    }))
    merged_metadata.update(_sanitize_langfuse_metadata(metadata))
    merged["metadata"] = merged_metadata

    merged["tags"] = _dedupe_tags([
        *(merged.get("tags") or []),
        *(tags or []),
        "adaptive-rag",
    ])
    return merged


def flush_langfuse() -> None:
    """Flush buffered Langfuse events during CLI/API shutdown."""
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        logger.debug("Langfuse flush skipped: %s", e)


def get_trace_url(run_id: str) -> str:
    """Return a LangSmith trace URL for compatibility with existing callers."""
    settings = get_settings()
    endpoint = settings.langsmith_endpoint.rstrip("/")
    return f"{endpoint}/o/{settings.langsmith_project}/r/{run_id}"


class PerformanceTracker:
    """Small in-process performance tracker."""

    def __init__(self):
        self._metrics: dict[str, list[float]] = {}

    @contextmanager
    def track(self, name: str, log: bool = True):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._metrics.setdefault(name, []).append(elapsed_ms)
            if log:
                logger.debug("[%s] elapsed %.0fms", name, elapsed_ms)

    def stats(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
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
        lines = ["Performance stats:"]
        for name, stat in sorted(self.stats().items()):
            lines.append(
                f"  [{name}] count={stat['count']} "
                f"avg={stat['avg_ms']:.0f}ms total={stat['total_ms']:.0f}ms"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        self._metrics.clear()


class TokenTracker:
    """Small in-process token usage tracker."""

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
            step,
            input_tokens,
            output_tokens,
            input_tokens + output_tokens,
            model,
        )

    def total(self) -> int:
        return sum(r["total_tokens"] for r in self._records)

    def by_step(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for r in self._records:
            result[r["step"]] = result.get(r["step"], 0) + r["total_tokens"]
        return result

    def by_model(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for r in self._records:
            result[r["model"]] = result.get(r["model"], 0) + r["total_tokens"]
        return result

    @property
    def stats(self) -> dict[str, float | int]:
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
        lines = [f"Token usage total: {self.total()}"]
        lines.append("  By step:")
        for step, count in sorted(self.by_step().items()):
            lines.append(f"    {step}: {count}")
        lines.append("  By model:")
        for model, count in sorted(self.by_model().items()):
            lines.append(f"    {model}: {count}")
        return "\n".join(lines)

    def reset(self) -> None:
        self._records.clear()


_perf_tracker = PerformanceTracker()
_token_tracker = TokenTracker()


def get_perf_tracker() -> PerformanceTracker:
    return _perf_tracker


def get_token_tracker() -> TokenTracker:
    return _token_tracker
