"""Tracing and serialization helpers for the Adaptive RAG API."""

from __future__ import annotations

import json
import uuid
from typing import Any


def create_request_id() -> str:
    return str(uuid.uuid4())


def truncate_text(value: Any, limit: int = 200) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_trace_metadata(
    *,
    entrypoint: str,
    session_id: str | None = None,
    request_id: str | None = None,
    user_id: str | None = None,
    query: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "entrypoint": entrypoint,
        "session_id": session_id,
        "request_id": request_id,
        "user_id": user_id,
    }
    if query is not None:
        metadata["query"] = truncate_text(query, 200)
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple, set)):
            metadata[key] = truncate_text(json.dumps(value, ensure_ascii=False, default=str), 200)
        else:
            metadata[key] = truncate_text(value, 200)
    return metadata


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def get_doc_content(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("page_content") or doc.get("content") or doc.get("text") or "")
    return str(getattr(doc, "page_content", "") or getattr(doc, "content", "") or "")


def get_doc_metadata(doc: Any) -> dict[str, Any]:
    if isinstance(doc, dict):
        metadata = doc.get("metadata") or {}
    else:
        metadata = getattr(doc, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {"metadata": truncate_text(metadata)}
    return {str(k): _json_safe(v) for k, v in metadata.items()}


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return truncate_text(value)


def serialize_doc(doc: Any, *, content_limit: int = 1000) -> dict[str, Any]:
    metadata = get_doc_metadata(doc)
    content = truncate_text(get_doc_content(doc), content_limit)
    source = (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("file_path")
        or metadata.get("document_id")
        or "unknown"
    )
    title = metadata.get("title") or metadata.get("name") or source
    return {
        "title": str(title),
        "source": str(source),
        "content": content,
        "metadata": metadata,
    }


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "complexity": state.get("complexity"),
        "strategy": state.get("selected_strategy"),
        "search_count": state.get("search_count"),
        "quality_score": state.get("quality_score"),
        "cache_hit": state.get("cache_hit") or state.get("from_cache"),
        "quality_passed": state.get("quality_passed"),
        "safety_risk_level": state.get("safety_risk_level"),
        "hitl_status": state.get("hitl_status"),
        "ragas_scores": state.get("ragas_scores") or {},
    }
