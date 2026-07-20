"""Pydantic schemas for the Adaptive RAG FastAPI surface."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class TracedRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None


class AskRequest(TracedRequest):
    query: str = Field(..., min_length=1)
    retrieval_filter: dict[str, Any] | None = None


class AskResponse(BaseModel):
    query: str
    answer: str
    complexity: str
    strategy: str
    search_count: int
    quality_score: float
    session_id: str
    request_id: str


class ChatStreamRequest(AskRequest):
    retrieval_filter: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    filepath: str
    strategy: str = "auto"
    chunk_size: int = 800


class IngestResponse(BaseModel):
    filepath: str
    raw_segments: int
    chunks: int
    indexed: int


class DocumentUploadResult(BaseModel):
    filename: str
    raw_segments: int = 0
    chunks: int = 0
    indexed: int = 0
    status: Literal["ok", "error"]
    document_id: str | None = None
    source_document_id: str | None = None
    parse_quality_score: float | None = None
    outline_preview: str | None = None
    element_count: int | None = None
    warning_count: int | None = None
    uploaded_at: str | None = None
    chunk_strategy: str | None = None
    target_tokens: int | None = None
    overlap_tokens: int | None = None
    chunk_plan_reason: str | None = None
    error: str | None = None


class DocumentUploadResponse(BaseModel):
    request_id: str
    results: list[DocumentUploadResult]
    total_indexed: int


class SessionDocumentsResponse(BaseModel):
    session_id: str
    documents: list[DocumentUploadResult]
    total_chunks: int


class SessionDocumentsDeleteResponse(BaseModel):
    session_id: str
    deleted: int


class EvalRequest(TracedRequest):
    query: str = Field(..., min_length=1)
    ground_truth: str | None = None
    retrieval_filter: dict[str, Any] | None = None


class EvalResponse(BaseModel):
    query: str
    direct_answer: dict[str, Any]
    standard_rag: dict[str, Any]
    adaptive_rag: dict[str, Any]
    conclusion: str
    session_id: str | None = None
    request_id: str | None = None


class DiagnosticsResponse(BaseModel):
    langfuse: dict[str, Any]
    performance: dict[str, Any]
    tokens: dict[str, Any]
    service: dict[str, Any]
