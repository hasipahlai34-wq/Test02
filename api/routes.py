"""FastAPI routes for Adaptive RAG."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import MemorySaver

from api.schemas import (
    AskRequest,
    AskResponse,
    ChatStreamRequest,
    DiagnosticsResponse,
    DocumentUploadResponse,
    DocumentUploadResult,
    EvalRequest,
    EvalResponse,
    IngestRequest,
    IngestResponse,
    SessionDocumentsDeleteResponse,
    SessionDocumentsResponse,
)
from api.tracing import (
    build_trace_metadata,
    create_request_id,
    serialize_doc,
    sse_event,
    summarize_state,
    truncate_text,
)

_graph_app = None
_graph_checkpointer: Optional[MemorySaver] = None
_recent_session_state: dict[str, dict[str, Any]] = {}
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from src.retrieval.single_step import warmup_reranker
        from src.utils.observability import setup_langfuse

        setup_langfuse()
        warmup_reranker()
    except Exception:
        pass
    yield


app = FastAPI(
    title="Adaptive RAG API",
    description="Adaptive RAG service based on LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_graph_app():
    global _graph_app, _graph_checkpointer
    if _graph_app is None:
        from src.graph.workflow import build_adaptive_rag_graph

        _graph_checkpointer = MemorySaver()
        _graph_app = build_adaptive_rag_graph(checkpointer=_graph_checkpointer)
    return _graph_app


def _initial_graph_state(
    *,
    query: str,
    session_id: str,
    retrieval_filter: dict[str, Any] | None = None,
    stream_tokens: bool = False,
) -> dict[str, Any]:
    return {
        "query": query,
        "session_id": session_id,
        "complexity": "medium",
        "complexity_confidence": 0.5,
        "retrieved_docs": [],
        "completed": False,
        "generated_answer": "",
        "quality_passed": None,
        "quality_score": 0.0,
        "ragas_scores": {},
        "retrieval_filter": retrieval_filter,
        "hitl_status": "none",
        "hitl_decision": "",
        "safety_input_check": None,
        "safety_output_check": None,
        "final_response": "",
        "stream_tokens": stream_tokens,
    }


def _merge_stream_update(state: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "retrieved_docs" and isinstance(value, list):
            existing = state.setdefault("retrieved_docs", [])
            if isinstance(existing, list):
                existing.extend(value)
            else:
                state[key] = value
        else:
            state[key] = value


def _serialize_sources(state: dict[str, Any]) -> list[dict[str, Any]]:
    docs = state.get("retrieved_docs") or []
    return [serialize_doc(doc) for doc in docs[:8]]


def _store_recent_state(session_id: str, state: dict[str, Any]) -> None:
    sanitized = dict(summarize_state(state))
    sanitized["answer"] = truncate_text(
        state.get("final_response") or state.get("generated_answer") or "",
        4000,
    )
    sanitized["sources"] = _serialize_sources(state)
    _recent_session_state[session_id] = sanitized


@app.get("/health")
async def health():
    return {"status": "ok", "service": "adaptive-rag"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        from src.utils.observability import langfuse_trace_context, with_langfuse_config

        app_graph = get_graph_app()
        metadata = build_trace_metadata(
            entrypoint="api.ask",
            query=req.query,
            session_id=req.session_id,
            request_id=req.request_id,
            user_id=req.user_id,
            retrieval_filter=req.retrieval_filter,
        )
        config = with_langfuse_config(
            {"configurable": {"thread_id": req.session_id}},
            trace_name="adaptive-rag.api.ask",
            session_id=req.session_id,
            user_id=req.user_id,
            metadata=metadata,
            tags=["api", "frontend"],
        )
        with langfuse_trace_context(
            trace_name="adaptive-rag.api.ask",
            session_id=req.session_id,
            user_id=req.user_id,
            metadata=metadata,
            tags=["api", "frontend"],
        ) as trace:
            result = await app_graph.ainvoke(
                _initial_graph_state(
                    query=req.query,
                    session_id=req.session_id,
                    retrieval_filter=req.retrieval_filter,
                ),
                config,
            )
            if trace is not None:
                trace.update(output={
                    "answer": truncate_text(
                        result.get("final_response") or result.get("generated_answer") or "",
                        1000,
                    ),
                    "complexity": result.get("complexity"),
                    "strategy": result.get("selected_strategy"),
                    "search_count": result.get("search_count"),
                    "quality_passed": result.get("quality_passed"),
                })

        _store_recent_state(req.session_id, result)
        return AskResponse(
            query=req.query,
            answer=result.get("final_response") or result.get("generated_answer", ""),
            complexity=str(result.get("complexity", "N/A")),
            strategy=str(result.get("selected_strategy", "N/A")),
            search_count=int(result.get("search_count", 0) or 0),
            quality_score=float(result.get("quality_score", 0.0) or 0.0),
            session_id=req.session_id,
            request_id=req.request_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatStreamRequest):
    async def event_generator():
        from src.utils.observability import langfuse_trace_context, with_langfuse_config

        current_state = _initial_graph_state(
            query=req.query,
            session_id=req.session_id,
            retrieval_filter=req.retrieval_filter,
            stream_tokens=True,
        )
        metadata = build_trace_metadata(
            entrypoint="api.chat_stream",
            query=req.query,
            session_id=req.session_id,
            request_id=req.request_id,
            user_id=req.user_id,
        )
        yield sse_event("metadata", {
            "session_id": req.session_id,
            "request_id": req.request_id,
            "trace_name": "adaptive-rag.api.chat_stream",
        })

        try:
            app_graph = get_graph_app()
            config = with_langfuse_config(
                {"configurable": {"thread_id": req.session_id}},
                trace_name="adaptive-rag.api.chat_stream",
                session_id=req.session_id,
                user_id=req.user_id,
                metadata=metadata,
                tags=["api", "stream", "frontend"],
            )
            last_answer = ""
            sources_sent = False
            with langfuse_trace_context(
                trace_name="adaptive-rag.api.chat_stream",
                session_id=req.session_id,
                user_id=req.user_id,
                metadata=metadata,
                tags=["api", "stream", "frontend"],
            ) as trace:
                async for event in app_graph.astream(current_state, config, stream_mode=["updates", "custom"]):
                    mode, data = event if isinstance(event, tuple) and len(event) == 2 else ("updates", event)

                    if mode == "custom":
                        if isinstance(data, dict) and data.get("event") == "answer_delta":
                            delta = str(data.get("text") or "")
                            if delta:
                                last_answer += delta
                                yield sse_event("answer_delta", {"text": delta})
                        continue

                    if not isinstance(data, dict):
                        continue
                    for node, update in data.items():
                        if isinstance(update, dict):
                            _merge_stream_update(current_state, update)
                        summary = summarize_state(current_state)
                        summary["node"] = node
                        yield sse_event("state_update", summary)

                        answer = current_state.get("final_response") or current_state.get("generated_answer") or ""
                        if answer and answer != last_answer:
                            last_answer = answer
                            yield sse_event("answer", {"text": answer})
                if trace is not None:
                    trace.update(output={
                        "answer": truncate_text(
                            current_state.get("final_response") or current_state.get("generated_answer") or last_answer,
                            1000,
                        ),
                        "complexity": current_state.get("complexity"),
                        "strategy": current_state.get("selected_strategy"),
                        "search_count": current_state.get("search_count"),
                        "quality_passed": current_state.get("quality_passed"),
                    })

            final_answer = current_state.get("final_response") or current_state.get("generated_answer") or ""
            if final_answer and final_answer != last_answer:
                yield sse_event("answer", {"text": final_answer})
            if current_state.get("retrieved_docs") and not sources_sent:
                yield sse_event("sources", {"sources": _serialize_sources(current_state)})
            _store_recent_state(req.session_id, current_state)
            done = summarize_state(current_state)
            done["source_count"] = len(current_state.get("retrieved_docs") or [])
            yield sse_event("done", done)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield sse_event("error", {"message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    try:
        from src.ingestion.chunker import ChunkingStrategy, auto_chunk, chunk_documents
        from src.ingestion.indexer import get_document_indexer
        from src.ingestion.loader import load_document

        raw_docs = await load_document(req.filepath)
        if req.strategy and req.strategy != "auto":
            strategy = ChunkingStrategy(req.strategy)
            chunks = chunk_documents(raw_docs, strategy=strategy, chunk_size=req.chunk_size)
        else:
            chunks = auto_chunk(req.filepath)

        indexer = get_document_indexer()
        count = await indexer.index_documents(chunks)
        return IngestResponse(
            filepath=req.filepath,
            raw_segments=len(raw_docs),
            chunks=len(chunks),
            indexed=count,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    strategy: str = Form("auto"),
    chunk_size: int = Form(800),
    session_id: str | None = Form(None),
    request_id: str | None = Form(None),
    user_id: str | None = Form(None),
):
    from src.utils.observability import langfuse_trace_context

    request_id = request_id or create_request_id()
    session_id = session_id or f"upload-{uuid.uuid4()}"
    filenames = [f.filename for f in files]
    metadata = build_trace_metadata(
        entrypoint="api.documents_upload",
        session_id=session_id,
        request_id=request_id,
        user_id=user_id,
        filenames=filenames,
        chunking_mode="dynamic_structure_aware" if strategy == "auto" else "manual",
        strategy=strategy,
        chunk_size=chunk_size if strategy != "auto" else None,
    )

    results: list[DocumentUploadResult] = []
    with langfuse_trace_context(
        trace_name="adaptive-rag.api.documents_upload",
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=["api", "upload", "frontend"],
    ) as trace:
        for uploaded_file in files:
            tmp_path: str | None = None
            try:
                from src.ingestion.chunker import ChunkingStrategy, auto_chunk, chunk_documents
                from src.ingestion.indexer import get_document_indexer
                from src.ingestion.loader import load_document

                suffix = os.path.splitext(uploaded_file.filename or "")[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(await uploaded_file.read())
                    tmp_path = tmp.name

                raw_docs = await load_document(tmp_path)
                if strategy == "auto":
                    chunks = auto_chunk(tmp_path)
                else:
                    chunks = chunk_documents(
                        raw_docs,
                        strategy=ChunkingStrategy(strategy),
                        chunk_size=chunk_size,
                    )
                document_id = f"{session_id}:{uuid.uuid4()}"
                uploaded_at = datetime.now(timezone.utc).isoformat()
                source_document_id = None
                if chunks:
                    source_document_id = chunks[0].metadata.get("document_id")
                for chunk in chunks:
                    original_document_id = chunk.metadata.get("document_id")
                    if original_document_id:
                        chunk.metadata["source_document_id"] = str(original_document_id)
                    chunk.metadata["session_id"] = session_id
                    chunk.metadata["request_id"] = request_id
                    chunk.metadata["document_id"] = document_id
                    chunk.metadata["upload_filename"] = uploaded_file.filename or "unknown"
                    chunk.metadata["uploaded_via"] = "frontend"
                    chunk.metadata["uploaded_at"] = uploaded_at
                    chunk.metadata["ingest_status"] = "ready"
                    if user_id:
                        chunk.metadata["user_id"] = user_id

                indexer = get_document_indexer()
                indexed = await indexer.index_documents(chunks)
                if indexed > 0:
                    visible = await indexer.wait_until_visible({
                        "session_id": session_id,
                        "document_id": document_id,
                    })
                    if not visible:
                        logger.warning(
                            "Indexed document is not visible in immediate verification; "
                            "continuing because Chroma write returned success. session_id=%s document_id=%s indexed=%d",
                            session_id,
                            document_id,
                            indexed,
                        )
                first_meta = chunks[0].metadata if chunks else {}
                results.append(DocumentUploadResult(
                    filename=uploaded_file.filename or "unknown",
                    raw_segments=len(raw_docs),
                    chunks=len(chunks),
                    indexed=indexed,
                    status="ok",
                    document_id=document_id,
                    source_document_id=str(source_document_id) if source_document_id else None,
                    parse_quality_score=first_meta.get("parse_quality_score"),
                    outline_preview=first_meta.get("outline_preview"),
                    element_count=first_meta.get("element_count"),
                    warning_count=first_meta.get("warning_count"),
                    uploaded_at=first_meta.get("uploaded_at"),
                    chunk_strategy=first_meta.get("chunk_strategy"),
                    target_tokens=first_meta.get("chunk_target_tokens"),
                    overlap_tokens=first_meta.get("chunk_overlap_tokens"),
                    chunk_plan_reason=first_meta.get("chunk_plan_reason"),
                ))
            except Exception as e:
                results.append(DocumentUploadResult(
                    filename=uploaded_file.filename or "unknown",
                    status="error",
                    error=str(e),
                ))
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        total_indexed = sum(result.indexed for result in results if result.status == "ok")
        if trace is not None:
            trace.update(output={
                "files": len(results),
                "total_indexed": total_indexed,
                "statuses": [result.status for result in results],
            })

    total_indexed = sum(result.indexed for result in results if result.status == "ok")
    return DocumentUploadResponse(
        request_id=request_id,
        results=results,
        total_indexed=total_indexed,
    )


@app.post("/eval", response_model=EvalResponse)
async def eval_compare(req: EvalRequest):
    try:
        from src.evaluation.compare import run_comparison
        from src.utils.observability import langfuse_trace_context

        metadata = build_trace_metadata(
            entrypoint="api.eval",
            query=req.query,
            session_id=req.session_id,
            request_id=req.request_id,
            user_id=req.user_id,
        )
        with langfuse_trace_context(
            trace_name="adaptive-rag.api.eval",
            session_id=req.session_id,
            user_id=req.user_id,
            metadata=metadata,
            tags=["api", "eval", "frontend"],
        ) as trace:
            result = await run_comparison(
                req.query,
                ground_truth=req.ground_truth,
                retrieval_filter=req.retrieval_filter,
            )
            if trace is not None:
                trace.update(output={
                    "winner": result.winner,
                    "conclusion": truncate_text(result.conclusion, 1000),
                    "standard_answer_time_ms": result.standard_rag.get("answer_time_ms") or result.standard_rag.get("time_ms"),
                    "standard_ragas_eval_time_ms": result.standard_rag.get("ragas_eval_time_ms"),
                    "standard_total_time_ms": result.standard_rag.get("total_time_ms"),
                    "adaptive_answer_time_ms": result.adaptive_rag.get("answer_time_ms") or result.adaptive_rag.get("time_ms"),
                    "adaptive_ragas_eval_time_ms": result.adaptive_rag.get("ragas_eval_time_ms"),
                    "adaptive_total_time_ms": result.adaptive_rag.get("total_time_ms"),
                })
        return EvalResponse(
            query=req.query,
            direct_answer=result.direct_answer,
            standard_rag=result.standard_rag,
            adaptive_rag=result.adaptive_rag,
            conclusion=result.conclusion,
            session_id=req.session_id,
            request_id=req.request_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def list_sources():
    try:
        from src.ingestion.indexer import get_document_indexer

        indexer = get_document_indexer()
        sources = await indexer.get_sources()
        count = await indexer.count()
        return {"sources": sources, "total_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/documents", response_model=SessionDocumentsResponse)
async def session_documents(session_id: str):
    try:
        from src.ingestion.indexer import get_document_indexer

        indexer = get_document_indexer()
        documents = await indexer.list_session_documents(session_id)
        return SessionDocumentsResponse(
            session_id=session_id,
            documents=[DocumentUploadResult(**document) for document in documents],
            total_chunks=sum(int(document.get("indexed") or 0) for document in documents),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}/documents", response_model=SessionDocumentsDeleteResponse)
async def delete_session_documents(session_id: str):
    try:
        from src.ingestion.indexer import get_document_indexer

        indexer = get_document_indexer()
        deleted = await indexer.delete_by_session(session_id)
        return SessionDocumentsDeleteResponse(session_id=session_id, deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics():
    try:
        from config.settings import get_settings
        from src.utils.observability import get_perf_tracker, get_token_tracker

        settings = get_settings()
        langfuse_configured = bool(settings.langfuse_public_key and settings.langfuse_secret_key)
        return DiagnosticsResponse(
            langfuse={
                "enabled": settings.langfuse_enabled,
                "configured": langfuse_configured,
                "base_url": settings.langfuse_base_url,
            },
            performance=get_perf_tracker().stats(),
            tokens=get_token_tracker().stats,
            service={"status": "ok", "name": "adaptive-rag"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/state")
async def session_state(session_id: str):
    return _recent_session_state.get(session_id, {})
