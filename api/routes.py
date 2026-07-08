"""
# ============================================================
# FastAPI 端点
# ← WeKnora: internal/handler/ HTTP 处理层
#   简化: Gin → FastAPI，去掉中间件链 (CORS/Auth/RBAC/Audit)
# ============================================================
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langgraph.checkpoint.memory import MemorySaver


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时后台预热 Reranker 模型，消除首次查询卡顿。"""
    try:
        from src.retrieval.single_step import warmup_reranker
        warmup_reranker()  # 独立 daemon 线程，不阻塞启动
    except Exception:
        pass  # 预热失败不阻止应用启动
    yield


app = FastAPI(
    title="Adaptive RAG API",
    description="自适应文档问答系统 — 基于 LangGraph 的 Adaptive-RAG 学术前沿落地项目",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Graph 单例 — 避免每次请求重建 StateGraph + MemorySaver
# MemorySaver 持久化确保 session_id 对话历史连续性
# ================================================================

_graph_app = None
_graph_checkpointer: Optional[MemorySaver] = None


def get_graph_app():
    """获取 Adaptive-RAG Graph 应用单例。

    使用模块级懒加载确保:
    1. StateGraph 只编译一次 (节点注册 + 条件边)
    2. MemorySaver 单例化 — 同一进程内所有请求共享检查点
    3. session_id 对话历史在请求间保持连续

    Returns:
        编译后的 StateGraph 实例
    """
    global _graph_app, _graph_checkpointer
    if _graph_app is None:
        from src.graph.workflow import build_adaptive_rag_graph
        _graph_checkpointer = MemorySaver()
        _graph_app = build_adaptive_rag_graph(checkpointer=_graph_checkpointer)
    return _graph_app


# ================================================================
# 请求/响应模型
# ================================================================


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户查询")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class AskResponse(BaseModel):
    query: str
    answer: str
    complexity: str
    strategy: str
    search_count: int
    quality_score: float
    session_id: str


class IngestRequest(BaseModel):
    filepath: str = Field(..., description="文档文件路径")
    strategy: str = Field(default="recursive", description="分块策略")
    chunk_size: int = Field(default=800, description="分块大小")


class IngestResponse(BaseModel):
    filepath: str
    raw_segments: int
    chunks: int
    indexed: int


class EvalRequest(BaseModel):
    query: str = Field(..., min_length=1)


class EvalResponse(BaseModel):
    query: str
    direct_answer: dict
    standard_rag: dict
    adaptive_rag: dict
    conclusion: str


# ================================================================
# 端点
# ================================================================


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "service": "adaptive-rag"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    问答接口 — Adaptive-RAG 完整流程
    <- WeKnora: handler/session/qa.go KnowledgeQA() / AgentQA()
    """
    try:
        from src.graph.state import GraphState

        app_graph = get_graph_app()
        result = await app_graph.ainvoke(
            GraphState(query=req.query, session_id=req.session_id),
            {"configurable": {"thread_id": req.session_id}},
        )

        return AskResponse(
            query=req.query,
            answer=result.get("generated_answer", ""),
            complexity=result.get("complexity", "N/A"),
            strategy=result.get("selected_strategy", "N/A"),
            search_count=result.get("search_count", 0),
            quality_score=result.get("quality_score", 0.0),
            session_id=req.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """
    文档摄入接口
    ← WeKnora: handler/knowledge.go UploadKnowledge()
    """
    try:
        from src.ingestion.loader import load_document
        from src.ingestion.chunker import auto_chunk, chunk_documents, ChunkingStrategy
        from src.ingestion.indexer import DocumentIndexer

        raw_docs = await load_document(req.filepath)

        # 如果指定了具体策略则用旧 API,否则自动选择最优策略
        if req.strategy and req.strategy != "auto":
            strategy = ChunkingStrategy(req.strategy)
            chunks = chunk_documents(raw_docs, strategy=strategy, chunk_size=req.chunk_size)
        else:
            chunks = auto_chunk(req.filepath)

        indexer = DocumentIndexer()
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


@app.post("/eval", response_model=EvalResponse)
async def eval_compare(req: EvalRequest):
    """
    三路对比评估接口
    ← 本项目设计: 同查询跑 3 条路径
    """
    try:
        from src.evaluation.compare import run_comparison

        result = await run_comparison(req.query)

        return EvalResponse(
            query=req.query,
            direct_answer=result.direct_answer,
            standard_rag=result.standard_rag,
            adaptive_rag=result.adaptive_rag,
            conclusion=result.conclusion,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources")
async def list_sources():
    """列出已索引的文档来源"""
    try:
        from src.ingestion.indexer import DocumentIndexer
        indexer = DocumentIndexer()
        sources = await indexer.get_sources()
        count = await indexer.count()
        return {"sources": sources, "total_chunks": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
