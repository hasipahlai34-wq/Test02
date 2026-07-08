"""
# ============================================================
# ★ LangGraph 主编排工作流 (StateGraph)
# ← WeKnora: internal/agent/engine.go — ReAct 循环的核心引擎
#   - executeLoop(): 主 ReAct 循环 (think → act → observe)
#   - runReActIteration(): 单次迭代
#   - isComplete(): 停止条件检查
#   - analyzeResponse(): 响应分析
#
#   我们用 LangGraph StateGraph 替代自研 ReAct 循环:
#   - 节点 (Node): 每个处理步骤 (分类/检索/生成/审核)
#   - 边 (Edge): 节点之间的数据流
#   - 条件边 (Conditional Edge): Adaptive-RAG 的动态路由
#   - 状态 (State): GraphState 在节点之间自动传递
# ============================================================

本模块构建完整的 Adaptive-RAG 工作流:
  classify → route → [no_retrieval|single_step|multi_step] →
  generate → review → ragas_evaluate → guard → hitl_gate → END

面试可讲:
"我用 LangGraph 替代了 WeKnora 的自研 ReAct 循环。
LangGraph 的好处是:
1. StateGraph 显式定义了每个处理步骤和它们之间的依赖关系
2. 条件边实现了 Adaptive-RAG 的动态路由
3. 状态自动在节点间传递, 不需要手动管理
4. 天然的流式支持 (stream_mode='updates')
5. 内置的检查点 (checkpointer) 支持暂停/恢复/回溯"
"""

from __future__ import annotations

import logging
import asyncio
import threading
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import GraphState
from src.graph.router import classify_query, route_by_complexity

logger = logging.getLogger(__name__)

CacheRoute = Literal["cache_hit", "cache_miss"]


def _route_by_cache(state: GraphState) -> CacheRoute:
    if state.get("cache_hit"):
        return "cache_hit"
    return "cache_miss"


# ================================================================
# 构建 Adaptive-RAG StateGraph
# ================================================================


def build_adaptive_rag_graph(
    checkpointer: Optional[MemorySaver] = None,
) -> StateGraph:
    """
    构建完整的 Adaptive-RAG LangGraph 工作流
    ← WeKnora: engine.go Execute() + executeLoop() + 插件链

    图结构:
    ```
    START
      │
      ▼
    [classify_query] ← 查询复杂度分类
      │
      ├─ simple ──→ [no_retrieval] ──┐
      ├─ medium ──→ [single_step]  ──┤
      └─ complex ─→ [multi_step]   ──┘
                                        │
                                        ▼
                                  [generate]       ← LLM 生成答案
                                        │
                                        ▼
                                  [review]         ← 质量审核 + 熔断
                                        │
                                        ▼
                                  [ragas_evaluate] ← ★ RAGAS 在线评估
                                        │
                                        ▼
                                  [guard]          ← 内容护栏
                                        │
                                        ▼
                                  [hitl_gate]      ← ★ HITL 审核门禁
                                        │
                                        ▼
                                       END
    ```

    Returns:
        编译后的 StateGraph (可调用 .invoke() / .astream())
    """
    workflow = StateGraph(GraphState)

    # ---- 注册节点 ----
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("cache_lookup", _cache_lookup_node)

    # 检索分支节点 (延迟导入避免循环依赖)
    workflow.add_node("no_retrieval", _no_retrieval_node)
    workflow.add_node("single_step", _single_step_node)
    workflow.add_node("multi_step", _multi_step_node)

    # 生成 + 审核 + RAGAS 评估 + 安全 + HITL
    workflow.add_node("generate", _generate_node)
    workflow.add_node("review", _review_node)
    workflow.add_node("ragas_evaluate", _ragas_evaluate_node)
    workflow.add_node("guard", _guard_node)
    workflow.add_node("hitl_gate", _hitl_gate_node)
    workflow.add_node("cache_store", _cache_store_node)
    workflow.add_node("route_by_complexity", lambda state: {})

    # ---- 设置入口 ----
    workflow.set_entry_point("classify_query")

    # ---- 条件边: Adaptive-RAG 核心路由 ----
    workflow.add_edge("classify_query", "cache_lookup")
    workflow.add_conditional_edges(
        "cache_lookup",
        _route_by_cache,
        {
            "cache_hit": END,
            "cache_miss": "route_by_complexity",
        },
    )
    workflow.add_conditional_edges(
        "route_by_complexity",
        route_by_complexity,
        {
            "no_retrieval": "no_retrieval",
            "single_step": "single_step",
            "multi_step": "multi_step",
        },
    )

    # ---- 所有检索分支汇聚到 generate ----
    workflow.add_edge("no_retrieval", "generate")
    workflow.add_edge("single_step", "generate")
    workflow.add_edge("multi_step", "generate")

    # ---- 生成 → 审核 → RAGAS → 护栏 → HITL → 结束 ----
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", "ragas_evaluate")
    workflow.add_edge("ragas_evaluate", "guard")
    workflow.add_edge("guard", "hitl_gate")
    workflow.add_edge("hitl_gate", "cache_store")
    workflow.add_edge("cache_store", END)

    # ---- 编译 ----
    if checkpointer is None:
        checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("Adaptive-RAG Graph 编译完成: %d 节点", len(workflow.nodes))

    return app


# ================================================================
# LangGraph 节点实现
# (在 workflow.py 中作为模块级函数定义，也可独立为文件)
# ================================================================


async def _cache_lookup_node(state: GraphState) -> dict:
    query = state.get("query", "")
    if not query:
        return {"cache_hit": False, "from_cache": False}
    if state.get("retrieval_filter"):
        logger.debug("[semantic_cache] skip lookup for scoped retrieval")
        return {"cache_hit": False, "from_cache": False}

    try:
        from src.cache.semantic_cache import get_semantic_cache

        cache = get_semantic_cache()
        cached_answer = cache.lookup_exact(query)
        if cached_answer is None and cache.has_semantic_entries():
            cached_answer = await asyncio.wait_for(cache.lookup(query), timeout=2.0)
        if cached_answer:
            logger.info("[semantic_cache] hit: '%s...'", query[:60])
            return {
                "generated_answer": cached_answer,
                "final_response": cached_answer,
                "completed": True,
                "from_cache": True,
                "cache_hit": True,
                "search_count": 0,
                "search_result_summary": "semantic_cache_hit",
                "quality_passed": True,
                "quality_score": state.get("quality_score", 1.0) or 1.0,
                "ragas_scores": state.get("ragas_scores"),
                "ragas_review_failed": False,
                "safety_risk_level": state.get("safety_risk_level", "low"),
                "needs_human_review": False,
                "hitl_status": "none",
                "hitl_decision": "cache_hit",
            }
    except asyncio.TimeoutError:
        logger.debug("[semantic_cache] lookup timed out; continuing without cache")
        return {"cache_hit": False, "from_cache": False}
    except Exception as e:
        logger.warning("[semantic_cache] lookup failed; continuing without cache: %s", e)
        return {
            "cache_hit": False,
            "from_cache": False,
            "cache_lookup_error": str(e)[:200],
        }

    return {"cache_hit": False, "from_cache": False}


async def _cache_store_node(state: GraphState) -> dict:
    if state.get("from_cache"):
        return {}
    if state.get("retrieval_filter"):
        logger.debug("[semantic_cache] skip store for scoped retrieval")
        return {}

    query = state.get("query", "")
    answer = state.get("generated_answer", "")
    if not query or not answer:
        return {}

    if state.get("needs_human_review") or state.get("hitl_status") == "pending":
        logger.debug("[semantic_cache] skip store for pending review")
        return {}

    try:
        from src.cache.semantic_cache import get_semantic_cache

        cache = get_semantic_cache()
        cache.store_exact(query, answer)
        if cache.embedding_ready():
            asyncio.create_task(_store_semantic_cache_background(query, answer))
        logger.debug("[semantic_cache] stored: '%s...'", query[:60])
    except Exception as e:
        logger.warning("[semantic_cache] store failed; answer already generated: %s", e)

    return {}


async def _store_semantic_cache_background(query: str, answer: str) -> None:
    try:
        from src.cache.semantic_cache import get_semantic_cache

        await asyncio.wait_for(get_semantic_cache().store(query, answer), timeout=2.0)
    except Exception as e:
        logger.debug("[semantic_cache] semantic store skipped: %s", e)


async def _no_retrieval_node(state: GraphState) -> dict:
    """
    无检索节点 — 简单查询跳过检索
    ← Adaptive-RAG: simple 查询路由到此
    """
    query = state.get("query", "")
    logger.info("[no_retrieval] 跳过检索: '%s...'", query[:60])
    return {
        "retrieved_docs": [],
        "search_count": 0,
        "search_result_summary": "无检索 (查询复杂度: simple)",
    }


async def _run_retrieval_with_input_safety(query: str, retrieve_coro) -> dict:
    from src.safety.content_guard import check_input_safety

    safety_task = asyncio.create_task(check_input_safety(query))
    retrieval_task = asyncio.create_task(retrieve_coro)

    done, _ = await asyncio.wait(
        {safety_task, retrieval_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    if safety_task in done:
        input_result = safety_task.result()
        if not input_result.get("safe", True):
            retrieval_task.cancel()
            await asyncio.gather(retrieval_task, return_exceptions=True)
            return _input_safety_blocked_result(input_result)

    retrieval_result = await retrieval_task
    input_result = await safety_task
    if not input_result.get("safe", True):
        return _input_safety_blocked_result(input_result)

    return {
        **retrieval_result,
        "safety_input_check": input_result,
        "input_safety_blocked": False,
    }


def _input_safety_blocked_result(input_result: dict) -> dict:
    return {
        "retrieved_docs": [],
        "search_count": 0,
        "search_result_summary": "input_safety_blocked",
        "safety_input_check": input_result,
        "input_safety_blocked": True,
        "generated_answer": "输入内容不安全，已拦截。",
        "completed": True,
        "error": "input_content_unsafe",
    }


def _format_search_summary(docs: list, max_items: int = 5) -> str:
    """★ 检索结果格式化 (共享工具函数) — 消除 _single_step_node / _multi_step_node 重复"""
    if not docs:
        return "未检索到相关内容"
    return "\n".join(
        f"[{i+1}] (分数:{d.score:.2f}) {d.content[:150]}..."
        for i, d in enumerate(docs[:max_items])
    )


async def _csv_sources_from_active_scope(
    retrieval_filter: dict | None,
    indexer: object | None = None,
) -> list[str]:
    """Find CSV sources from indexed metadata when retrieval returned no CSV docs."""
    if not retrieval_filter:
        return []

    try:
        if indexer is None:
            from src.ingestion.indexer import DocumentIndexer

            indexer = DocumentIndexer()
        ensure_initialized = getattr(indexer, "_ensure_initialized", None)
        if ensure_initialized is not None:
            await ensure_initialized()

        session_id = retrieval_filter.get("session_id")
        document_ids = set(retrieval_filter.get("document_ids") or [])
        sources: list[str] = []
        seen: set[str] = set()
        for doc in indexer.get_all_documents():
            metadata = doc.get("metadata") or {}
            if session_id and metadata.get("session_id") != session_id:
                continue
            if document_ids and metadata.get("document_id") not in document_ids:
                continue
            source = str(metadata.get("source") or "")
            if not source.lower().endswith(".csv") or source in seen:
                continue
            seen.add(source)
            sources.append(source)
        return sources
    except Exception as e:
        logger.warning("[single_step] CSV scope source lookup failed: %s", e)
        return []


async def _single_step_node(state: GraphState) -> dict:
    query = state.get("rewritten_query") or state.get("query", "")
    return await _run_retrieval_with_input_safety(
        query,
        _single_step_retrieve_node(state),
    )


async def _single_step_retrieve_node(state: GraphState) -> dict:
    """
    单步检索节点 — BM25 + Dense + Rerank
    ← WeKnora: chat_pipeline/ 完整 RAG Pipeline

    ★ CSV 聚合增强: 检测聚合查询 → pandas 直接计算 CSV → 结果前置
    """
    from src.retrieval.single_step import (
        calculate_markdown_table_aggregation,
        get_single_step,
    )
    from src.retrieval.csv_aggregator import (
        is_aggregate_query,
        execute_csv_aggregation,
        find_csv_sources_from_docs,
    )
    from src.types import AgentState, Document as TypedDocument, MatchType

    query = state.get("rewritten_query") or state.get("query", "")

    try:
        strategy = await get_single_step()  # ★ C5: 单例复用 BM25 索引

        agent_state = AgentState.from_graph_state(dict(state))
        agent_state.query = query
        result = await strategy.retrieve(
            query,
            agent_state,
            retrieval_filter=state.get("retrieval_filter"),
            top_k=state.get("retrieval_top_k"),
        )

        docs = result.documents

        md_agg_text = calculate_markdown_table_aggregation(
            query,
            [doc.content for doc in docs],
        )
        if md_agg_text:
            docs.insert(0, TypedDocument(
                content=md_agg_text,
                score=1.0,
                match_type=MatchType.HYBRID,
                chunk_index=-1,
                metadata={
                    "aggregate_result": "true",
                    "query_type": "markdown_table_aggregation",
                },
            ))

        # ★ CSV 聚合查询增强: 检测聚合关键词 → pandas 计算 → 结果前置
        if is_aggregate_query(query):
            csv_paths = find_csv_sources_from_docs(docs)
            if not csv_paths:
                csv_paths = await _csv_sources_from_active_scope(
                    state.get("retrieval_filter"),
                    getattr(strategy, "_indexer", None),
                )
            for csv_path in csv_paths:
                try:
                    agg_text = execute_csv_aggregation(csv_path, query)
                    if agg_text:
                        agg_doc = TypedDocument(
                            content=f"[CSV 聚合计算结果]\n{agg_text}",
                            score=1.0,
                            match_type=MatchType.HYBRID,
                            chunk_index=-1,
                            metadata={
                                "source": csv_path,
                                "aggregate_result": "true",
                                "query_type": "csv_aggregation",
                            },
                        )
                        docs.insert(0, agg_doc)  # ★ 聚合结果前置
                        logger.info(
                            "[single_step] CSV 聚合命中: %s → %d chars",
                            csv_path, len(agg_text),
                        )
                        break  # 只处理第一个匹配的 CSV 文件
                except Exception as e:
                    logger.warning("[single_step] CSV 聚合降级: %s", e)

        summary = _format_search_summary(docs)

        logger.info("[single_step] 检索完成: %d 个文档", len(docs))
    except (ConnectionError, TimeoutError) as e:
        # 网络/向量数据库不可用 → 降级为空检索（预期内的故障）
        logger.warning("[single_step] 服务不可用，降级为空检索: %s", e)
        docs = []
        summary = "检索服务暂不可用，请稍后重试"
    except Exception as e:
        # 未预期的代码逻辑错误 → 记录 critical 并降级
        logger.critical("[single_step] 检索异常: %s", e, exc_info=True)
        docs = []
        summary = f"检索内部错误: {type(e).__name__}"

    return {
        "retrieved_docs": docs,
        "search_count": len(docs),
        "search_result_summary": summary,
    }


async def _multi_step_node(state: GraphState) -> dict:
    query = state.get("rewritten_query") or state.get("query", "")
    return await _run_retrieval_with_input_safety(
        query,
        _multi_step_retrieve_node(state),
    )


async def _multi_step_retrieve_node(state: GraphState) -> dict:
    """
    多步迭代检索节点 — 迭代检索 + HyDE + 改写
    ← Adaptive-RAG: complex 查询路由到此
    """
    from src.retrieval.multi_step import MultiStepStrategy
    from src.retrieval.single_step import calculate_markdown_table_aggregation
    from src.types import AgentState, Document as TypedDocument, MatchType

    query = state.get("rewritten_query") or state.get("query", "")
    hyde_update: dict = {}

    try:
        strategy = MultiStepStrategy()

        agent_state = AgentState.from_graph_state(dict(state))
        agent_state.query = query
        result = await strategy.retrieve(
            query,
            agent_state,
            retrieval_filter=state.get("retrieval_filter"),
        )

        docs = result.documents
        md_agg_text = calculate_markdown_table_aggregation(
            query,
            [doc.content for doc in docs],
        )
        if md_agg_text:
            docs.insert(0, TypedDocument(
                content=md_agg_text,
                score=1.0,
                match_type=MatchType.HYBRID,
                chunk_index=-1,
                metadata={
                    "aggregate_result": "true",
                    "query_type": "markdown_table_aggregation",
                },
            ))
        summary = _format_search_summary(docs)

        hyde_hypothesis = getattr(agent_state, "hyde_hypothesis", "")
        if hyde_hypothesis:
            hyde_update["hyde_hypothesis"] = hyde_hypothesis

        logger.info("[multi_step] 多步检索完成: %d 个文档", len(docs))
    except (ConnectionError, TimeoutError) as e:
        logger.warning("[multi_step] 服务不可用，降级为空检索: %s", e)
        docs = []
        summary = "检索服务暂不可用，请稍后重试"
    except Exception as e:
        logger.critical("[multi_step] 检索异常: %s", e, exc_info=True)
        docs = []
        summary = f"检索内部错误: {type(e).__name__}"

    return {
        "retrieved_docs": docs,
        "search_count": len(docs),
        "search_result_summary": summary,
        **hyde_update,
    }


async def _generate_node(state: GraphState) -> dict:
    """
    生成节点 — 组装 Prompt + LLM 生成答案
    ← WeKnora: chat_pipeline/chat_completion.go + chat_completion_stream.go
    """
    from src.agents.generator import generate_answer

    if state.get("input_safety_blocked"):
        return {
            "generated_answer": state.get("generated_answer", "输入内容不安全，已拦截。"),
            "completed": True,
            "error": state.get("error", "input_content_unsafe"),
        }

    result = await generate_answer(state)
    return result


async def _review_node(state: GraphState) -> dict:
    """
    审核节点 — 答案质量评估 + 熔断器更新
    ← WeKnora: engine.go analyzeResponse() + 原项目 B 质量熔断
    """
    from src.agents.reviewer import review_answer

    if state.get("complexity") == "simple":
        return {
            "quality_score": 0.7,
            "quality_passed": True,
            "review_reason": "skipped_for_simple_query",
        }

    result = await review_answer(state)
    return result


async def _guard_node(state: GraphState) -> dict:
    """
    安全护栏节点 — 内容审核 + HITL 判断
    ← 原项目 B content_guard.py
    """
    from src.safety.content_guard import check_output_safety

    result = await check_output_safety(state)
    return result


async def _ragas_evaluate_node(state: GraphState) -> dict:
    """
    ★ RAGAS 在线评估节点 — 每次查询后自动评分
    异步非阻塞, 失败不影响主流程
    """
    threading.Thread(
        target=lambda: asyncio.run(_ragas_evaluate_and_log(dict(state))),
        daemon=True,
    ).start()
    return {
        "ragas_scores": None,
        "ragas_eval_error": "pending_async",
        "ragas_review_failed": not state.get("quality_passed", True),
    }


async def _ragas_evaluate_and_log(state: GraphState) -> None:
    try:
        import json
        from datetime import datetime, timezone
        from pathlib import Path

        from src.evaluation.online_evaluator import ragas_evaluate_node

        result = await ragas_evaluate_node(state)
        log_dir = Path("data") / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_item = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": state.get("query", ""),
            "session_id": state.get("session_id", ""),
            "result": result,
        }
        with open(log_dir / "ragas_async.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_item, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning("[ragas_async] evaluation/logging failed: %s", e)


async def _hitl_gate_node(state: GraphState) -> dict:
    """
    ★ HITL 审核门禁节点 — 综合判断是否需要人工介入
    支持 interrupt (Streamlit) 和 file_queue (CLI/API) 双模式
    """
    from src.graph.hitl import hitl_gate_node

    result = await hitl_gate_node(state)
    return result


# ================================================================
# 便捷执行函数
# ================================================================


async def run_adaptive_rag(
    query: str,
    session_id: str = "default",
    config: Optional[dict] = None,
) -> GraphState:
    """
    一键执行 Adaptive-RAG 工作流 (同步等待完整结果)

    Args:
        query: 用户查询
        session_id: 会话 ID
        config: LangGraph 配置 (thread_id 等)

    Returns:
        最终的 GraphState (包含 generated_answer)
    """
    app = build_adaptive_rag_graph()

    if config is None:
        config = {"configurable": {"thread_id": session_id}}

    initial_state: GraphState = {
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
        "hitl_status": "none",
        "hitl_decision": "",
        "safety_input_check": None,
        "safety_output_check": None,
        "final_response": "",
    }

    final_state = await app.ainvoke(initial_state, config)
    return final_state


async def run_adaptive_rag_stream(
    query: str,
    session_id: str = "default",
    config: Optional[dict] = None,
    retrieval_filter: Optional[dict] = None,
):
    """
    流式执行 Adaptive-RAG 工作流
    ← WeKnora: SSE Stream → LangGraph stream_mode="updates"

    Yields:
        每个节点的输出更新 (dict)
    """
    app = build_adaptive_rag_graph()

    if config is None:
        config = {"configurable": {"thread_id": session_id}}

    initial_state: GraphState = {
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
    }

    logger.info("开始流式执行 Adaptive-RAG: '%s...'", query[:60])

    async for event in app.astream(initial_state, config, stream_mode="updates"):
        yield event

    logger.info("Adaptive-RAG 流式执行完成")
