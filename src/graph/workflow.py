"""Docstring."""

from __future__ import annotations

import logging
import asyncio
import inspect
import threading
from typing import Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import GraphState
from src.graph.router import classify_query, route_by_complexity
from src.utils.observability import langfuse_observation, langfuse_trace_context, with_langfuse_config

logger = logging.getLogger(__name__)

CacheRoute = Literal["cache_hit", "cache_miss"]
ReviewRoute = Literal["ragas_evaluate", "guard"]


def _route_by_cache(state: GraphState) -> CacheRoute:
    if state.get("cache_hit"):
        return "cache_hit"
    return "cache_miss"


def _route_after_review(state: GraphState) -> ReviewRoute:
    from config.settings import get_settings

    if get_settings().ragas_online_enabled:
        return "ragas_evaluate"
    return "guard"


def _trace_state_summary(state: GraphState | dict) -> dict:
    """Small state summary for Langfuse node spans; avoid tracing full documents."""
    return {
        "query": str(state.get("rewritten_query") or state.get("query") or "")[:300],
        "session_id": state.get("session_id"),
        "complexity": state.get("complexity"),
        "strategy": state.get("selected_strategy"),
        "search_count": state.get("search_count"),
        "cache_hit": state.get("cache_hit") or state.get("from_cache"),
        "quality_passed": state.get("quality_passed"),
        "quality_score": state.get("quality_score"),
        "hitl_status": state.get("hitl_status"),
        "has_retrieval_filter": bool(state.get("retrieval_filter")),
    }


def _trace_output_summary(output: object) -> object:
    if not isinstance(output, dict):
        return output
    return {
        "complexity": output.get("complexity"),
        "strategy": output.get("selected_strategy"),
        "search_count": output.get("search_count"),
        "cache_hit": output.get("cache_hit") or output.get("from_cache"),
        "quality_passed": output.get("quality_passed"),
        "quality_score": output.get("quality_score"),
        "hitl_status": output.get("hitl_status"),
        "completed": output.get("completed"),
        "answer_length": len(str(output.get("generated_answer") or output.get("final_response") or "")),
        "retrieved_docs": len(output.get("retrieved_docs") or []),
        "error": output.get("error"),
    }


def _observed_node(name: str, func, *, as_type: str = "span"):
    """Wrap a LangGraph node in a Langfuse observation without changing behavior."""

    async def wrapped(state: GraphState) -> dict:
        with langfuse_observation(
            name=f"workflow.{name}",
            as_type=as_type,
            input=_trace_state_summary(state),
            metadata={"node": name},
        ) as observation:
            result = func(state)
            if inspect.isawaitable(result):
                result = await result
            if observation is not None:
                observation.update(output=_trace_output_summary(result))
            return result

    return wrapped


# ================================================================
# 閺嬪嫬缂?Adaptive-RAG StateGraph
# ================================================================


def build_adaptive_rag_graph(
    checkpointer: Optional[MemorySaver] = None,
) -> StateGraph:
    """Docstring."""
    workflow = StateGraph(GraphState)

    # ---- 濞夈劌鍞介懞鍌滃仯 ----
    workflow.add_node("classify_query", _observed_node("classify_query", classify_query, as_type="chain"))
    workflow.add_node("cache_lookup", _observed_node("cache_lookup", _cache_lookup_node, as_type="span"))

    # 濡偓缁便垹鍨庨弨顖濆Ν閻?(瀵ゆ儼绻滅€电厧鍙嗛柆鍨帳瀵邦亞骞嗘笟婵婄)
    workflow.add_node("no_retrieval", _observed_node("no_retrieval", _no_retrieval_node, as_type="span"))
    workflow.add_node("single_step", _observed_node("single_step", _single_step_node, as_type="retriever"))
    workflow.add_node("multi_step", _observed_node("multi_step", _multi_step_node, as_type="retriever"))

    # 閻㈢喐鍨?+ 鐎光剝鐗?+ RAGAS 鐠囧嫪鍙?+ 鐎瑰鍙?+ HITL
    workflow.add_node("generate", _observed_node("generate", _generate_node, as_type="chain"))
    workflow.add_node("review", _observed_node("review", _review_node, as_type="evaluator"))
    workflow.add_node("ragas_evaluate", _observed_node("ragas_evaluate", _ragas_evaluate_node, as_type="evaluator"))
    workflow.add_node("guard", _observed_node("guard", _guard_node, as_type="guardrail"))
    workflow.add_node("hitl_gate", _observed_node("hitl_gate", _hitl_gate_node, as_type="span"))
    workflow.add_node("cache_store", _observed_node("cache_store", _cache_store_node, as_type="span"))
    workflow.add_node("route_by_complexity", _observed_node("route_by_complexity", lambda state: {}, as_type="span"))

    # ---- 鐠佸墽鐤嗛崗銉ュ經 ----
    workflow.set_entry_point("classify_query")

    # ---- 閺夆€叉鏉? Adaptive-RAG 閺嶇绺剧捄顖滄暠 ----
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

    # ---- 閹碘偓閺堝顥呯槐銏犲瀻閺€顖涚湽閼辨艾鍩?generate ----
    workflow.add_edge("no_retrieval", "generate")
    workflow.add_edge("single_step", "generate")
    workflow.add_edge("multi_step", "generate")

    # ---- 閻㈢喐鍨?閳?鐎光剝鐗?閳?RAGAS 閳?閹躲倖鐖?閳?HITL 閳?缂佹挻娼?----
    workflow.add_edge("generate", "review")
    workflow.add_conditional_edges(
        "review",
        _route_after_review,
        {
            "ragas_evaluate": "ragas_evaluate",
            "guard": "guard",
        },
    )
    workflow.add_edge("ragas_evaluate", "guard")
    workflow.add_edge("guard", "hitl_gate")
    workflow.add_edge("hitl_gate", "cache_store")
    workflow.add_edge("cache_store", END)

    # ---- 缂傛牞鐦?----
    if checkpointer is None:
        checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("Adaptive-RAG graph compiled: %d nodes", len(workflow.nodes))

    return app


# ================================================================
# LangGraph 閼哄倻鍋ｇ€圭偟骞?
# (閸?workflow.py 娑擃厺缍旀稉鐑樐侀崸妤冮獓閸戣姤鏆熺€规矮绠熼敍灞肩瘍閸欘垳瀚粩瀣╄礋閺傚洣娆?
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
    """Docstring."""
    query = state.get("query", "")
    logger.info("[no_retrieval] 鐠哄疇绻冨Λ鈧槐? '%s...'", query[:60])
    return {
        "selected_strategy": "no_retrieval",
        "retrieved_docs": [],
        "search_count": 0,
        "search_result_summary": "閺冪姵顥呯槐?(閺屻儴顕楁径宥嗘絽鎼? simple)",
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
        "generated_answer": "Input content is unsafe; request blocked.",
        "completed": True,
        "error": "input_content_unsafe",
    }


def _format_search_summary(docs: list, max_items: int = 5) -> str:
    """Docstring."""
    if not docs:
        return "閺堫亝顥呯槐銏犲煂閻╃鍙ч崘鍛啇"
    return "\n".join(
        f"[{i+1}] (閸掑棙鏆?{d.score:.2f}) {d.content[:150]}..."
        for i, d in enumerate(docs[:max_items])
    )


async def _csv_sources_from_active_scope(
    retrieval_filter: dict | None,
    indexer: object | None = None,
) -> list[str]:
    """Docstring."""
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
    result = await _run_retrieval_with_input_safety(
        query,
        _single_step_retrieve_node(state),
    )
    result["selected_strategy"] = "single_step"
    return result


async def _single_step_retrieve_node(state: GraphState) -> dict:
    """Docstring."""
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
        strategy = await get_single_step()  # 閳?C5: 閸楁洑绶ユ径宥囨暏 BM25 缁便垹绱?

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

        # 閳?CSV 閼辨艾鎮庨弻銉嚄婢х偛宸? 濡偓濞村浠涢崥鍫濆彠闁款喛鐦?閳?pandas 鐠侊紕鐣?閳?缂佹挻鐏夐崜宥囩枂
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
                            content=f"[CSV 閼辨艾鎮庣拋锛勭暬缂佹挻鐏塢\n{agg_text}",
                            score=1.0,
                            match_type=MatchType.HYBRID,
                            chunk_index=-1,
                            metadata={
                                "source": csv_path,
                                "aggregate_result": "true",
                                "query_type": "csv_aggregation",
                            },
                        )
                        docs.insert(0, agg_doc)  # 閳?閼辨艾鎮庣紒鎾寸亯閸撳秶鐤?
                        logger.info(
                            "[single_step] CSV 閼辨艾鎮庨崨鎴掕厬: %s 閳?%d chars",
                            csv_path, len(agg_text),
                        )
                        break  # 閸欘亜顦╅悶鍡欘儑娑撯偓娑擃亜灏柊宥囨畱 CSV 閺傚洣娆?
                except Exception as e:
                    logger.warning("[single_step] CSV 閼辨艾鎮庨梽宥囬獓: %s", e)

        summary = _format_search_summary(docs)

        logger.info("[single_step] retrieval completed: %d docs", len(docs))
    except (ConnectionError, TimeoutError) as e:
        # 缂冩垹绮?閸氭垿鍣洪弫鐗堝祦鎼存挷绗夐崣顖滄暏 閳?闂勫秶楠囨稉铏光敄濡偓缁鳖澁绱欐０鍕埂閸愬懐娈戦弫鍛存閿?        logger.warning("[single_step] 閺堝秴濮熸稉宥呭讲閻㈩煉绱濋梽宥囬獓娑撹櫣鈹栧Λ鈧槐? %s", e)
        docs = []
        summary = "Retrieval service is unavailable; please retry later."
    except Exception as e:
        # 閺堫亪顣╅張鐔烘畱娴狅絿鐖滈柅鏄忕帆闁挎瑨顕?閳?鐠佹澘缍?critical 楠炲爼妾风痪?        logger.critical("[single_step] 濡偓缁便垹绱撶敮? %s", e, exc_info=True)
        docs = []
        summary = f"濡偓缁便垹鍞撮柈銊╂晩鐠? {type(e).__name__}"

    return {
        "retrieved_docs": docs,
        "search_count": len(docs),
        "search_result_summary": summary,
    }


async def _multi_step_node(state: GraphState) -> dict:
    query = state.get("rewritten_query") or state.get("query", "")
    result = await _run_retrieval_with_input_safety(
        query,
        _multi_step_retrieve_node(state),
    )
    result["selected_strategy"] = "multi_step"
    return result


async def _multi_step_retrieve_node(state: GraphState) -> dict:
    """Docstring."""
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

        logger.info("[multi_step] retrieval completed: %d docs", len(docs))
    except (ConnectionError, TimeoutError) as e:
        logger.warning("[multi_step] 閺堝秴濮熸稉宥呭讲閻㈩煉绱濋梽宥囬獓娑撹櫣鈹栧Λ鈧槐? %s", e)
        docs = []
        summary = "Retrieval service is unavailable; please retry later."
    except Exception as e:
        logger.critical("[multi_step] 濡偓缁便垹绱撶敮? %s", e, exc_info=True)
        docs = []
        summary = f"濡偓缁便垹鍞撮柈銊╂晩鐠? {type(e).__name__}"

    return {
        "retrieved_docs": docs,
        "search_count": len(docs),
        "search_result_summary": summary,
        **hyde_update,
    }


async def _generate_node(state: GraphState) -> dict:
    """Generate an answer; optionally emit token deltas via LangGraph custom stream."""
    from src.agents.generator import generate_answer

    if state.get("input_safety_blocked"):
        return {
            "generated_answer": state.get("generated_answer", "Input content is unsafe; request blocked."),
            "completed": True,
            "error": state.get("error", "input_content_unsafe"),
        }

    if not state.get("stream_tokens"):
        return await generate_answer(state)

    result = await generate_answer(state, stream=True)
    answer_stream = result.pop("answer_stream", None)
    if answer_stream is None:
        return result

    answer_parts: list[str] = []
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        writer = None

    async for delta in answer_stream:
        if not delta:
            continue
        text = str(delta)
        answer_parts.append(text)
        if writer is not None:
            writer({"event": "answer_delta", "text": text})

    answer = "".join(answer_parts)
    return {
        **result,
        "generated_answer": answer,
        "completed": True,
    }


async def _review_node(state: GraphState) -> dict:
    """Docstring."""
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
    """Docstring."""
    from src.safety.content_guard import check_output_safety

    result = await check_output_safety(state)
    return result


async def _ragas_evaluate_node(state: GraphState) -> dict:
    """Docstring."""
    from config.settings import get_settings

    if not get_settings().ragas_online_enabled:
        return {
            "ragas_scores": None,
            "ragas_eval_error": "disabled",
            "ragas_review_failed": False,
        }

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
    """Docstring."""
    from src.graph.hitl import hitl_gate_node

    result = await hitl_gate_node(state)
    return result


# ================================================================
# 娓氭寧宓庨幍褑顢戦崙鑺ユ殶
# ================================================================


async def run_adaptive_rag(
    query: str,
    session_id: str = "default",
    config: Optional[dict] = None,
) -> GraphState:
    """Docstring."""
    app = build_adaptive_rag_graph()

    if config is None:
        config = {"configurable": {"thread_id": session_id}}
    config = with_langfuse_config(
        config,
        trace_name="adaptive-rag.ask",
        session_id=session_id,
        metadata={"entrypoint": "workflow", "query": query},
        tags=["workflow"],
    )

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

    with langfuse_trace_context(
        trace_name="adaptive-rag.ask",
        session_id=session_id,
        metadata={"entrypoint": "workflow", "query": query},
        tags=["workflow"],
    ):
        final_state = await app.ainvoke(initial_state, config)
    return final_state


async def run_adaptive_rag_stream(
    query: str,
    session_id: str = "default",
    config: Optional[dict] = None,
    retrieval_filter: Optional[dict] = None,
):
    """Docstring."""
    app = build_adaptive_rag_graph()

    if config is None:
        config = {"configurable": {"thread_id": session_id}}
    config = with_langfuse_config(
        config,
        trace_name="adaptive-rag.stream",
        session_id=session_id,
        metadata={"entrypoint": "workflow.stream", "query": query},
        tags=["workflow", "stream"],
    )

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

    logger.info("瀵偓婵绁﹀蹇斿⒔鐞?Adaptive-RAG: '%s...'", query[:60])

    with langfuse_trace_context(
        trace_name="adaptive-rag.stream",
        session_id=session_id,
        metadata={"entrypoint": "workflow.stream", "query": query},
        tags=["workflow", "stream"],
    ):
        async for event in app.astream(initial_state, config, stream_mode="updates"):
            yield event

    logger.info("Adaptive-RAG streaming execution completed")
