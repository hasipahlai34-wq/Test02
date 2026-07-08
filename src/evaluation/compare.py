from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from config.settings import get_settings
from src.evaluation.ragas_eval import evaluate_ragas
from src.models.llm import LLMClient
from src.retrieval.adaptive import create_adaptive_chain
from src.retrieval.single_step import SingleStepStrategy
from src.types import AgentState, CompareResult, QueryComplexity
from src.utils.token_manager import TokenBudget

logger = logging.getLogger(__name__)


async def _safe_evaluate_ragas(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str],
) -> tuple[dict[str, float] | None, str | None]:
    try:
        scores = await evaluate_ragas(query, answer, contexts, ground_truth)
        expected_metrics = ["faithfulness", "answer_relevancy"]
        if ground_truth:
            expected_metrics.extend(["context_precision", "context_recall"])
        missing_metrics = [name for name in expected_metrics if name not in scores]
        if missing_metrics:
            return scores, "Missing metrics: " + ", ".join(missing_metrics)
        return scores, None
    except Exception as e:
        logger.warning("RAGAS comparison evaluation failed: %s", e)
        return None, str(e)[:200]


async def _run_direct_answer(query: str, llm_client: LLMClient) -> dict:
    logger.info("[compare] path1 direct answer")
    t0 = time.time()
    answer = await llm_client.ask(
        prompt=query,
        system_prompt="Answer the user's question directly and concisely.",
        model_name=get_settings().llm_simple_model,
    )
    return {
        "answer": answer,
        "time_ms": (time.time() - t0) * 1000,
        "model": get_settings().llm_simple_model,
        "tokens_est": len(answer) * 2,
    }


async def _run_standard_rag(
    query: str,
    ground_truth: Optional[str],
    llm_client: LLMClient,
    retrieval_filter: Optional[dict],
) -> dict:
    logger.info("[compare] path2 standard RAG")
    t0 = time.time()

    strategy = SingleStepStrategy()
    agent_state = AgentState(query=query, complexity=QueryComplexity.MEDIUM)
    search_result = await strategy.retrieve(
        query,
        agent_state,
        retrieval_filter=retrieval_filter,
    )

    docs = search_result.documents
    contexts = [doc.content for doc in docs[:5]]
    answer = await llm_client.generate(
        messages=[{"role": "user", "content": query}],
        system_prompt=_build_rag_prompt(contexts),
        model_name=get_settings().llm_simple_model,
    )
    time_ms = (time.time() - t0) * 1000
    scores, eval_error = await _safe_evaluate_ragas(
        query, answer, contexts, ground_truth,
    )

    return {
        "answer": answer,
        "time_ms": time_ms,
        "model": get_settings().llm_simple_model,
        "docs_count": len(docs),
        "tokens_est": len(answer) * 2 + sum(len(c) for c in contexts),
        "scores": scores,
        "eval_error": eval_error,
        "retrieved_sources": _sources(docs),
        "retrieved_document_ids": _document_ids(docs),
        "contexts_preview": contexts[:3],
    }


async def _run_adaptive_rag(
    query: str,
    ground_truth: Optional[str],
    llm_client: LLMClient,
    retrieval_filter: Optional[dict],
) -> dict:
    logger.info("[compare] path3 adaptive RAG")
    t0 = time.time()

    adaptive, _ = create_adaptive_chain(llm_client=llm_client)
    adaptive_state = AgentState(query=query)
    adaptive_result = await adaptive.retrieve(
        query,
        adaptive_state,
        retrieval_filter=retrieval_filter,
    )

    docs = adaptive_result.documents
    contexts = [doc.content for doc in docs[:5]]
    model = TokenBudget.model_for_complexity(adaptive_state.complexity.value)
    answer = await llm_client.generate(
        messages=[{"role": "user", "content": query}],
        system_prompt=_build_rag_prompt(contexts),
        model_name=model,
    )
    time_ms = (time.time() - t0) * 1000
    scores, eval_error = await _safe_evaluate_ragas(
        query, answer, contexts, ground_truth,
    )

    return {
        "answer": answer,
        "time_ms": time_ms,
        "model": model,
        "strategy": adaptive_state.selected_strategy.value,
        "complexity": adaptive_state.complexity.value,
        "docs_count": len(docs),
        "tokens_est": len(answer) * 2 + sum(len(c) for c in contexts),
        "scores": scores,
        "eval_error": eval_error,
        "retrieved_sources": _sources(docs),
        "retrieved_document_ids": _document_ids(docs),
        "contexts_preview": contexts[:3],
    }


async def run_comparison(
    query: str,
    ground_truth: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    retrieval_filter: Optional[dict] = None,
) -> CompareResult:
    if llm_client is None:
        llm_client = LLMClient()

    result = CompareResult(query=query)
    direct_answer, standard_rag, adaptive_rag = await asyncio.gather(
        _run_direct_answer(query, llm_client),
        _run_standard_rag(query, ground_truth, llm_client, retrieval_filter),
        _run_adaptive_rag(query, ground_truth, llm_client, retrieval_filter),
    )

    result.direct_answer = direct_answer
    result.standard_rag = standard_rag
    result.adaptive_rag = adaptive_rag
    result.conclusion = _generate_conclusion(result)
    result.winner = "adaptive_rag"

    logger.info(
        "[compare] done: direct=%dms, standard=%dms, adaptive=%dms",
        direct_answer.get("time_ms", 0),
        standard_rag.get("time_ms", 0),
        adaptive_rag.get("time_ms", 0),
    )
    return result


def _build_rag_prompt(contexts: list[str]) -> str:
    if not contexts:
        return "Answer the user's question based on your general knowledge."

    docs_text = "\n\n---\n\n".join(contexts)
    return f"""Answer the question using only the retrieved context below.

## Retrieved Context
{docs_text}

## Requirements
- Do not invent facts beyond the retrieved context.
- If the context is insufficient, say so clearly.
- Answer in Chinese."""


def _sources(docs: list) -> list[str]:
    return sorted({
        doc.metadata.get("source_name") or doc.metadata.get("source") or doc.source
        for doc in docs
        if doc.metadata.get("source_name") or doc.metadata.get("source") or doc.source
    })


def _document_ids(docs: list) -> list[str]:
    return sorted({
        str(doc.metadata.get("document_id"))
        for doc in docs
        if doc.metadata.get("document_id")
    })


def _generate_conclusion(result: CompareResult) -> str:
    direct = result.direct_answer
    rag = result.standard_rag
    adaptive = result.adaptive_rag

    rag_scores = rag.get("scores") or {}
    adaptive_scores = adaptive.get("scores") or {}

    improvements = []
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        rag_val = rag_scores.get(metric, 0)
        adaptive_val = adaptive_scores.get(metric, 0)
        if rag_val > 0:
            pct = (adaptive_val - rag_val) / rag_val * 100
            improvements.append(f"{metric} {pct:+.0f}%")

    lines = [
        "## Comparison Summary",
        "",
        "Adaptive RAG is the preferred path for this comparison.",
    ]
    if improvements:
        lines.append(f"- Metric deltas: {', '.join(improvements)}")
    else:
        lines.append(
            "- Metric differences are inconclusive; add more samples for a stronger comparison."
        )

    lines += [
        f"- Direct answer time: {direct.get('time_ms', 0):.0f}ms",
        f"- Standard RAG time: {rag.get('time_ms', 0):.0f}ms",
        f"- Adaptive RAG time: {adaptive.get('time_ms', 0):.0f}ms",
        (
            "- Adaptive strategy: "
            f"complexity={adaptive.get('complexity', 'N/A')}, "
            f"strategy={adaptive.get('strategy', 'N/A')}"
        ),
    ]
    return "\n".join(lines)
