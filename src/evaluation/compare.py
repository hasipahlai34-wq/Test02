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
from src.utils.observability import langfuse_observation
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


async def _timed_evaluate_ragas(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str],
) -> tuple[dict[str, float] | None, str | None, float]:
    with langfuse_observation(
        name="eval.ragas",
        as_type="evaluator",
        input={
            "query": query,
            "answer_length": len(answer),
            "contexts": len(contexts),
            "has_ground_truth": bool(ground_truth),
        },
    ) as observation:
        t0 = time.time()
        scores, eval_error = await _safe_evaluate_ragas(query, answer, contexts, ground_truth)
        elapsed_ms = (time.time() - t0) * 1000
        if observation is not None:
            observation.update(output={
                "scores": scores,
                "eval_error": eval_error,
                "time_ms": elapsed_ms,
            })
        return scores, eval_error, elapsed_ms


async def _run_direct_answer(query: str, llm_client: LLMClient) -> dict:
    with langfuse_observation(
        name="eval.direct_answer",
        as_type="chain",
        input={"query": query},
    ) as observation:
        logger.info("[compare] path1 direct answer")
        t0 = time.time()
        answer = await llm_client.ask(
            prompt=query,
            system_prompt="Answer the user's question directly and concisely.",
            model_name=get_settings().llm_simple_model,
        )
        result = {
            "answer": answer,
            "time_ms": (time.time() - t0) * 1000,
            "model": get_settings().llm_simple_model,
            "tokens_est": len(answer) * 2,
        }
        if observation is not None:
            observation.update(output={
                "answer_length": len(answer),
                "time_ms": result["time_ms"],
                "model": result["model"],
            })
        return result


async def _run_standard_rag(
    query: str,
    ground_truth: Optional[str],
    llm_client: LLMClient,
    retrieval_filter: Optional[dict],
) -> dict:
    with langfuse_observation(
        name="eval.standard_rag",
        as_type="chain",
        input={"query": query, "scoped": bool(retrieval_filter)},
    ) as observation:
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
        answer_time_ms = (time.time() - t0) * 1000
        scores, eval_error, ragas_eval_time_ms = await _timed_evaluate_ragas(
            query, answer, contexts, ground_truth,
        )
        total_time_ms = (time.time() - t0) * 1000

        result = {
            "answer": answer,
            "time_ms": answer_time_ms,
            "answer_time_ms": answer_time_ms,
            "ragas_eval_time_ms": ragas_eval_time_ms,
            "total_time_ms": total_time_ms,
            "model": get_settings().llm_simple_model,
            "docs_count": len(docs),
            "tokens_est": len(answer) * 2 + sum(len(c) for c in contexts),
            "scores": scores,
            "eval_error": eval_error,
            "retrieved_sources": _sources(docs),
            "retrieved_document_ids": _document_ids(docs),
            "contexts_preview": contexts[:3],
        }
        if observation is not None:
            observation.update(output={
                "answer_length": len(answer),
                "answer_time_ms": answer_time_ms,
                "ragas_eval_time_ms": ragas_eval_time_ms,
                "total_time_ms": total_time_ms,
                "docs_count": len(docs),
                "scores": scores,
                "eval_error": eval_error,
            })
        return result


async def _run_adaptive_rag(
    query: str,
    ground_truth: Optional[str],
    llm_client: LLMClient,
    retrieval_filter: Optional[dict],
) -> dict:
    with langfuse_observation(
        name="eval.adaptive_rag",
        as_type="chain",
        input={"query": query, "scoped": bool(retrieval_filter)},
    ) as observation:
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
        answer_time_ms = (time.time() - t0) * 1000
        scores, eval_error, ragas_eval_time_ms = await _timed_evaluate_ragas(
            query, answer, contexts, ground_truth,
        )
        total_time_ms = (time.time() - t0) * 1000

        result = {
            "answer": answer,
            "time_ms": answer_time_ms,
            "answer_time_ms": answer_time_ms,
            "ragas_eval_time_ms": ragas_eval_time_ms,
            "total_time_ms": total_time_ms,
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
        if observation is not None:
            observation.update(output={
                "answer_length": len(answer),
                "answer_time_ms": answer_time_ms,
                "ragas_eval_time_ms": ragas_eval_time_ms,
                "total_time_ms": total_time_ms,
                "docs_count": len(docs),
                "complexity": result["complexity"],
                "strategy": result["strategy"],
                "scores": scores,
                "eval_error": eval_error,
            })
        return result


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
    result.winner = _select_recommended_path(result)["winner"]
    result.conclusion = _generate_conclusion(result)

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


def _metric_label(metric: str) -> str:
    return {
        "faithfulness": "忠实度",
        "answer_relevancy": "答案相关性",
        "context_precision": "上下文精确率",
        "context_recall": "上下文召回率",
        "numeric_match": "数值匹配",
    }.get(metric, metric)


def _complexity_label(complexity: str) -> str:
    return {
        "simple": "简单",
        "medium": "中等",
        "complex": "复杂",
    }.get(complexity, complexity or "N/A")


def _strategy_label(strategy: str) -> str:
    return {
        "no_retrieval": "不检索，直接回答",
        "single_step": "单步检索",
        "multi_step": "多步检索",
        "adaptive": "自适应路由",
    }.get(strategy, strategy or "N/A")


def _path_label(path: str) -> str:
    return {
        "standard_rag": "标准 RAG",
        "adaptive_rag": "自适应 RAG",
    }.get(path, path or "N/A")


def _safe_float(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def _quality_score(scores: dict) -> float | None:
    """Calculate a compact RAG quality score from available RAGAS metrics."""
    weights = {
        "faithfulness": 0.4,
        "answer_relevancy": 0.3,
        "context_precision": 0.1,
        "context_recall": 0.1,
        "numeric_match": 0.3,
    }
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, weight in weights.items():
        value = _safe_float(scores.get(metric))
        if value is None:
            continue
        weighted_sum += value * weight
        total_weight += weight
    if total_weight == 0:
        return None
    return weighted_sum / total_weight


def _time_ms(payload: dict) -> float:
    return float(payload.get("answer_time_ms") or payload.get("time_ms") or 0)


def _fmt_ms(payload: dict, key: str, fallback_key: str | None = None) -> str:
    value = payload.get(key)
    if value is None and fallback_key is not None:
        value = payload.get(fallback_key)
    try:
        return f"{float(value):.0f}ms"
    except (TypeError, ValueError):
        return "-"


def _select_recommended_path(result: CompareResult) -> dict[str, object]:
    """Select the better RAG path by quality first, then latency."""
    standard = result.standard_rag
    adaptive = result.adaptive_rag
    standard_quality = _quality_score(standard.get("scores") or {})
    adaptive_quality = _quality_score(adaptive.get("scores") or {})
    standard_time = _time_ms(standard)
    adaptive_time = _time_ms(adaptive)

    quality_tie_threshold = 0.03
    min_latency_gain_ms = 500

    if standard_quality is None and adaptive_quality is None:
        winner = "standard_rag" if standard_time <= adaptive_time else "adaptive_rag"
        reason = "RAGAS 未返回可用质量分，按耗时更低的路径推荐。"
    elif standard_quality is None:
        winner = "adaptive_rag"
        reason = "标准 RAG 缺少可用质量分，自适应 RAG 的评估结果更完整。"
    elif adaptive_quality is None:
        winner = "standard_rag"
        reason = "自适应 RAG 缺少可用质量分，标准 RAG 的评估结果更完整。"
    else:
        quality_delta = adaptive_quality - standard_quality
        if abs(quality_delta) > quality_tie_threshold:
            winner = "adaptive_rag" if quality_delta > 0 else "standard_rag"
            reason = (
                f"{_path_label(winner)} 的综合质量分更高："
                f"标准 RAG {standard_quality:.2f}，自适应 RAG {adaptive_quality:.2f}。"
            )
        else:
            winner = "standard_rag" if standard_time <= adaptive_time else "adaptive_rag"
            faster_time = min(standard_time, adaptive_time)
            slower_time = max(standard_time, adaptive_time)
            latency_gap = slower_time - faster_time
            if latency_gap >= min_latency_gain_ms:
                reason = (
                    "两条 RAG 路径质量差异不明显，"
                    f"{_path_label(winner)} 回答延迟更低："
                    f"{faster_time:.0f}ms vs {slower_time:.0f}ms。"
                )
            else:
                reason = (
                    "两条 RAG 路径质量与耗时差异都不明显，"
                    f"优先选择实现更直接的 {_path_label(winner)}。"
                )

    return {
        "winner": winner,
        "reason": reason,
        "standard_quality": standard_quality,
        "adaptive_quality": adaptive_quality,
    }


def _generate_conclusion(result: CompareResult) -> str:
    direct = result.direct_answer
    rag = result.standard_rag
    adaptive = result.adaptive_rag
    recommendation = _select_recommended_path(result)

    rag_scores = rag.get("scores") or {}
    adaptive_scores = adaptive.get("scores") or {}

    improvements = []
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        rag_val = rag_scores.get(metric)
        adaptive_val = adaptive_scores.get(metric)
        if isinstance(rag_val, (int, float)) and isinstance(adaptive_val, (int, float)) and rag_val > 0:
            pct = (adaptive_val - rag_val) / rag_val * 100
            improvements.append(f"{_metric_label(metric)} {pct:+.0f}%")

    lines = [
        "## 对比总结",
        "",
        f"推荐路径：{_path_label(str(recommendation['winner']))}",
        f"推荐理由：{recommendation['reason']}",
    ]
    if recommendation["standard_quality"] is not None and recommendation["adaptive_quality"] is not None:
        lines.append(
            "综合质量分："
            f"标准 RAG {float(recommendation['standard_quality']):.2f}，"
            f"自适应 RAG {float(recommendation['adaptive_quality']):.2f}"
        )
    if improvements:
        lines.append(f"- 指标变化：{', '.join(improvements)}")
    else:
        lines.append(
            "- 指标差异暂不明显；建议增加更多样本后再做更稳定的对比。"
        )

    lines += [
        f"- 直接回答耗时：{_fmt_ms(direct, 'answer_time_ms', 'time_ms')}",
        f"- 标准 RAG 回答耗时（不含 RAGAS）：{_fmt_ms(rag, 'answer_time_ms', 'time_ms')}",
        f"- 标准 RAG 的 RAGAS 评估耗时：{_fmt_ms(rag, 'ragas_eval_time_ms')}",
        f"- 标准 RAG 总耗时（含 RAGAS）：{_fmt_ms(rag, 'total_time_ms')}",
        f"- 自适应 RAG 回答耗时（不含 RAGAS）：{_fmt_ms(adaptive, 'answer_time_ms', 'time_ms')}",
        f"- 自适应 RAG 的 RAGAS 评估耗时：{_fmt_ms(adaptive, 'ragas_eval_time_ms')}",
        f"- 自适应 RAG 总耗时（含 RAGAS）：{_fmt_ms(adaptive, 'total_time_ms')}",
        (
            "- 自适应策略："
            f"查询复杂度={_complexity_label(str(adaptive.get('complexity') or 'N/A'))}，"
            f"检索策略={_strategy_label(str(adaptive.get('strategy') or 'N/A'))}"
        ),
    ]
    return "\n".join(lines)
