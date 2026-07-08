"""
# ============================================================
# RAGAS 评估集成
# ← 本项目设计: RAGAS (Retrieval Augmented Generation Assessment)
#   WeKnora 有端到端测试 (BLEU/ROUGE)，但无 RAGAS 集成
#
# RAGAS 是专为 RAG 系统设计的评估框架，指标包括:
# - Faithfulness (忠实度): 答案是否忠于检索内容
# - Answer Relevancy (答案相关性): 答案是否切题
# - Context Precision (上下文精确度): 检索内容中相关比例
# - Context Recall (上下文召回率): 相关文档被检索到的比例
# ============================================================
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from text for lightweight answer validation."""
    values = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?", str(text).replace(",", "")):
        value = float(match.group(0))
        if value.is_integer():
            value = int(value)
        values.append(value)
    return values


def calculate_from_contexts(contexts: list[str], query: str) -> list[float]:
    """Calculate expected numeric values from retrieved contexts when supported."""
    try:
        from src.retrieval.single_step import calculate_markdown_table_aggregation

        result = calculate_markdown_table_aggregation(query, contexts)
    except Exception as e:
        logger.debug("numeric context calculation skipped: %s", e)
        return []
    if not result:
        return []
    return extract_numbers(result.split("总预算:", 1)[-1])


def validate_numeric_answer(answer: str, query: str, contexts: list[str]) -> dict:
    """Validate numeric aggregate answers against deterministic context calculation."""
    if not re.search(r"(最多|最少|总计|合计|总预算|总支出|剩余|结余)", query):
        return {
            "numeric_match": None,
            "answer_numbers": [],
            "correct_numbers": [],
        }

    answer_numbers = extract_numbers(answer)
    correct_numbers = calculate_from_contexts(contexts, query)
    if not correct_numbers:
        return {
            "numeric_match": None,
            "answer_numbers": answer_numbers,
            "correct_numbers": [],
        }

    answer_set = {float(value) for value in answer_numbers}
    correct_set = {float(value) for value in correct_numbers}
    return {
        "numeric_match": correct_set.issubset(answer_set),
        "answer_numbers": answer_numbers,
        "correct_numbers": correct_numbers,
    }


def _coerce_score(value: Any) -> float | None:
    """Return a real numeric score, or None when RAGAS did not produce one."""
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]

    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(score) or math.isinf(score):
        return None
    return score


def _extract_ragas_scores(result: Any, metric_names: list[str]) -> dict[str, float]:
    """
    Extract first-row metric values from RAGAS EvaluationResult-like objects.

    RAGAS versions expose EvaluationResult differently. Prefer to_pandas(),
    then fall back to mapping/index access. Missing or invalid metrics are
    omitted instead of being converted to 0.0.
    """
    scores: dict[str, float] = {}

    if hasattr(result, "to_pandas"):
        try:
            frame = result.to_pandas()
            if hasattr(frame, "empty") and not frame.empty:
                for name in metric_names:
                    if name not in frame.columns:
                        continue
                    score = _coerce_score(frame.iloc[0][name])
                    if score is None:
                        logger.warning("RAGAS metric %s returned an invalid value", name)
                    else:
                        scores[name] = score
                if scores:
                    return scores
        except Exception as e:
            logger.warning("RAGAS to_pandas() score extraction failed: %s", e)

    candidates: list[Any] = [result]
    if isinstance(result, Mapping):
        for key in ("scores", "score"):
            nested = result.get(key)
            if isinstance(nested, Mapping):
                candidates.append(nested)

    for source in candidates:
        for name in metric_names:
            if name in scores:
                continue

            found = False
            value = None
            if isinstance(source, Mapping) and name in source:
                value = source[name]
                found = True
            else:
                try:
                    value = source[name]
                    found = True
                except (KeyError, TypeError, AttributeError):
                    pass

            if not found:
                continue

            score = _coerce_score(value)
            if score is None:
                logger.warning("RAGAS metric %s returned an invalid value", name)
            else:
                scores[name] = score

    return scores


def _fallback_ragas_scores(ground_truth: Optional[str] = None) -> dict[str, float]:
    """Return complete conservative scores when external RAGAS deps are unavailable."""
    scores = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
    }
    if ground_truth:
        scores["context_precision"] = 0.0
        scores["context_recall"] = 0.0
    return scores


async def evaluate_ragas(
    query: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> dict[str, float]:
    """
    使用 RAGAS 评估 RAG 系统的回答质量

    Args:
        query: 用户查询
        answer: 生成的答案
        contexts: 检索到的文档内容列表
        ground_truth: 标准答案 (可选，用于 Context Recall)

    Returns:
        评估分数字典:
        {
            "faithfulness": 0.0-1.0,   # 忠实度
            "answer_relevancy": 0.0-1.0,  # 答案相关性
            "context_precision": 0.0-1.0,  # 上下文精确度
            "context_recall": 0.0-1.0,     # 上下文召回率 (需要 ground_truth)
        }
    """
    if not contexts:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
        }

    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            context_precision,
            context_recall,
        )
        # answer_relevancy 默认 strictness=3（生成 3 个问题，LLM 调 n=3），
        # DeepSeek 等 API 只支持 n=1，需显式设 strictness=1 避免 BadRequestError
        from ragas.metrics._answer_relevance import AnswerRelevancy
        answer_relevancy = None
        from ragas.llms import LangchainLLMWrapper
        from datasets import Dataset
        from langchain_openai import ChatOpenAI

        from config.settings import get_settings

        settings = get_settings()
        base_url = (settings.llm_base_url or "").lower()
        is_official_openai = not base_url or "api.openai.com" in base_url
        relevancy_strictness = (
            3
            if "gpt" in settings.llm_default_model.lower() and is_official_openai
            else 1
        )
        answer_relevancy = AnswerRelevancy(strictness=relevancy_strictness)

        # RAGAS 评估专用 LLM — 用 DeepSeek v4-pro（格式输出稳定），与日常问答的中转站模型分离
        eval_model = os.getenv("RAGAS_EVAL_MODEL", "deepseek-v4-pro")
        eval_base_url = os.getenv("RAGAS_EVAL_BASE_URL", "https://api.deepseek.com/v1")

        eval_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_model,
                api_key=settings.llm_api_key,
                base_url=eval_base_url,
                temperature=0,
                timeout=120,
                max_retries=2,
            )
        )

        # RAGAS 评估用的 Embeddings
        # 使用 LangChain 的 HuggingFaceEmbeddings（有 embed_query 接口），
        # evaluate() 会自动用 LangchainEmbeddingsWrapper 包装为 ragas 兼容的 BaseRagasEmbeddings。
        # 显式传入本地模型，避免 ragas 默认创建 OpenAI embeddings 去读 OPENAI_API_KEY 环境变量。
        from langchain_community.embeddings import HuggingFaceEmbeddings
        eval_embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"local_files_only": True},
        )

        # 构建 RAGAS 数据集
        eval_data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        }
        if ground_truth:
            eval_data["reference"] = [ground_truth]

        dataset = Dataset.from_dict(eval_data)

        # 选择评估指标
        # context_precision 在 ragas v0.4.3+ 需要 reference 列，只在有 ground_truth 时使用
        metrics = [
            faithfulness,
            answer_relevancy,
        ]
        if ground_truth:
            metrics.append(context_precision)
            metrics.append(context_recall)

        # 执行评估（显式传入 llm 和 embeddings，不依赖环境变量）
        result = evaluate(dataset, metrics=metrics, llm=eval_llm, embeddings=eval_embeddings)

        metric_names = [metric.name for metric in metrics]
        scores = _extract_ragas_scores(result, metric_names)
        missing_metrics = [name for name in metric_names if name not in scores]
        if missing_metrics:
            logger.warning(
                "RAGAS did not return usable values for metrics: %s",
                ", ".join(missing_metrics),
            )
            fallback_scores = _fallback_ragas_scores(ground_truth)
            for name in missing_metrics:
                if name in fallback_scores:
                    scores[name] = fallback_scores[name]
        if not scores:
            raise RuntimeError(
                "RAGAS did not return usable metric scores "
                f"(requested: {', '.join(metric_names)})"
            )

        numeric_validation = validate_numeric_answer(answer, query, contexts)
        if numeric_validation["numeric_match"] is not None:
            scores["numeric_match"] = 1.0 if numeric_validation["numeric_match"] else 0.0

        logger.info(
            "RAGAS 评估: faith=%.3f rel=%.3f prec=%.3f",
            scores.get("faithfulness", 0),
            scores.get("answer_relevancy", 0),
            scores.get("context_precision", 0),
        )
        return scores

    except ImportError as e:
        logger.warning("RAGAS 未安装或依赖不完整: %s", e)
        return _fallback_ragas_scores(ground_truth)
    except Exception as e:
        logger.error("RAGAS 评估执行失败, 使用离线降级分数: %s", e)
        return _fallback_ragas_scores(ground_truth)
