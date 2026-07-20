"""
# ============================================================
# 审核 Agent 节点
# ← WeKnora: engine.go analyzeResponse() — 响应分析 + 停止条件判断
#           检查: 连续相同内容、空响应、幻觉标记
# ← 原项目 B: 质量熔断逻辑 (质量评分 + 不达标则拒绝)
# ============================================================

本模块是审核 Agent 的 LangGraph 节点实现。
负责:
1. 答案质量评分 (忠实度 + 相关性 + 完整性)
2. 幻觉检测
3. 更新熔断器统计
4. 决定是否需要重新生成
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config.settings import get_settings
from src.graph.state import GraphState
from src.models.llm import LLMClient, get_llm_client
from src.types import Document
from src.utils.json_parser import parse_llm_json_response
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def _select_review_docs(docs: list[Document], limit: int = 12) -> list[Document]:
    """Pick evidence-rich documents for answer review.

    Structure-aware retrieval may return outline/heading chunks before section
    chunks. Reviewing only the first few chunks can therefore miss the actual
    evidence and falsely flag correct answers as hallucinations.
    """
    outline_docs = []
    content_docs = []
    fallback_docs = []

    for doc in docs:
        metadata = getattr(doc, "metadata", {}) or {}
        element_type = str(metadata.get("element_type") or "")
        chunk_type = str(metadata.get("chunk_type") or "")
        content = str(getattr(doc, "content", "") or "")
        stripped = content.strip()

        if element_type == "outline" or chunk_type == "document_outline":
            outline_docs.append(doc)
        elif element_type in {"section", "table", "row_group"} or len(stripped) >= 80:
            content_docs.append(doc)
        else:
            fallback_docs.append(doc)

    selected: list[Document] = []
    if outline_docs:
        selected.append(outline_docs[0])

    for doc in content_docs + fallback_docs:
        if doc not in selected:
            selected.append(doc)
        if len(selected) >= limit:
            break

    return selected


async def review_answer(state: GraphState) -> dict:
    """
    LangGraph 审核节点: 评估答案质量
    ← WeKnora: engine.go analyzeResponse()
    ← 原项目 B: 质量熔断

    审核维度:
    - 忠实度 (Faithfulness): 答案是否基于检索内容
    - 相关性 (Relevance): 答案是否切题
    - 完整性 (Completeness): 答案是否覆盖所有方面
    - 幻觉检测: 是否存在编造的信息

    Args:
        state: 当前 GraphState

    Returns:
        state 部分更新 (quality_score, quality_passed, etc.)
    """
    answer = state.get("generated_answer", "")
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])

    # 没有答案 → 跳过审核
    if not answer:
        return {
            "quality_score": 0.0,
            "quality_passed": False,
        }

    # 没有检索内容 → 跳过忠实度审核 (直接回答模式)
    if not docs:
        # 简单检查: 回答非空即可
        return {
            "quality_score": 0.7,
            "quality_passed": True,
        }

    # ---- ★ 优化：高置信度快速通道 ----
    # medium 查询 + 检索质量高 + 答案简短 → 跳过 LLM 审核
    complexity = state.get("complexity", "medium")
    if get_settings().opt_review_fast_path and complexity == "medium":
        avg_score = (
            sum(float(getattr(d, "score", 0) or 0) for d in docs) / len(docs)
            if docs else 0
        )
        is_short = len(answer) <= 250
        has_uncertainty = any(
            phrase in answer
            for phrase in ("可能", "大概", "据称", "据说", "或许", "也许", "不太确定", "没有找到")
        )
        if avg_score >= 0.65 and is_short and not has_uncertainty:
            logger.info(
                "[review] 快速通道: avg_score=%.2f answer_len=%d → 跳过 LLM 审核",
                avg_score, len(answer),
            )
            return {
                "quality_score": 0.8,
                "quality_passed": True,
                "review_reason": "fast_path_high_confidence",
            }

    # 组装检索内容。优先使用正文 section，避免审核器只看到标题/大纲后误判。
    review_docs = _select_review_docs(docs)
    contexts = "\n---\n".join(
        f"[文档{i+1}] {doc.content[:1000]}"
        for i, doc in enumerate(review_docs)
    )

    # LLM 质量评估
    llm = get_llm_client()  # ★ M2: 复用单例
    prompt = load_prompt(
        "quality_review",
        filename="quality_review",
        query=query,
        answer=answer[:2000],  # 截断长答案
        contexts=contexts,
        language=get_settings().default_language,
    )

    try:
        response = await llm.ask(prompt=prompt, model_name=get_settings().llm_simple_model)

        result = parse_llm_json_response(response)

        faithfulness = float(result.get("faithfulness", 0.8))
        relevance = float(result.get("relevance", 0.8))
        completeness = float(result.get("completeness", 0.7))
        has_hallucination = result.get("has_hallucination", False)
        passed = result.get("passed", True)
        overall = float(result.get("overall_score", 0.8))
        suggestion = result.get("suggestion", "")

        logger.info(
            "[review] faith=%.2f rel=%.2f comp=%.2f overall=%.2f passed=%s",
            faithfulness, relevance, completeness, overall, passed,
        )

        if has_hallucination:
            logger.warning(
                "[review] ⚠️ 检测到幻觉: %s",
                result.get("hallucination_details", []),
            )

        return {
            "quality_score": overall,
            "quality_passed": passed,
        }

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("[review] JSON 解析失败: %s → 默认不通过", e)
        return {
            "quality_score": 0.0,
            "quality_passed": False,
        }
    except Exception as e:
        logger.error("[review] 质量评估异常: %s → 默认不通过 (fail-safe)", e, exc_info=True)
        return {
            "quality_score": 0.0,
            "quality_passed": False,
        }
