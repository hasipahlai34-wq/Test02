"""
# ============================================================
# ★ RAGAS 在线评估节点 (LangGraph Node)
# ← 每次查询后自动执行, 异步非阻塞
#
# 设计要点:
# - 评估失败不影响主流程 (非阻塞)
# - review 未通过时仍执行评估, 标记 ragas_review_failed=True
# - 评估结果写入 GraphState.ragas_scores
# - 低于阈值时记录警告, 供 HITL 节点决策
#
# 面试可讲:
# "我在 RAG 工作流中加入了在线评估闭环:
#  每次查询后自动运行 RAGAS 指标, 评估结果既用于
#  实时质量监控, 也作为 HITL 审核的触发依据。
#  评估是异步非阻塞的 — 失败不影响用户获取答案。"
# ============================================================
"""

from __future__ import annotations

import logging
from typing import Optional

from config.settings import get_settings
from src.graph.state import GraphState
from src.evaluation.ragas_eval import evaluate_ragas

logger = logging.getLogger(__name__)


async def ragas_evaluate_node(state: GraphState) -> dict:
    """
    ★ RAGAS 在线评估节点 — 每次查询后自动评分

    在 review 节点之后、guard 节点之前执行。
    提取 query/generated_answer/retrieved_docs, 调用 RAGAS 指标计算。

    行为:
    - 正常: 计算 faithfulness / answer_relevancy / context_precision
    - review 未通过: 仍然执行评估, 但设置 ragas_review_failed=True
    - 评估失败: 记录错误到 ragas_eval_error, 不阻塞流程
    - 配置关闭: 跳过评估, 返回空

    Args:
        state: 当前 GraphState

    Returns:
        state 部分更新 (ragas_scores, ragas_eval_error, ragas_review_failed)
    """
    settings = get_settings()

    # 配置开关: 关闭时跳过
    if not settings.ragas_online_enabled:
        logger.debug("RAGAS 在线评估已关闭 (ragas_online_enabled=False)")
        return {
            "ragas_scores": None,
            "ragas_eval_error": None,
            "ragas_review_failed": False,
        }

    query = state.get("query", "")
    answer = state.get("generated_answer", "")
    docs = state.get("retrieved_docs", [])
    quality_passed = state.get("quality_passed", True)
    ground_truth = state.get("ground_truth")

    # 没有答案 → 跳过
    if not answer:
        logger.debug("RAGAS 在线评估: 无生成答案, 跳过")
        return {
            "ragas_scores": None,
            "ragas_eval_error": "无生成答案",
            "ragas_review_failed": False,
        }

    # 提取文档内容为字符串列表
    contexts: list[str] = []
    for doc in docs[:10]:  # 最多取前 10 个文档
        if hasattr(doc, "content"):
            contexts.append(str(doc.content)[:1000])  # 截断长文档

    review_failed = not quality_passed

    try:
        logger.info(
            "RAGAS 在线评估: query='%s...' contexts=%d answer_len=%d review_failed=%s",
            query[:50], len(contexts), len(answer), review_failed,
        )

        scores = await evaluate_ragas(
            query=query,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

        # 检查阈值
        faith = scores.get("faithfulness")
        relev = scores.get("answer_relevancy")
        prec = scores.get("context_precision")

        below_threshold: list[str] = []
        if faith is not None and faith < settings.ragas_faithfulness_threshold:
            below_threshold.append(
                f"faithfulness={faith:.3f} < {settings.ragas_faithfulness_threshold}"
            )
        if relev is not None and relev < settings.ragas_relevancy_threshold:
            below_threshold.append(
                f"answer_relevancy={relev:.3f} < {settings.ragas_relevancy_threshold}"
            )
        if prec is not None and prec < settings.ragas_context_precision_threshold:
            below_threshold.append(
                f"context_precision={prec:.3f} < {settings.ragas_context_precision_threshold}"
            )

        if below_threshold:
            logger.warning(
                "RAGAS 在线评估: 低于阈值 → %s",
                ", ".join(below_threshold),
            )

        logger.info(
            "RAGAS 在线评估完成: faith=%.3f relev=%.3f prec=%.3f",
            faith or 0, relev or 0, prec or 0,
        )

        return {
            "ragas_scores": scores,
            "ragas_eval_error": None,
            "ragas_review_failed": review_failed,
        }

    except ImportError:
        logger.warning("RAGAS 未安装, 跳过在线评估")
        return {
            "ragas_scores": None,
            "ragas_eval_error": "RAGAS 未安装",
            "ragas_review_failed": review_failed,
        }
    except Exception as e:
        logger.warning("RAGAS 在线评估失败 (非阻塞): %s", e)
        return {
            "ragas_scores": None,
            "ragas_eval_error": str(e)[:200],
            "ragas_review_failed": review_failed,
        }
