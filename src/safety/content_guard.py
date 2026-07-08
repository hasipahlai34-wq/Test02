"""
# ============================================================
# ★ 内容安全护栏 + HITL 人机协同
# ← 原项目 B 特性
# ← WeKnora 无此功能 (GAP_ANALYSIS.md #6: Guardrails 完全缺失)
#
# 双层护栏:
# 1. 输入护栏: 检测提示注入、越狱尝试、敏感内容
# 2. 输出护栏: 幻觉检测、高风险答案标记、HITL 触发
# ============================================================

本模块实现了 AI 应用的内容安全保障:
- check_input_safety(): 输入层 → 提示注入检测 + 敏感过滤
- check_output_safety(): 输出层 → 幻觉检测 + HITL 触发
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config.settings import get_settings
from src.graph.state import GraphState
from src.models.llm import LLMClient, get_llm_client
from src.types import SafetyLevel
from src.utils.json_parser import parse_llm_json_response
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


# ================================================================
# 输入护栏 (← 原项目 B 输入层)
# ================================================================


async def check_input_safety(
    content: str,
    llm_client: Optional[LLMClient] = None,
) -> dict:
    """
    输入安全检测 — 提示注入、越狱、敏感内容
    ← 原项目 B 输入护栏

    Args:
        content: 用户输入内容
        llm_client: LLM 客户端

    Returns:
        {"safe": bool, "risk_level": str, "issues": list}
    """
    if llm_client is None:
        llm_client = get_llm_client()  # ★ M2: 复用单例

    prompt = load_prompt(
        "input_guard",
        filename="guardrails",
        content=content,
    )

    try:
        settings = get_settings()
        safety_model = settings.safety_model or settings.llm_default_model
        response = await llm_client.ask(prompt=prompt, model_name=safety_model)
        result = parse_llm_json_response(response)
        if not result:
            raise json.JSONDecodeError("empty or invalid JSON object", response, 0)
        safe = result.get("safe", True)
        risk_level = result.get("risk_level", "low")

        if not safe:
            logger.warning(
                "🛡️ 输入护栏拦截: risk=%s issues=%s",
                risk_level, result.get("detected_issues", []),
            )

        return result

    except json.JSONDecodeError as e:
        logger.warning("输入安全检测 JSON 解析失败: %s → 默认拦截", e)
        return {"safe": False, "risk_level": "high", "detected_issues": ["安全检测响应格式异常"]}
    except (ConnectionError, TimeoutError) as e:
        logger.warning("输入安全检测网络异常: %s → 默认拦截", e)
        return {"safe": False, "risk_level": "high", "detected_issues": ["安全检测服务不可用"]}
    except Exception as e:
        logger.critical("输入安全检测未预期异常: %s → 默认拦截 (fail-safe)", e, exc_info=True)
        return {"safe": False, "risk_level": "high", "detected_issues": ["安全检测内部错误"]}


# ================================================================
# 输出护栏 (← 原项目 B 输出层 + HITL)
# ================================================================


async def check_output_safety(
    state: GraphState,
    llm_client: Optional[LLMClient] = None,
) -> dict:
    """
    输出安全检测 — LaGraph 节点实现
    ← 原项目 B 输出护栏 + HITL

    检测 AI 生成答案的安全风险:
    - low → 安全，直接返回
    - medium → 轻微风险，标注"仅供参考"后返回
    - high → 明显风险，标记需要人工审核
    - critical → 严重风险，拦截

    HITL 触发条件:
    1. 风险等级为 high 或 critical
    2. 答案涉及法律、医疗、投资等高敏感领域
    3. 幻觉检测标记了严重的事实错误

    Args:
        state: GraphState
        llm_client: LLM 客户端

    Returns:
        state 部分更新 (safety_risk_level, needs_human_review, etc.)
    """
    answer = state.get("generated_answer", "")
    query = state.get("query", "")

    if not answer:
        return {"safety_risk_level": SafetyLevel.LOW.value, "needs_human_review": False}

    if llm_client is None:
        llm_client = get_llm_client()  # ★ M2: 复用单例

    prompt = load_prompt(
        "output_guard",
        filename="guardrails",
        content=answer[:2000],
        query=query,
    )

    try:
        settings = get_settings()
        safety_model = settings.safety_model or settings.llm_default_model
        response = await llm_client.ask(prompt=prompt, model_name=safety_model)
        result = parse_llm_json_response(response)
        if not result:
            raise json.JSONDecodeError("empty or invalid JSON object", response, 0)

        safe = result.get("safe", True)
        risk_level = result.get("risk_level", "low")
        needs_review = result.get("needs_human_review", False)
        review_reason = result.get("review_reason", "")
        suggested_action = result.get("suggested_action", "return")

        if needs_review:
            logger.warning(
                "🛡️ HITL 触发: risk=%s reason='%s' action=%s",
                risk_level, review_reason, suggested_action,
            )

        return {
            "safety_risk_level": risk_level,
            "needs_human_review": needs_review,
            "review_reason": review_reason,
        }

    except json.JSONDecodeError as e:
        logger.warning("输出安全检测 JSON 解析失败: %s → 默认高危拦截", e)
        return {
            "safety_risk_level": SafetyLevel.HIGH.value,
            "needs_human_review": True,
        }
    except (ConnectionError, TimeoutError) as e:
        logger.warning("输出安全检测网络异常: %s → 默认高危拦截", e)
        return {
            "safety_risk_level": SafetyLevel.HIGH.value,
            "needs_human_review": True,
        }
    except Exception as e:
        logger.critical("输出安全检测未预期异常: %s → 默认高危拦截 (fail-safe)", e, exc_info=True)
        return {
            "safety_risk_level": SafetyLevel.HIGH.value,
            "needs_human_review": True,
        }


# ================================================================
# 便捷函数
# ================================================================


async def full_safety_check(
    query: str,
    answer: str,
    llm_client: Optional[LLMClient] = None,
) -> dict:
    """
    完整的安全检查: 输入 + 输出

    Returns:
        {"input_safe": bool, "output_safe": bool,
         "needs_review": bool, "details": dict}
    """
    input_result = await check_input_safety(query, llm_client)
    if not input_result.get("safe", True):
        return {
            "input_safe": False,
            "output_safe": True,
            "needs_review": True,
            "details": {"input": input_result},
        }

    # 构造最小 GraphState 用于输出安全检测。
    # 注意: GraphState 使用 total=False (所有字段可选),
    # 因此可以只传入必要字段。如果将来改为 total=True,
    # 需要同步更新此构造。
    output_result = await check_output_safety(
        GraphState(generated_answer=answer, query=query),
        llm_client,
    )

    return {
        "input_safe": True,
        "output_safe": output_result.get("safety_risk_level", "low") != "critical",
        "needs_review": output_result.get("needs_human_review", False),
        "details": {"output": output_result},
    }
