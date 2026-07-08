"""
# ============================================================
# ★ Adaptive-RAG 查询分类器节点 (LangGraph Node)
# ← Adaptive-RAG 论文 (NAACL 2024): Query Complexity Classifier
# ← WeKnora: chat_pipeline/query_understand.go — 意图分类 (9分类)
#   我们的分类维度不同: 复杂度三元分类 (simple/medium/complex)
#
# 此节点是 LangGraph 工作流的第一个决策点。
# 它评估查询复杂度后，Graph 通过条件边将请求路由到不同的检索分支。
# ============================================================

本模块是 Adaptive-RAG 工作流的**入口路由器**:
- classify_query(): 使用 LLM 分类查询复杂度
- route_by_complexity(): 条件路由 → 决定走哪个检索分支
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal, Optional

from config.settings import get_settings
from src.graph.state import GraphState
from src.models.llm import LLMClient, get_llm_client
from src.utils.json_parser import parse_llm_json_response
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# 路由目标类型
RouteTarget = Literal["no_retrieval", "single_step", "multi_step"]

SIMPLE_PATTERNS = [
    r"^(你好|您好|hi|hello)[\s!！。.,，]*$",
    r"^(谢谢|感谢|thanks)[\s!！。.,，]*$",
    r"^什么是\S+[？?]?$",
    r"^\S+是什么[？?]?$",
    r"^(帮助|help|怎么用|使用说明)[？?]?$",
]

COMPLEX_PATTERNS = [
    r"(比较|对比|区别|vs\.?|和.*有什么.*不同)",
    r"(为什么|原因|原理|如何实现|怎么做到)",
    r"(分析|评估|总结|概括|归纳)",
]

COMPLEX_DIAGNOSTIC_PATTERNS = [
    r"(异常|矛盾|不一致|到底|是否正常|有没有.*问题)",
    r"(延期|延迟).*(原因|异常|问题|状态)",
]

IMPLICIT_INFERENCE_PATTERNS = [
    r"(谁|哪位|哪个人).*(可能|适合|抽调|支援|帮忙|候选)",
    r"(如果|假如).*(需要|紧急|支援|加人)",
    r"(为什么|原因).*(可能|适合|推荐)",
]


AGGREGATE_PATTERNS = [
    r"(总共|合计|总和|总预算|总支出|总人数|一共)",
    r"(最多|最少|最高|最低|最大|最小).*(?:项目|部门|人|金额|预算|支出)",
    r"(剩余|结余|余额).*(?:最多|最少|哪个|什么)",
    r"(平均|均值|人均)",
]

LOW_COST_RETRIEVAL_PATTERNS = [
    r"(预算|支出|技术栈|负责人|部门|状态|来源).*(多少|什么|哪个|是谁)",
    r"(有多少|多少个|几个|哪些|分别).*(项目|部门|人员|成员)",
    r"(列出|列举).*(所有|全部)?\s*(项目|部门|人员|成员)",
]

SINGLE_FACT_PATTERNS = [
    r"^(?:\S{1,20})项目的?(?:预算|支出|技术栈|负责人|部门|状态)是(?:多少|什么|谁|哪个).*[？?]?$",
    r"^(?:\S{1,20})项目.*(?:预算|支出).*(?:多少).*[？?]?$",
    r"^(?:\S{1,20})项目.*技术栈.*(?:什么|为什么选择).*[？?]?$",
    r"^(?:\S{1,20})(?:属于哪个部门|负责人是谁).*[？?]?$",
]


def is_aggregate_query(query: str) -> bool:
    """Return whether a query needs deterministic aggregate/table handling."""
    normalized = (query or "").strip()
    return any(re.search(pattern, normalized, re.IGNORECASE) for pattern in AGGREGATE_PATTERNS)


def is_low_cost_retrieval_query(query: str) -> bool:
    """Return whether a scoped document query can use a small single-step retrieval."""
    normalized = (query or "").strip()
    if is_complex_diagnostic_query(normalized) or is_implicit_inference_query(normalized):
        return False
    if is_single_fact_query(normalized):
        return True
    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in LOW_COST_RETRIEVAL_PATTERNS
    )


def is_complex_diagnostic_query(query: str) -> bool:
    """Return whether a query asks for diagnosis, contradiction, or abnormality."""
    normalized = (query or "").strip()
    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in COMPLEX_DIAGNOSTIC_PATTERNS
    )


def is_implicit_inference_query(query: str) -> bool:
    """Return whether a query needs evidence-backed implicit inference."""
    normalized = (query or "").strip()
    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in IMPLICIT_INFERENCE_PATTERNS
    )


def is_single_fact_query(query: str) -> bool:
    """Return whether a query asks for facts about one named entity."""
    normalized = (query or "").strip()
    if (
        not normalized
        or is_complex_diagnostic_query(normalized)
        or is_implicit_inference_query(normalized)
    ):
        return False
    return any(
        re.search(pattern, normalized, re.IGNORECASE)
        for pattern in SINGLE_FACT_PATTERNS
    )


def is_list_aggregation_query(query: str) -> bool:
    """Return whether a query asks for a complete list rather than one fact."""
    normalized = (query or "").strip()
    return bool(re.search(r"(有多少|多少个|几个|哪些|分别).*(项目|部门|人员|成员)", normalized)) or bool(
        re.search(r"(列出|列举).*(所有|全部)?\s*(项目|部门|人员|成员)", normalized)
    )


def quick_classify(query: str) -> Optional[str]:
    """Return simple/complex for obvious queries, otherwise None."""
    normalized = (query or "").strip()
    if not normalized:
        return "simple"

    if is_aggregate_query(normalized):
        return "medium"
    if is_implicit_inference_query(normalized):
        return "complex"
    if is_list_aggregation_query(normalized):
        return "medium"
    if is_single_fact_query(normalized):
        return "medium"

    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return "simple"
    if is_complex_diagnostic_query(normalized):
        return "complex"
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return "complex"
    return None


# ================================================================
# LangGraph Node: 查询分类
# ================================================================


async def classify_query(state: GraphState) -> dict:
    """
    ★ Adaptive-RAG 查询复杂度分类节点
    ← Adaptive-RAG 论文 Section 3.1

    使用轻量 LLM (gpt-4o-mini) 将查询分为:
    - simple → 直接回答，跳过检索
    - medium → 单步 BM25 + Dense + Rerank
    - complex → 多步迭代检索 + HyDE + 查询改写

    Args:
        state: GraphState

    Returns:
        state 的部分更新 (complexity, confidence, reasoning)
    """
    query = state.get("query", "")
    conversation = state.get("conversation_context", "")

    if not query:
        return {
            "complexity": "simple",
            "complexity_confidence": 1.0,
            "classification_reasoning": "空查询",
        }

    quick_result = quick_classify(query)
    if quick_result is not None:
        logger.info("[classify_query] quick rule: %s -> %s", query[:60], quick_result)
        result = {
            "complexity": quick_result,
            "complexity_confidence": 1.0,
            "classification_reasoning": "quick_rule",
        }
        if is_low_cost_retrieval_query(query):
            result["retrieval_top_k"] = 5 if is_list_aggregation_query(query) else 3
            result["query_type"] = "low_cost_retrieval"
        if is_aggregate_query(query):
            result["query_type"] = "aggregate"
        return result

    llm = get_llm_client()  # ★ M2: 复用单例

    prompt = load_prompt(
        "complexity_classifier",
        filename="classification",
        query=query,
        conversation=conversation,
        language=get_settings().default_language,
    )

    try:
        response = await llm.ask(prompt=prompt, model_name=get_settings().llm_simple_model)

        result = parse_llm_json_response(response)
        complexity = result.get("complexity", "medium")
        confidence = float(result.get("confidence", 0.5))
        reasoning = result.get("reasoning", "")

        logger.info(
            "[classify_query] %s → %s (confidence=%.2f) '%s'",
            query[:60], complexity, confidence, reasoning,
        )

        result_state = {
            "complexity": complexity,
            "complexity_confidence": confidence,
            "classification_reasoning": reasoning,
        }
        if is_low_cost_retrieval_query(query):
            result_state["retrieval_top_k"] = 5 if is_list_aggregation_query(query) else 3
            result_state["query_type"] = "low_cost_retrieval"
        if is_aggregate_query(query):
            result_state["query_type"] = "aggregate"
        return result_state

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("[classify_query] JSON 解析失败: %s → 默认 medium", e)
        return {
            "complexity": "medium",
            "complexity_confidence": 0.5,
            "classification_reasoning": f"JSON解析失败: {e}",
        }
    except (ConnectionError, TimeoutError) as e:
        # 网络/服务不可用 → 降级到默认分类（预期内的故障）
        logger.warning("[classify_query] 网络异常: %s → 默认 medium", e)
        return {
            "complexity": "medium",
            "complexity_confidence": 0.5,
            "classification_reasoning": f"网络异常降级: {e}",
        }
    except Exception as e:
        # 未预期的代码逻辑错误 → 记录 critical 但仍降级（保证工作流不中断）
        logger.critical("[classify_query] 未预期异常: %s", e, exc_info=True)
        return {
            "complexity": "medium",
            "complexity_confidence": 0.5,
            "classification_reasoning": f"内部错误: {e}",
        }


# ================================================================
# LangGraph 条件路由函数
# ================================================================


def route_by_complexity(state: GraphState) -> RouteTarget:
    """
    ★ 条件路由: 根据复杂度决定下一步走哪个节点
    这是 LangGraph 条件边的核心 —— 根据 state 动态选择路径

    路由规则:
      simple  → "no_retrieval"    (跨过检索，直接生成)
      medium  → "single_step"     (单步检索)
      complex → "multi_step"      (多步迭代检索)
      其他    → "single_step"     (安全默认值)

    Args:
        state: 当前 GraphState

    Returns:
        下一个节点的名称
    """
    complexity = state.get("complexity", "medium")
    confidence = state.get("complexity_confidence", 0.5)
    query = state.get("rewritten_query") or state.get("query", "")
    if is_aggregate_query(query):
        logger.info("[route_by_complexity] aggregate query forced to single_step")
        return "single_step"
    if state.get("retrieval_filter") and complexity == "simple":
        logger.info("[路由决策] active retrieval_filter present; upgrading simple to single_step")
        complexity = "medium"

    route_map: dict[str, RouteTarget] = {
        "simple": "no_retrieval",
        "medium": "single_step",
        "complex": "multi_step",
    }

    target = route_map.get(complexity, "single_step")

    logger.info(
        "[路由决策] %s (confidence=%.2f) → %s",
        complexity, confidence, target,
    )
    return target
