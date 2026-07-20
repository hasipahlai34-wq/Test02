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
import re
from typing import Optional

from langchain_openai import ChatOpenAI

from config.settings import get_settings
from src.graph.state import GraphState
from src.models.llm import LLMClient, get_llm_client
from src.types import SafetyLevel
from src.utils.json_parser import parse_llm_json_response
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

# ================================================================
# ★ 安全 LLM 客户端（独立 endpoint，走 DeepSeek flash）
# ================================================================

_safety_llm_client: Optional[LLMClient] = None


def _get_safety_llm_client() -> LLMClient:
    """获取安全检测专用 LLM 客户端。

    优先使用 SAFETY_BASE_URL / SAFETY_API_KEY 配置的独立 endpoint
    （如 DeepSeek flash），否则回退到默认 LLM 客户端。
    """
    global _safety_llm_client
    if _safety_llm_client is not None:
        return _safety_llm_client

    settings = get_settings()
    safety_base = settings.safety_base_url.strip()
    safety_key = settings.safety_api_key.strip()

    if safety_base and safety_key:
        model = settings.safety_model or settings.llm_default_model
        logger.info(
            "安全 LLM 客户端: model=%s base_url=%s",
            model, safety_base,
        )
        _safety_llm_client = LLMClient(
            model_name=model,
            settings=settings,
            base_url=safety_base,
            api_key=safety_key,
        )
        return _safety_llm_client

    # 回退：使用默认 LLM 客户端
    logger.debug("安全 LLM: 未配置独立 endpoint，复用默认 LLM 客户端")
    return get_llm_client()


# ================================================================
# ★ 优化：输入安全预筛正则（不调用 LLM）
# ================================================================

_INJECTION_PATTERNS = [
    # 英文 SQL 注入 / 越狱
    r"(?i)\b(SELECT\s+.+\s+FROM|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UNION\s+SELECT|1\s*=\s*1|'\s*--|'\s*;\s*)",
    r"(?i)(ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|directives?))",
    r"(?i)(DAN\s+mode|jailbreak|developer\s+mode|override\s+(system|safety))",
    r"(?i)(you\s+are\s+now\s+|from\s+now\s+on\s+you\s+(are|must|will))",
    r"(?i)(pretend|act\s+as\s+if|roleplay|role\s*[-]\s*play)",
    r"(?i)\b(hack(ing)?|exploit|malware|ransomware|phishing|backdoor)\b",
    r"^[A-Za-z0-9+/=]{200,}$",
    # ★ 中文越狱 / 提示注入
    r"(忽略|忘记|无视|跳过).{0,10}(之前|之前所有|上面|上述|系统|所有).{0,10}(指令|提示|提示词|规则|限制|约束)",
    r"(从现在开始|从现在起|现在开始|现在起).{0,5}(你是|你就是|你扮演|你假装|你是我的)",
    r"(扮演|假装|角色扮演|cosplay|模拟).{0,10}(角色|身份|人格|DAN|越狱)",
    r"(输出|打印|告诉我|显示|泄露|暴露).{0,10}(系统提示|system\s*prompt|内部指令|隐藏规则|后台|管理)",
    r"(绕过|突破|解除|取消|禁用).{0,10}(限制|安全|过滤|审查|规则|约束)",
]

_SAFE_QUERY_PATTERNS = [
    r"^[一-鿿\w\s，。？?！!、：:（）()《》""''0-9+%.-]{1,200}$",
]


def _prescreen_input_safety(content: str) -> dict | None:
    """对用户输入做正则预筛。

    Returns:
        None          → 预筛不确定，需走 LLM
        {safe: True}  → 预筛判定安全，跳过 LLM
    """
    if not content or not content.strip():
        return {"safe": False, "risk_level": "low", "detected_issues": ["空输入"]}

    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, content):
            logger.warning(
                "🛡️ 输入预筛命中可疑特征: query='%s...'",
                content[:60],
            )
            return None  # 可疑，走 LLM 深度检测

    for pattern in _SAFE_QUERY_PATTERNS:
        if re.search(pattern, content):
            logger.info(
                "🛡️ 输入预筛安全放行: query='%s...'",
                content[:60],
            )
            return {
                "safe": True,
                "risk_level": "low",
                "detected_issues": [],
                "_prescreened": True,
            }

    return None


# ================================================================
# ★ 优化：文档 QA 领域安全白名单
# ================================================================

_LOW_RISK_DOMAIN_PATTERNS = [
    r"(预算|支出|费用|成本|金额|资金|经费)",
    r"(进度|状态|延期|滞后|交付|上线|发布|截止)",
    r"(技术栈|架构|框架|语言|工具|平台|数据库)",
    r"(负责人|部门|团队|人员|成员|角色|职责)",
    r"(项目|任务|需求|迭代|版本|里程碑)",
    r"(文档|报告|表格|数据|图表|附件|文件)",
    r"(列出|查看|显示|告诉我|查一下|查询)",
    r"^(?:什么是|怎么|如何|多少|几个|哪些|哪个|谁|何时|什么时候)",
]

_HIGH_RISK_PHRASES = [
    r"(法律|诉讼|起诉|违法|犯罪)",
    r"(投资|理财|股票|基金|买入|卖出|推荐.*购买)",
    r"(医疗|诊断|治疗|药物|处方|手术)",
    r"(自杀|自残|伤害|暴力|武器|炸药)",
]


def _is_low_risk_domain(query: str, answer: str, docs: list) -> bool:
    """判断当前问答是否属于低风险的文档 QA 领域。

    条件（全部满足才跳过 LLM 安全检查）：
    1. query 匹配已知的低风险领域关键词
    2. 答案不包含高风险建议措辞
    """
    domain_match = any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in _LOW_RISK_DOMAIN_PATTERNS
    )
    if not domain_match:
        return False

    for phrase in _HIGH_RISK_PHRASES:
        if re.search(phrase, answer, re.IGNORECASE):
            return False

    return True


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
    # ★ 优化：正则预筛快速通道
    if get_settings().opt_input_safety_prescreen:
        prescreen_result = _prescreen_input_safety(content)
        if prescreen_result is not None and prescreen_result.get("safe"):
            return prescreen_result

    if llm_client is None:
        llm_client = _get_safety_llm_client()  # ★ 优化：走安全专用 endpoint (DeepSeek flash)

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

    # ★ 优化：文档 QA 领域快速通道
    if get_settings().opt_output_safety_domain_skip:
        docs = state.get("retrieved_docs", [])
        if _is_low_risk_domain(query, answer, docs):
            logger.info(
                "🛡️ 输出安全域跳过: query='%s...' domain=document_qa",
                query[:60],
            )
            return {
                "safety_risk_level": SafetyLevel.LOW.value,
                "needs_human_review": False,
                "_prescreened": True,
            }

    if llm_client is None:
        llm_client = _get_safety_llm_client()  # ★ 优化：走安全专用 endpoint (DeepSeek flash)

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
