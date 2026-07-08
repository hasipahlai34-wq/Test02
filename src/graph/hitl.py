"""
# ============================================================
# ★ HITL 人机协同审核节点 (LangGraph Node)
# ← 高风险回答暂停工作流, 等待人工确认
#
# 设计要点:
# - 双模式: interrupt (Streamlit) / file_queue (CLI/API)
# - 统一 JSON 队列格式 (两种模式使用相同格式)
# - 30 分钟超时: 超时后保持文件队列, 不自动放行
# - 拒绝处理: 返回 "回答未通过质量审核，请重新提问"
#
# 触发条件 (任一满足即触发):
# 1. quality_passed == False (review 未通过)
# 2. needs_human_review == True (guard/安全检测标记)
# 3. ragas_scores 任意指标低于阈值
# 4. safety_risk_level in ("high", "critical")
#
# 面试可讲:
# "我实现了 Human-in-the-Loop 审核机制:
#  当回答质量评分低、安全风险高或 RAGAS 指标不达标时,
#  工作流自动暂停并写入待审核队列。
#  在 Streamlit UI 中直接弹出审核界面, 支持 30 分钟超时;
#  CLI/API 模式下降级为文件队列, 提供独立审核工具。"
# ============================================================
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from src.graph.state import GraphState

logger = logging.getLogger(__name__)

# ================================================================
# 队列 JSON 格式常量 (两种模式统一)
# ================================================================

QUEUE_ITEM_KEYS = [
    "review_id", "session_id", "query", "answer",
    "retrieved_docs_preview", "search_count",
    "complexity", "selected_strategy",
    "quality_score", "quality_passed",
    "ragas_scores", "ragas_review_failed",
    "safety_risk_level", "needs_human_review",
    "trigger_reasons", "review_reason",
    "hitl_status", "hitl_decision", "hitl_edited_answer",
    "created_at", "updated_at", "timeout_at", "mode",
]


class HITLGate:
    """Compatibility wrapper for callers that expect a HITL gate object."""

    async def __call__(self, state: GraphState) -> dict:
        return await hitl_gate_node(state)


def _is_streamlit_runtime() -> bool:
    """检测是否在 Streamlit 运行时环境中"""
    try:
        from streamlit.runtime import exists
        return exists()
    except (ImportError, RuntimeError):
        return False


# ================================================================
# 队列管理
# ================================================================


def _ensure_queue_dir() -> Path:
    """确保队列目录存在"""
    settings = get_settings()
    queue_dir = Path(settings.hitl_queue_dir)
    results_dir = Path(settings.hitl_results_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return queue_dir


def _build_queue_item(state: GraphState, review_id: str, trigger_reasons: list[str]) -> dict:
    """
    构建统一的审核队列项

    格式向后兼容, 供 interrupt 和 file_queue 两种模式使用。
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)

    docs = state.get("retrieved_docs", [])
    docs_preview = []
    for doc in docs[:5]:
        if hasattr(doc, "content"):
            docs_preview.append({
                "content": str(doc.content)[:300],
                "score": doc.score if hasattr(doc, "score") else None,
                "source": doc.metadata.get("source", "") if hasattr(doc, "metadata") else "",
            })

    item = {
        "review_id": review_id,
        "session_id": state.get("session_id", ""),
        "query": state.get("query", ""),
        "answer": state.get("generated_answer", ""),
        "retrieved_docs_preview": docs_preview,
        "search_count": state.get("search_count", 0),
        "complexity": state.get("complexity", ""),
        "selected_strategy": state.get("selected_strategy", ""),
        "quality_score": state.get("quality_score", 0),
        "quality_passed": state.get("quality_passed", True),
        "ragas_scores": state.get("ragas_scores"),
        "ragas_review_failed": state.get("ragas_review_failed", False),
        "safety_risk_level": state.get("safety_risk_level", "low"),
        "needs_human_review": state.get("needs_human_review", False),
        "trigger_reasons": trigger_reasons,
        "review_reason": state.get("review_reason", ""),
        "hitl_status": "pending",
        "hitl_decision": None,
        "hitl_edited_answer": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "timeout_at": (now + timedelta(seconds=settings.hitl_interrupt_timeout_seconds)).isoformat(),
        "mode": "interrupt" if _is_streamlit_runtime() else "file_queue",
    }
    return item


def write_queue_item(item: dict) -> Path:
    """将审核项写入文件队列"""
    queue_dir = _ensure_queue_dir()
    filepath = queue_dir / f"{item['review_id']}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)
    logger.info("HITL 队列写入: %s (mode=%s)", filepath.name, item["mode"])
    return filepath


def update_queue_item(review_id: str, updates: dict) -> Optional[Path]:
    """更新队列项状态"""
    queue_dir = _ensure_queue_dir()
    filepath = queue_dir / f"{review_id}.json"

    if not filepath.exists():
        # 检查是否已移动到 results_dir
        settings = get_settings()
        results_dir = Path(settings.hitl_results_dir)
        result_path = results_dir / f"{review_id}.json"
        if result_path.exists():
            logger.warning("HITL 队列项已归档: %s", review_id)
            return None
        logger.warning("HITL 队列项不存在: %s", review_id)
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        item = json.load(f)

    item.update(updates)
    item["updated_at"] = datetime.now(timezone.utc).isoformat()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=2)

    return filepath


def archive_queue_item(review_id: str) -> Optional[Path]:
    """将已处理的审核项移动到 results 目录"""
    settings = get_settings()
    queue_dir = Path(settings.hitl_queue_dir)
    results_dir = Path(settings.hitl_results_dir)

    filepath = queue_dir / f"{review_id}.json"
    if not filepath.exists():
        logger.warning("HITL 归档失败: 文件不存在 %s", review_id)
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / f"{review_id}.json"
    filepath.rename(dest)
    logger.info("HITL 归档: %s → %s", review_id, dest)
    return dest


# ================================================================
# 触发条件评估
# ================================================================


def _evaluate_trigger_reasons(state: GraphState) -> list[str]:
    """
    评估所有 HITL 触发条件, 返回触发原因列表

    检查顺序:
    1. quality_passed == False → 质量审核未通过
    2. needs_human_review == True → 安全检测标记
    3. safety_risk_level high/critical → 安全高风险
    4. ragas_scores 各项低于阈值
    """
    settings = get_settings()
    reasons: list[str] = []

    # 1. 质量审核未通过
    if not state.get("quality_passed", True):
        reasons.append("quality_not_passed")

    # 2-3. 安全检测
    if state.get("needs_human_review", False):
        reasons.append("safety_flagged")
    safety_level = state.get("safety_risk_level", "low")
    if safety_level in ("high", "critical"):
        reasons.append(f"safety_{safety_level}")

    # 4. RAGAS 指标
    ragas = state.get("ragas_scores")
    if ragas and isinstance(ragas, dict):
        faith = ragas.get("faithfulness", 1.0)
        relev = ragas.get("answer_relevancy", 1.0)
        prec = ragas.get("context_precision", 1.0)

        if faith < settings.ragas_faithfulness_threshold:
            reasons.append(
                f"ragas_faithfulness_{faith:.2f}_lt_{settings.ragas_faithfulness_threshold}"
            )
        if relev < settings.ragas_relevancy_threshold:
            reasons.append(
                f"ragas_relevancy_{relev:.2f}_lt_{settings.ragas_relevancy_threshold}"
            )
        if prec < settings.ragas_context_precision_threshold:
            reasons.append(
                f"ragas_precision_{prec:.2f}_lt_{settings.ragas_context_precision_threshold}"
            )

    return reasons


# ================================================================
# ★ LangGraph HITL Gate 节点
# ================================================================


async def hitl_gate_node(state: GraphState) -> dict:
    """
    ★ HITL 审核门禁节点

    在所有质量/安全检测之后执行, 综合判断是否需要人工介入。

    模式切换:
    - Streamlit 运行时 → interrupt() 暂停工作流
    - CLI/API 模式 → 文件队列 (不暂停)

    超时处理:
    - interrupt 模式: 30 分钟后超时, 保持文件队列, 不自动放行
    - 超时由调用方 (Streamlit UI) 通过 Command(resume=...) 实现

    拒绝处理:
    - 被拒绝的回答替换为 "回答未通过质量审核，请重新提问"

    Args:
        state: 当前 GraphState

    Returns:
        state 部分更新 (hitl_status, hitl_review_id, hitl_decision, etc.)
    """
    settings = get_settings()

    # 配置开关: 关闭时跳过
    if not settings.hitl_enabled:
        return {
            "hitl_status": "none",
            "hitl_review_id": None,
            "hitl_decision": None,
            "hitl_edited_answer": None,
            "hitl_trigger_reasons": [],
        }

    # 评估触发条件
    trigger_reasons = _evaluate_trigger_reasons(state)

    if not trigger_reasons:
        # 无需审核 → 直接放行
        logger.debug("HITL: 无需审核, 放行")
        return {
            "hitl_status": "none",
            "hitl_review_id": None,
            "hitl_decision": "pass",
            "hitl_edited_answer": None,
            "hitl_trigger_reasons": [],
        }

    # 生成审核 ID
    review_id = str(uuid.uuid4())[:12]  # 短 ID, 便于日志

    # 构建并写入队列项
    queue_item = _build_queue_item(state, review_id, trigger_reasons)
    write_queue_item(queue_item)

    logger.warning(
        "HITL 触发: id=%s reasons=%s",
        review_id, trigger_reasons,
    )

    # 模式选择
    is_streamlit = _is_streamlit_runtime()

    if is_streamlit:
        # ---- interrupt 模式: 暂停工作流 ----
        try:
            from langgraph.types import interrupt

            logger.info(
                "HITL interrupt 模式: id=%s (timeout=%ds)",
                review_id, settings.hitl_interrupt_timeout_seconds,
            )

            # interrupt() 会暂停 Graph 执行, 等待 Command(resume=...)
            # 返回值为 resume 时传入的数据。
            #
            # 超时机制: interrupt() 本身不超时 — 依赖调用方 (Streamlit UI)
            # 在超时后发送 Command(resume={"hitl_decision": "pending_timeout"})。
            # 如果调用方未实现超时逻辑, interrupt 将无限期阻塞。
            # Streamlit UI 应在 app.ainvoke() 外层使用 asyncio.wait_for()
            # 或在 UI 层设置定时器自动 resume。
            decision_data = interrupt(queue_item)

            # ---- Graph 已恢复, 处理决策 ----
            decision = decision_data.get("hitl_decision", "pending_timeout") if decision_data else "pending_timeout"
            edited_answer = decision_data.get("hitl_edited_answer") if decision_data else None

            now = datetime.now(timezone.utc)
            timeout_at = datetime.fromisoformat(queue_item["timeout_at"])
            is_timeout = now > timeout_at

            if decision == "approve":
                update_queue_item(review_id, {
                    "hitl_status": "approved",
                    "hitl_decision": "approve",
                    "updated_at": now.isoformat(),
                })
                archive_queue_item(review_id)
                return {
                    "hitl_status": "approved",
                    "hitl_review_id": review_id,
                    "hitl_decision": "approve",
                    "hitl_edited_answer": None,
                    "hitl_trigger_reasons": trigger_reasons,
                }

            elif decision == "reject":
                update_queue_item(review_id, {
                    "hitl_status": "rejected",
                    "hitl_decision": "reject",
                    "updated_at": now.isoformat(),
                })
                archive_queue_item(review_id)
                logger.warning("HITL 拒绝: id=%s → 返回拒绝提示", review_id)
                return {
                    "hitl_status": "rejected",
                    "hitl_review_id": review_id,
                    "hitl_decision": "reject",
                    "hitl_edited_answer": None,
                    "hitl_trigger_reasons": trigger_reasons,
                    "generated_answer": "回答未通过质量审核，请重新提问",
                }

            elif decision == "edit":
                update_queue_item(review_id, {
                    "hitl_status": "edited",
                    "hitl_decision": "edit",
                    "hitl_edited_answer": edited_answer,
                    "updated_at": now.isoformat(),
                })
                archive_queue_item(review_id)
                logger.info("HITL 编辑通过: id=%s answer_len=%d", review_id, len(edited_answer or ""))
                return {
                    "hitl_status": "edited",
                    "hitl_review_id": review_id,
                    "hitl_decision": "edit",
                    "hitl_edited_answer": edited_answer,
                    "hitl_trigger_reasons": trigger_reasons,
                    "generated_answer": edited_answer or state.get("generated_answer", ""),
                }

            else:
                # pending_timeout 或其他
                if is_timeout:
                    update_queue_item(review_id, {
                        "hitl_status": "pending_timeout",
                        "hitl_decision": "pending_timeout",
                        "updated_at": now.isoformat(),
                    })
                logger.warning(
                    "HITL 超时/未知决策: id=%s decision=%s → 保持待审核",
                    review_id, decision,
                )
                return {
                    "hitl_status": "pending_timeout",
                    "hitl_review_id": review_id,
                    "hitl_decision": decision,
                    "hitl_edited_answer": None,
                    "hitl_trigger_reasons": trigger_reasons,
                }

        except ImportError:
            logger.warning(
                "HITL: langgraph.types.interrupt 不可用, 降级为文件队列模式"
            )
            is_streamlit = False  # 触发下面的降级逻辑
        except Exception as e:
            logger.error("HITL interrupt 异常: %s → 降级为文件队列", e)
            is_streamlit = False

    # ---- 文件队列模式 (非 Streamlit 或降级) ----
    if not is_streamlit:
        # 不暂停流程, 但标记为待审核
        # guard 节点的 needs_human_review 和 review_reason 已记录
        return {
            "hitl_status": "pending",
            "hitl_review_id": review_id,
            "hitl_decision": None,
            "hitl_edited_answer": None,
            "hitl_trigger_reasons": trigger_reasons,
        }


# ================================================================
# HITL 超时辅助 (供 Streamlit UI 等调用方使用)
# ================================================================


async def invoke_with_hitl_timeout(
    app,
    initial_state: GraphState,
    config: dict,
    timeout_seconds: int | None = None,
) -> GraphState:
    """带 HITL 超时的 graph 调用包装器。

    在 interrupt 模式下，interrupt() 无内置超时 — 依赖调用方实现。
    此函数用 asyncio.wait_for() 包装 app.ainvoke(),
    超时后自动将最后一轮审核项标记为 pending_timeout。

    Args:
        app: 编译后的 StateGraph
        initial_state: 初始 GraphState
        config: LangGraph config (含 thread_id)
        timeout_seconds: 超时秒数, 默认读取 settings.hitl_interrupt_timeout_seconds

    Returns:
        最终 GraphState (含超时决策)

    Raises:
        asyncio.TimeoutError: 仅当 graph 在非 interrupt 节点超时时抛出
    """
    import asyncio

    settings = get_settings()
    if timeout_seconds is None:
        timeout_seconds = settings.hitl_interrupt_timeout_seconds

    try:
        final_state = await asyncio.wait_for(
            app.ainvoke(initial_state, config),
            timeout=timeout_seconds,
        )
        return final_state
    except asyncio.TimeoutError:
        logger.warning(
            "HITL: graph 调用超时 (%ds), 检查是否有 pending 审核项",
            timeout_seconds,
        )
        # 查找并标记超时项
        pending = list_pending_items()
        for item in pending:
            update_queue_item(item["review_id"], {
                "hitl_status": "pending_timeout",
                "hitl_decision": "pending_timeout",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            logger.info("HITL: 超时标记 %s", item["review_id"])

        # 重新抛出, 让调用方决定如何处理
        raise


# ================================================================
# 队列查询工具 (供 CLI 审核工具使用)
# ================================================================


def list_pending_items() -> list[dict]:
    """列出所有待审核项"""
    settings = get_settings()
    queue_dir = Path(settings.hitl_queue_dir)
    if not queue_dir.exists():
        return []

    items = []
    for filepath in sorted(queue_dir.glob("*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                item = json.load(f)
            if item.get("hitl_status") == "pending" or item.get("hitl_status") == "pending_timeout":
                items.append(item)
        except Exception as e:
            logger.warning("HITL: 读取队列文件失败 %s: %s", filepath.name, e)

    return items


def get_pending_item(review_id: str) -> Optional[dict]:
    """获取单个待审核项"""
    settings = get_settings()
    queue_dir = Path(settings.hitl_queue_dir)
    filepath = queue_dir / f"{review_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
