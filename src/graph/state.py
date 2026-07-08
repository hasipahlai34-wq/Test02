"""
# ============================================================
# LangGraph 状态定义
# ← WeKnora: agent.go AgentState struct
#   WeKnora 的 AgentState 只有 5 个字段 (CurrentRound,
#   RoundSteps, IsComplete, FinalAnswer, KnowledgeRefs)
#
#   我们基于 LangGraph 的 AgentState 更加丰富:
#   - 使用 TypedDict + Annotated 定义 LangGraph State Schema
#   - 每个字段可以有 reducer 函数 (合并多节点写入)
#   - 状态在 Graph 的每个 node 之间自动传递
# ============================================================

本模块定义了 LangGraph StateGraph 的状态 Schema。
每个 node 函数接收 state，返回 state 的部分更新 (LangGraph 自动合并)。

设计要点:
- 使用 Annotated + operator.add 实现 list 字段的自动追加 (而非覆盖)
- 关键字段使用 MessagesState 兼容 LangChain 生态
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    """
    LangGraph StateGraph 状态 Schema

    每个字段的 Annotated 注解定义了该字段在多个 node 并行写入时的合并策略:
    - operator.add: 追加 (用于 list)
    - 无注解: 覆盖 (最后一个写入的值)
    - add_messages: LangChain Message 专用的合并逻辑
    """

    # ================================================================
    # 输入输入
    # ================================================================
    query: str                                    # 用户查询
    session_id: str                               # 会话 ID
    messages: Annotated[list, add_messages]        # 对话消息 (LangChain 兼容)

    # ================================================================
    # Adaptive-RAG 路由分类
    # ================================================================
    complexity: str                                # simple / medium / complex
    complexity_confidence: float                   # 分类置信度
    selected_strategy: str                         # 选中的检索策略名称
    classification_reasoning: str                  # 分类理由

    # ================================================================
    # 查询改写
    # ================================================================
    rewritten_query: str                           # 改写后的查询
    keyword_expanded_query: str                    # 关键词扩展
    hyde_hypothesis: str                           # HyDE 假设文档

    # ================================================================
    # 检索结果
    # ================================================================
    retrieved_docs: Annotated[list, operator.add]  # 检索到的文档列表
    search_result_summary: str                     # 检索结果摘要
    search_count: int                              # 检索命中数
    retrieval_filter: Optional[dict]               # active session/document metadata filter

    # ================================================================
    # 生成
    # ================================================================
    system_prompt: str                             # 组装后的系统提示词
    context_prompt: str                            # 组装后的上下文 (检索内容)
    generated_answer: str                          # 生成的回答
    answer_stream: Optional[Any]                   # 流式回答 (AsyncGenerator)
    from_cache: bool
    cache_hit: bool
    cache_lookup_error: Optional[str]

    # ================================================================
    # 审核 + 安全
    # ================================================================
    quality_score: float                           # 质量评分
    quality_passed: bool                           # 质量审核是否通过
    safety_risk_level: str                         # 安全风险等级
    needs_human_review: bool                       # 是否需要人工审核
    review_reason: str                             # 审核理由

    # ================================================================
    # HITL 人机协同 (新增)
    # ================================================================
    hitl_status: str                               # "none" | "pending" | "approved" | "rejected" | "edited" | "pending_timeout"
    hitl_review_id: Optional[str]                  # 审核项 ID (UUID)
    hitl_decision: Optional[str]                   # 人工决策: "approve" | "reject" | "edit"
    hitl_edited_answer: Optional[str]              # 人工编辑后的答案
    hitl_trigger_reasons: Annotated[list, operator.add]  # 触发 HITL 的原因列表

    # ================================================================
    # RAGAS 在线评估 (新增)
    # ================================================================
    ragas_scores: Optional[dict]                   # RAGAS 在线评估分数
    ragas_eval_error: Optional[str]                # RAGAS 评估错误 (非阻塞)
    ragas_review_failed: bool                      # review 未通过时执行 RAGAS 的标记

    # ================================================================
    # 熔断器
    # ================================================================
    circuit_quality_state: str                     # 质量熔断器状态
    circuit_freq_state: str                        # 频率熔断器状态

    # ================================================================
    # Agent 步骤追踪
    # ================================================================
    agent_steps: Annotated[list, operator.add]     # Agent 执行步骤列表
    current_iteration: int                         # 当前迭代

    # ================================================================
    # 记忆
    # ================================================================
    conversation_context: str                      # 对话历史文本
    relevant_memories: Annotated[list, operator.add]  # 相关记忆

    # ================================================================
    # Token 追踪
    # ================================================================
    total_tokens_used: int                         # 总 Token 消耗
    token_budget: int                              # Token 预算

    # ================================================================
    # 评估 (compare 模式 + 在线评估)
    # ================================================================
    ground_truth: Optional[str]                    # 标准答案 (评估用)

    # ================================================================
    # 元信息
    # ================================================================
    completed: bool                                # 是否完成
    error: Optional[str]                           # 错误信息
    node_times: Optional[dict]                     # 各节点耗时
