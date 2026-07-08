"""
# ============================================================
# 系统核心数据模型 (Pydantic)
# ← WeKnora: internal/types/ 下的所有 struct 定义
#    - chat_manage.go → PipelineRequest, PipelineState, ChatManage
#    - agent.go → AgentState, AgentStep, ToolCall, ToolResult
#    - search.go → SearchResult, SearchParams
#    - message.go → Message, History
#    - session.go → Session, SummaryConfig, ContextConfig
# ============================================================

本模块定义了系统所有核心数据结构的 Pydantic 模型。
每个模型都是系统各组件之间的**数据契约**——LangGraph State、检索结果、
Agent 步骤、记忆条目等全部由此定义。

设计原则:
- 使用 Pydantic v2 以获得更好的序列化性能和类型检查
- 模型之间保持清晰的层次关系
- 所有字段提供默认值以便增量构建状态
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.graph.state import GraphState


# ============================================================================
# 枚举定义
# ============================================================================


class QueryComplexity(str, Enum):
    """
    ★ Adaptive-RAG 查询复杂度分类
    ← Adaptive-RAG 论文 (NAACL 2024): 三元分类器输出
    ← WeKnora: QueryIntent (kb_search / web_search / greeting / ...)
               但我们这里的分类维度不同——不是"意图"而是"复杂度"
    """
    SIMPLE = "simple"       # 简单问题 → NoRetrievalStrategy → 直接 LLM 回答
    MEDIUM = "medium"       # 中等问题 → SingleStepStrategy → 单步检索+回答
    COMPLEX = "complex"     # 复杂问题 → MultiStepStrategy → 迭代检索+HyDE+改写


class RetrievalStrategy(str, Enum):
    """检索策略类型标识"""
    NO_RETRIEVAL = "no_retrieval"       # 不检索，直接 LLM 回答
    SINGLE_STEP = "single_step"          # 单步 BM25 + Dense + Rerank
    MULTI_STEP = "multi_step"            # 迭代检索 + HyDE + 查询改写
    ADAPTIVE = "adaptive"                # ★ 动态路由到上述三种策略


class MessageRole(str, Enum):
    """← WeKnora: message.go Message.Role ("user" / "assistant" / "system")"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MatchType(str, Enum):
    """← WeKnora: search.go MatchType (检索匹配类型)"""
    VECTOR = "vector"        # 向量语义匹配
    KEYWORD = "keyword"      # BM25 关键词匹配
    HYBRID = "hybrid"        # 混合匹配
    HYDE = "hyde"            # HyDE 假设文档匹配


class CircuitState(str, Enum):
    """
    ★ 三态熔断器状态机
    ← 原项目 B 双重熔断 + 标准断路器模式
    """
    CLOSED = "closed"            # 正常运行，请求正常通过
    OPEN = "open"                # 熔断触发，直接拒绝请求
    HALF_OPEN = "half_open"      # 半开探测，放行少量请求测试恢复


class MemoryType(str, Enum):
    """★ 三级记忆类型 ← 原项目 B"""
    SHORT_TERM = "short_term"      # 会话内上下文 (WeKnora: load_history.go)
    MEDIUM_TERM = "medium_term"    # 跨会话用户偏好
    LONG_TERM = "long_term"        # 持久化知识摘要 (WeKnora: memory/consolidator.go)


class SafetyLevel(str, Enum):
    """内容安全风险等级 ← 原项目 B HITL"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FallbackStrategy(str, Enum):
    """← WeKnora: session.go FallbackStrategy"""
    FIXED = "fixed"   # 返回固定回复
    MODEL = "model"   # 使用 LLM 生成兜底回复


# ============================================================================
# 文档与检索结果
# ============================================================================


class Document(BaseModel):
    """
    检索到的文档片段
    ← WeKnora: search.go SearchResult struct
       简化: 去掉 GORM 标签、去掉 ChunkMetadata、ImageInfo 等企业特性
       保留: 核心检索字段——ID、内容、来源、分数、匹配类型
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="文档片段文本内容")
    source: str = Field(default="", description="来源文档名称")
    source_path: str = Field(default="", description="来源文档路径")
    chunk_index: int = Field(default=0, description="在源文档中的分块序号")
    score: float = Field(default=0.0, description="检索相关性得分")
    match_type: MatchType = Field(default=MatchType.HYBRID, description="匹配类型")
    metadata: dict[str, str] = Field(default_factory=dict, description="附加元数据")

    def __str__(self) -> str:
        snippet = self.content[:100].replace("\n", " ")
        return f"[{self.match_type.value}:{self.score:.3f}] {snippet}..."


class SearchResult(BaseModel):
    """
    检索结果集合（一次检索可能返回多个 Document）
    ← WeKnora: search.go SearchResult[] — 在 WeKnora 中搜索本身返回切片，
       我们包装为单一结果对象以便 Pipeline 传递
    """
    query: str = Field(..., description="原始查询文本")
    documents: list[Document] = Field(default_factory=list)
    strategy: RetrievalStrategy = Field(default=RetrievalStrategy.SINGLE_STEP)
    total_found: int = Field(default=0)
    search_time_ms: float = Field(default=0.0, description="检索耗时(毫秒)")


# ============================================================================
# 查询相关
# ============================================================================


class QueryRequest(BaseModel):
    """
    用户查询请求
    ← WeKnora: chat_manage.go PipelineRequest (精简版)
    """
    query: str = Field(..., min_length=1, description="用户输入的查询文本")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    enable_memory: bool = Field(default=True, description="是否启用记忆增强")
    enable_guardrails: bool = Field(default=True, description="是否启用安全护栏")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewrittenQuery(BaseModel):
    """
    ★ 三级查询改写结果 ← 原项目 B
    ← WeKnora: chat_manage.go PipelineState.RewriteQuery (仅一级改写)
    """
    original: str = Field(..., description="原始查询")
    keyword_expanded: str = Field(default="", description="关键词扩展后的查询")
    semantic_rewritten: str = Field(default="", description="语义重写后的查询")
    hyde_hypothesis: str = Field(default="", description="HyDE 生成的假设文档")
    final_query: str = Field(default="", description="最终使用的查询文本")

    @field_validator("final_query", mode="before")
    @classmethod
    def set_final_query(cls, v: str, info: Any) -> str:
        if v:
            return v
        data = info.data
        return (data.get("hyde_hypothesis")
                or data.get("semantic_rewritten")
                or data.get("keyword_expanded")
                or data["original"])


# ============================================================================
# 对话消息与记忆
# ============================================================================


class Message(BaseModel):
    """
    会话消息
    ← WeKnora: message.go Message struct
       精简: 去掉 GORM hooks、IM channel、KnowledgeID、ImageInfo
       保留: ID、SessionID、Content、Role、References、Completed 等核心字段
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(default="")
    content: str = Field(..., description="消息文本内容")
    role: MessageRole = Field(default=MessageRole.USER)
    knowledge_refs: list[Document] = Field(default_factory=list, description="引用的知识片段")
    is_completed: bool = Field(default=False, description="生成是否完成")
    is_fallback: bool = Field(default=False, description="是否为兜底回复")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class History(BaseModel):
    """
    对话历史条目 (Query-Answer 对)
    ← WeKnora: message.go History struct
    """
    query: str
    answer: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    knowledge_refs: list[Document] = Field(default_factory=list)


class ConversationContext(BaseModel):
    """
    会话上下文 (短期记忆)
    ← WeKnora: chat_pipeline/load_history.go — 加载最近 N 轮对话
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = Field(default_factory=list)
    history: list[History] = Field(default_factory=list)
    max_rounds: int = Field(default=10)
    current_round: int = Field(default=0)


# ============================================================================
# 记忆系统
# ============================================================================


class MemoryEntry(BaseModel):
    """
    ★ 记忆条目 (三级记忆系统的统一数据模型)
    ← 原项目 B 特性
    ← WeKnora: agent/memory/consolidator.go — 仅 LLM 摘要 (长期记忆)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType
    content: str = Field(..., description="记忆内容")
    embedding: Optional[list[float]] = Field(default=None, description="向量嵌入")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性评分")
    access_count: int = Field(default=0, description="访问次数")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = Field(default=None)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserPreference(BaseModel):
    """
    ★ 用户偏好 (中期记忆)
    ← 原项目 B 特性
    WeKnora 无此概念——这是跨会话偏好学习的核心数据结构
    """
    user_id: str = Field(default="default")
    preferred_topics: list[str] = Field(default_factory=list)
    preferred_style: str = Field(default="detailed", description="回答风格偏好")
    frequently_asked: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# Agent 执行步骤 (← WeKnora: agent.go)
# ============================================================================


class ToolCall(BaseModel):
    """
    工具调用记录
    ← WeKnora: agent.go ToolCall struct
       简化: 去掉 ProviderMetadata (provider 特定元数据)
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="工具名称")
    args: dict[str, Any] = Field(default_factory=dict, description="调用参数")
    result: Optional[dict[str, Any]] = Field(default=None, description="执行结果")
    error: Optional[str] = Field(default=None, description="错误信息")
    duration_ms: int = Field(default=0, description="执行耗时(毫秒)")


class AgentStep(BaseModel):
    """
    ★ ReAct 循环中的单步执行记录
    ← WeKnora: agent.go AgentStep struct
       保留: iteration, thought, tool_calls, timestamp
       去掉: ReasoningContent (thinking mode 专用)
    """
    iteration: int = Field(default=0, description="迭代序号")
    thought: str = Field(default="", description="LLM 推理/思考内容")
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observation: str = Field(default="", description="观察结果摘要")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# LangGraph 工作流状态 (★ 核心)
# ============================================================================


class AgentState(BaseModel):
    """
    ★ LangGraph StateGraph 的核心状态对象
    ← WeKnora: agent.go AgentState struct (精简+扩展)
       WeKnora 的 AgentState 只有 5 个字段，我们的基于 LangGraph 扩展了很多
       因为 LangGraph 的状态驱动模式需要状态对象承担更多职责

    此对象在 LangGraph 的每个节点之间流转，每个节点读取/写入字段。
    使用 Pydantic 提供的类型安全和默认值。
    """
    # ---- 输入 ----
    query: str = Field(default="", description="用户原始查询")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # ---- Adaptive-RAG 路由 ----
    complexity: QueryComplexity = Field(default=QueryComplexity.MEDIUM)
    complexity_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    selected_strategy: RetrievalStrategy = Field(default=RetrievalStrategy.SINGLE_STEP)
    classification_reasoning: str = Field(default="", description="分类理由 (从 GraphState 映射)")

    # ---- 查询改写 ----
    rewritten_query: Optional[RewrittenQuery] = Field(default=None)
    hyde_hypothesis: str = Field(default="")

    # ---- 检索结果 ----
    search_result: Optional[SearchResult] = Field(default=None)
    retrieved_docs: list[Document] = Field(default_factory=list)

    # ---- 生成 ----
    generated_answer: str = Field(default="")
    generated_answer_stream: Optional[Any] = Field(default=None, exclude=True)

    # ---- 审核 ----
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    quality_passed: bool = Field(default=True)
    safety_level: SafetyLevel = Field(default=SafetyLevel.LOW)
    needs_human_review: bool = Field(default=False)

    # ---- 熔断器 ----
    circuit_state: CircuitState = Field(default=CircuitState.CLOSED)
    circuit_quality_failures: int = Field(default=0)
    circuit_quality_total: int = Field(default=0)

    # ---- Agent 步骤追踪 ----
    agent_steps: list[AgentStep] = Field(default_factory=list)
    current_iteration: int = Field(default=0)
    max_iterations: int = Field(default=10)  # ← WeKnora: agent_service.go MAX_ITERATIONS=100, 我们减到10

    # ---- 记忆 ----
    conversation_context: Optional[ConversationContext] = Field(default=None)
    relevant_memories: list[MemoryEntry] = Field(default_factory=list)
    user_preference: Optional[UserPreference] = Field(default=None)

    # ---- Token 追踪 ----
    total_tokens_used: int = Field(default=0)
    token_budget: int = Field(default=4000)

    # ---- 元信息 ----
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed: bool = Field(default=False)
    error: Optional[str] = Field(default=None)

    # ---- 评估用 ----
    ground_truth: Optional[str] = Field(default=None, description="评估模式下的标准答案")
    ragas_scores: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ----------------------------------------------------------------
    # ★ 从 GraphState 构造 AgentState (方案 B — 单一映射入口)
    # ----------------------------------------------------------------

    @classmethod
    def from_graph_state(cls, state: dict) -> "AgentState":
        """
        从 LangGraph GraphState (TypedDict) 构造 AgentState (Pydantic)。

        自动映射同名字段并处理类型转换 (str→Enum 等)。
        所有 GraphState 中不存在的字段使用 AgentState 默认值。

        面试可讲:
        "LangGraph 使用 TypedDict 做节点间状态流转，
        但检索策略层需要 Pydantic 的验证和类型安全。
        我通过 from_graph_state 工厂方法实现两者的无损桥接，
        避免字段丢失和手动逐字段复制。"
        """
        # 直接映射的同名字段 (str/bool/int/float 类型兼容)
        direct_fields = (
            "query", "session_id",
            "complexity_confidence",
            "hyde_hypothesis",
            "generated_answer",
            "quality_score", "quality_passed",
            "needs_human_review",
            "current_iteration",
            "total_tokens_used", "token_budget",
            "ground_truth",
            "completed", "error",
            "classification_reasoning",
        )
        kwargs: dict[str, Any] = {
            f: state[f] for f in direct_fields if f in state and state[f] is not None
        }

        # 类型转换: str → Enum
        if "complexity" in state:
            raw = state["complexity"]
            kwargs["complexity"] = QueryComplexity(raw) if isinstance(raw, str) else raw
        if "selected_strategy" in state:
            raw = state["selected_strategy"]
            kwargs["selected_strategy"] = RetrievalStrategy(raw) if isinstance(raw, str) else raw
        if "safety_risk_level" in state:
            raw = state["safety_risk_level"]
            kwargs["safety_level"] = SafetyLevel(raw) if isinstance(raw, str) else raw

        # 检索结果 (list 直接传递)
        if "retrieved_docs" in state:
            kwargs["retrieved_docs"] = state["retrieved_docs"]

        logger = __import__("logging").getLogger(__name__)
        logger.debug(
            "AgentState.from_graph_state: mapped %d fields (graph has %d)",
            len(kwargs), len(state),
        )
        return cls(**kwargs)


# ============================================================================
# 模型配置 (← WeKnora: internal/models/ 精简)
# ============================================================================


class LLMConfig(BaseModel):
    """LLM 配置 ← WeKnora: models/chat/ ChatConfig"""
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-4o-mini")
    api_key: str = Field(default="")
    base_url: str = Field(default="https://api.openai.com/v1")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096)
    timeout: int = Field(default=120)
    max_retries: int = Field(default=3)


class EmbeddingConfig(BaseModel):
    """Embedding 模型配置（独立于 LLM 配置）"""
    provider: str = Field(default="local")
    model_name: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    dimensions: int = Field(default=1536)
    base_url: str = Field(default="https://api.openai.com/v1", description="仅 provider=openai 时使用")
    api_key: str = Field(default="", description="为空时自动复用 LLM_API_KEY")


# ============================================================================
# 熔断器配置
# ============================================================================


class CircuitBreakerConfig(BaseModel):
    """
    ★ 三态熔断器配置
    ← 原项目 B 双重熔断 + 标准断路器模式
    """
    # 质量熔断 (滑动窗口)
    quality_window_size: int = Field(default=20)
    quality_failure_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    quality_timeout_seconds: int = Field(default=60)

    # 频率熔断 (令牌桶)
    freq_max_requests: int = Field(default=20)
    freq_refill_rate: int = Field(default=20)


# ============================================================================
# 评估 (← RAGAS)
# ============================================================================


class CompareResult(BaseModel):
    """
    ★ RAGAS 三路对比评估结果
    ← 本项目设计: 同查询跑 3 条路径 → 输出对比报告
    """
    query: str
    direct_answer: dict[str, Any] = Field(default_factory=dict)     # 直接回答
    standard_rag: dict[str, Any] = Field(default_factory=dict)      # 标准 RAG
    adaptive_rag: dict[str, Any] = Field(default_factory=dict)      # 自适应 RAG
    conclusion: str = Field(default="")
    winner: str = Field(default="")


# ============================================================================
# API 响应
# ============================================================================


class APIResponse(BaseModel):
    """统一 API 响应格式"""
    success: bool = True
    data: Any = None
    error: Optional[str] = None
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tokens_used: int = 0
    duration_ms: float = 0.0
