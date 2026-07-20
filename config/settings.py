"""
# ============================================================
# 全局配置管理 (pydantic-settings)
# ← WeKnora: config/config.yaml + 环境变量读取
#   WeKnora 使用 Viper + YAML，我们使用 pydantic-settings + .env
#   两者都是配置管理模式，pydantic-settings 更 Pythonic
# ============================================================

配置优先级: .env 文件 > 环境变量 > 默认值

所有配置项集中于此，通过 `get_settings()` 获取全局单例。
使用 lru_cache 确保整个应用生命周期内只加载一次配置。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录 (adaptive-rag/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    全局配置 ← WeKnora: config/config.yaml + 环境变量
    使用 pydantic-settings 的 BaseSettings，自动从 .env 和环境变量读取
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",       # 忽略 .env 中未定义的多余字段
        case_sensitive=False, # 环境变量名大小写不敏感
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Project-local .env should win over machine-level environment
        # variables. This prevents stale global EMBEDDING_API_KEY/RERANK_API_KEY
        # values from silently overriding the key configured for this project.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # ================================================================
    # LLM 配置 (OpenAI 兼容协议)
    # ← WeKnora: config/config.yaml conversation 段
    #
    # 同时支持新旧环境变量名:
    #   LLM_API_KEY      ← 新推荐名 (provider-neutral)
    #   OPENAI_API_KEY   ← 旧名 (向后兼容)
    #   LLM_BASE_URL     ← 新推荐名
    #   OPENAI_BASE_URL  ← 旧名
    # ================================================================
    llm_api_key: str = Field(
        default="sk-placeholder",
        validation_alias=AliasChoices("llm_api_key", "openai_api_key"),
        description="LLM API Key（也支持旧名 OPENAI_API_KEY）",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("llm_base_url", "openai_base_url"),
        description="LLM Base URL（也支持旧名 OPENAI_BASE_URL）",
    )
    llm_default_model: str = Field(
        default="gpt-4o-mini",
        description="默认 LLM 模型",
    )
    llm_simple_model: str = Field(
        default="gpt-4o-mini",
        description="简单查询模型 (可设为与 llm_default_model 相同以节省配置)",
    )
    llm_medium_model: str = Field(
        default="gpt-4o-mini",
        description="中等复杂度模型 (可设为与 llm_default_model 相同以节省配置)",
    )
    llm_complex_model: str = Field(
        default="gpt-4o",
        description="复杂多步推理模型",
    )
    safety_model: str = Field(
        default="",
        description="安全检测专用模型（空字符串 = 复用 llm_default_model）",
    )
    safety_base_url: str = Field(
        default="",
        description="安全检测专用 Base URL（空字符串 = 复用 llm_base_url）",
    )
    safety_api_key: str = Field(
        default="",
        description="安全检测专用 API Key（空字符串 = 复用 llm_api_key）",
    )

    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    llm_timeout: int = 120
    llm_max_retries: int = 2

    # ---- 向后兼容属性 (旧代码可能引用 openai_* 字段名) ----
    @property
    def openai_api_key(self) -> str:
        """@deprecated: 请使用 llm_api_key"""
        return self.llm_api_key

    @property
    def openai_base_url(self) -> str:
        """@deprecated: 请使用 llm_base_url"""
        return self.llm_base_url

    # ================================================================
    # Embedding 配置 ← WeKnora: config/config.yaml embedding 段
    #
    # provider=local:  使用本地 sentence-transformers 模型（推荐）
    #                  首次运行自动下载 ~80MB，之后使用缓存，零 API 费用
    # provider=openai: 使用 OpenAI 兼容 Embedding API
    #                  需要配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY
    # ================================================================
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 10

    # 仅当 EMBEDDING_PROVIDER=openai 时使用以下配置
    # embedding_api_key 为空时自动回退使用 LLM_API_KEY
    embedding_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="Embedding API Base URL（独立于 LLM_BASE_URL，防止 Chat 和 Embedding 混用）",
    )
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("embedding_api_key", "dashscope_api_key"),
        description="Embedding API Key（为空时自动复用 LLM_API_KEY）",
    )
    dashscope_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("dashscope_api_key"),
        description="DashScope API Key shared by embedding and rerank.",
    )

    # ================================================================
    # Rerank 重排序 ← WeKnora: chat_pipeline/rerank.go
    # ================================================================
    rerank_enabled: bool = True
    rerank_provider: str = "dashscope"
    rerank_model: str = "qwen3-rerank"
    rerank_base_url: str = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    rerank_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("rerank_api_key", "dashscope_api_key"),
        description="Rerank API Key. Falls back to EMBEDDING_API_KEY when empty.",
    )
    rerank_candidate_top_k: int = 20
    rerank_top_k: int = 6
    rerank_threshold: float = 0.0
    rerank_instruct: str = "Given a web search query, retrieve relevant passages that answer the query."
    rerank_timeout: int = 30

    # ================================================================
    # 向量数据库 ← WeKnora: 多向量库后端 → ChromaDB
    # ================================================================
    chroma_persist_dir: str = str(PROJECT_ROOT / "data" / "chroma")
    chroma_collection_name: str = "adaptive_rag_docs"

    # ================================================================
    # 检索参数 ← WeKnora: chat_manage.go 中的 VectorThreshold 等
    # ================================================================
    retrieval_top_k: int = 10
    retrieval_threshold: float = 0.0
    bm25_top_k: int = 10
    dense_top_k: int = 40

    # ================================================================
    # Document chunking
    # ================================================================
    # 目标 chunk 大小（字符数）
    chunk_size: int = Field(default=800)
    # chunk 重叠大小
    chunk_overlap: int = Field(default=100)
    # 最小 chunk 大小（小于此值合并到上一个 chunk）
    chunk_min_size: int = Field(default=100)
    # 是否启用结构感知分块
    chunk_structure_aware: bool = Field(default=True)
    # 结构分块时标题识别的最小字体差（pt）
    chunk_heading_font_delta: int = Field(default=2)
    # 分隔符优先级（中文优化）
    chunk_separators: list[str] = Field(
        default=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    # ================================================================
    # 上下文窗口管理 ← WeKnora: token/compress.go
    # ================================================================
    context_max_tokens: int = 8000
    context_compress_threshold: float = 0.5

    # ================================================================
    # 熔断器配置 ← 原项目 B + 标准断路器模式
    # ================================================================
    cb_quality_window_size: int = 20
    cb_quality_failure_threshold: float = 0.3
    cb_quality_timeout_seconds: int = 60
    cb_freq_max_requests: int = 20
    cb_freq_refill_rate: int = 20

    # ================================================================
    # 记忆系统 ← 原项目 B
    # ================================================================
    memory_short_term_max_rounds: int = 10
    memory_medium_term_db: str = str(PROJECT_ROOT / "data" / "sqlite" / "memory.db")
    ltm_chroma_dir: str = str(PROJECT_ROOT / "data" / "ltm_chroma")

    # ================================================================
    # RAGAS 在线评估 ← 新增: 每次查询后自动评估
    # ================================================================
    ragas_online_enabled: bool = True
    ragas_faithfulness_threshold: float = 0.6
    ragas_relevancy_threshold: float = 0.5
    ragas_context_precision_threshold: float = 0.5
    ragas_eval_model: str = "deepseek-v4-pro"
    ragas_eval_base_url: str = "https://api.deepseek.com/v1"
    ragas_eval_api_key: str = ""

    # ================================================================
    # HITL 人机协同审核 ← 新增: 高风险回答暂停等待人工确认
    # ================================================================
    hitl_enabled: bool = True
    hitl_interrupt_timeout_seconds: int = 1800  # 30 分钟
    hitl_queue_dir: str = str(PROJECT_ROOT / "data" / "hitl_queue")
    hitl_results_dir: str = str(PROJECT_ROOT / "data" / "hitl_results")

    # ================================================================
    # LangSmith 可观测 ← WeKnora: tracing/langfuse/ → LangSmith
    # ================================================================
    langsmith_api_key: str = ""
    langsmith_project: str = "adaptive-rag"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ================================================================
    # Langfuse 可观测性
    # ================================================================
    langfuse_enabled: bool = True
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = Field(
        default="http://localhost:3000",
        validation_alias=AliasChoices("langfuse_base_url", "langfuse_host"),
        description="Langfuse base URL / host",
    )

    # ================================================================
    # 服务配置
    # ================================================================
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    default_language: str = "中文"

    # ================================================================
    # 项目路径
    # ================================================================
    project_root: Path = PROJECT_ROOT
    prompts_dir: Path = PROJECT_ROOT / "config" / "prompts"
    samples_dir: Path = PROJECT_ROOT / "samples"
    data_dir: Path = PROJECT_ROOT / "data"

    # ================================================================
    # ★ 优化开关（可独立回滚，默认全部开启）
    # ================================================================
    opt_classify_rules_expanded: bool = Field(
        default=True,
        description="分类规则扩展：扩大正则覆盖以减少 LLM 分类调用",
    )
    opt_input_safety_prescreen: bool = Field(
        default=True,
        description="输入安全预筛：正则预筛安全查询，跳过 LLM 安全检测",
    )
    opt_output_safety_domain_skip: bool = Field(
        default=True,
        description="输出安全域跳过：文档 QA 领域查询跳过 LLM 输出安全检测",
    )
    opt_review_fast_path: bool = Field(
        default=True,
        description="审核快速通道：高置信度 medium 查询跳过 LLM 质量审核",
    )
    opt_sufficiency_heuristic: bool = Field(
        default=True,
        description="充分性启发式预检：首轮检索用规则判断是否充分",
    )


@lru_cache
def get_settings() -> Settings:
    """
    获取全局配置单例 (带缓存)
    使用 lru_cache 确保整个应用生命周期内只加载一次
    """
    return Settings()
