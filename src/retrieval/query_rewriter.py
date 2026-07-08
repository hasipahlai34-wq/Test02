"""
# ============================================================
# ★ 三级查询改写
# ← 原项目 B 特性
# ← WeKnora: chat_pipeline/query_understand.go (仅一级改写)
#           我们扩展为三级递进式:
#             Level 1: 关键词扩展 (同义词、相关术语)
#             Level 2: 语义重写 (指代消解、省略补全)
#             Level 3: HyDE 假设文档 (见 hyde.py)
# ============================================================

本模块实现了查询改写的完整流水线:
1. 关键词扩展 — 补充同义词和相关术语增加召回
2. 语义重写 — 将口语化查询改写为规范完整的检索语句

设计要点:
- 三级递进式改写，后一级以前一级为基础
- 失败自动降级 (LLM 调用失败则使用前一级结果)
- 改写后的查询更适合向量检索
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from config.settings import get_settings
from src.models.llm import LLMClient
from src.types import RewrittenQuery
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    三级查询改写器
    ← WeKnora: query_understand.go — LLM 改写 (仅一级)
               我们扩展为三级递进式

    面试可讲:
    "查询改写是 RAG 系统的重要前置步骤。用户输入往往存在三个问题:
    1) 用词不精确 (口语 vs 术语);
    2) 指代不明确 ('它'、'这个');
    3) 语义碎片化。
    我设计了三级递进式改写管道来解决这三个问题。"
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm = llm_client or LLMClient()

    # ----------------------------------------------------------------
    # Level 1: 关键词扩展
    # ----------------------------------------------------------------

    async def expand_keywords(
        self,
        query: str,
        conversation: str = "",
        language: str = "中文",
    ) -> str:
        """
        关键词扩展: 补充同义词和相关术语
        ← WeKnora 无此功能，我们新增

        例如: "营收增长" → "营收增长 收入增加 业绩提升 同比增长"

        Args:
            query: 原始查询
            conversation: 对话历史 (可选)
            language: 目标语言

        Returns:
            扩展后的关键词字符串
        """
        try:
            result = await self._llm.ask(
                prompt=load_prompt(
                    "keyword_expansion",
                    filename="rewrite",
                    query=query,
                    conversation=conversation,
                    language=language,
                ),
                model_name=get_settings().llm_simple_model,
            )
            expanded = result.strip()
            logger.debug("关键词扩展: '%s...' → '%s...'", query[:40], expanded[:40])
            return expanded if expanded else query

        except Exception as e:
            logger.warning("关键词扩展失败: %s，使用原始查询", e)
            return query

    # ----------------------------------------------------------------
    # Level 2: 语义重写 (← WeKnora: query_understand.go)
    # ----------------------------------------------------------------

    async def rewrite_semantic(
        self,
        query: str,
        conversation: str = "",
        language: str = "中文",
    ) -> str:
        """
        语义重写: 指代消解 + 省略补全
        ← WeKnora: query_understand.go → LLM 改写
           一模一样的核心逻辑: 将用户的追问改为独立完整的查询

        例如:
          对话历史: "2024年Q3营收是多少？" → "Q3营收为50亿元"
          当前查询: "增长了多少？"
          改写结果: "2024年Q3营收相比上一季度增长了多少"

        Args:
            query: 用户当前查询
            conversation: 对话历史上下文
            language: 目标语言

        Returns:
            改写后的完整独立查询
        """
        try:
            result = await self._llm.ask(
                prompt=load_prompt(
                    "semantic_rewrite",
                    filename="rewrite",
                    query=query,
                    conversation=conversation,
                    language=language,
                ),
                model_name=get_settings().llm_simple_model,
            )
            rewritten = result.strip()
            logger.info("语义重写: '%s...' → '%s...'", query[:40], rewritten[:40])
            return rewritten if rewritten else query

        except Exception as e:
            logger.warning("语义重写失败: %s，使用原始查询", e)
            return query

    # ----------------------------------------------------------------
    # 完整三级改写 Pipe
    # ----------------------------------------------------------------

    async def rewrite(
        self,
        query: str,
        conversation: str = "",
        enable_hyde: bool = False,
        hyde_generator=None,
        language: str = "中文",
    ) -> RewrittenQuery:
        """
        执行完整的三级查询改写管道

        Pipe: 原始查询 → 关键词扩展 → 语义重写 → (可选) HyDE

        Args:
            query: 用户原始查询
            conversation: 对话历史
            enable_hyde: 是否启用 HyDE 假设文档
            hyde_generator: HyDEGenerator 实例
            language: 目标语言

        Returns:
            RewrittenQuery: 包含每一级改写结果的完整对象
        """
        result = RewrittenQuery(original=query)

        # Level 1: 关键词扩展
        result.keyword_expanded = await self.expand_keywords(
            query, conversation, language,
        )

        # Level 2: 语义重写 (基于关键词扩展后的结果)
        semantic_input = result.keyword_expanded if result.keyword_expanded != query else query
        result.semantic_rewritten = await self.rewrite_semantic(
            semantic_input, conversation, language,
        )

        # 决定最终查询: 语义重写 > 关键词扩展 > 原始
        result.final_query = (
            result.semantic_rewritten or
            result.keyword_expanded or
            result.original
        )

        # Level 3: HyDE (可选，耗时较长，仅复杂查询启用)
        if enable_hyde and hyde_generator:
            try:
                result.hyde_hypothesis = await hyde_generator.generate(
                    result.final_query, language,
                )
                # HyDE 假设文档作为最终查询
                if result.hyde_hypothesis:
                    result.final_query = result.hyde_hypothesis
            except Exception as e:
                logger.warning("HyDE 生成失败: %s", e)

        logger.info(
            "三级改写完成: '%s...' → '%s...'",
            query[:40], result.final_query[:40],
        )
        return result
