"""
# ============================================================
# 无检索策略 (NoRetrievalStrategy)
# ← Adaptive-RAG 论文: 简单查询不检索，直接 LLM 回答
#
#   适用场景:
#   - 常识性问题 ("什么是AI")
#   - 简单闲聊 ("你好")
#   - 翻译任务
#   - 不需要文档就能回答的问题
#
#   设计目的:
#   - 节省 Token 成本 (不需要检索 + 不需要上下文)
#   - 提高响应速度 (跳过整个检索管道)
#   - 减少检索噪音 (不相关文档反而干扰 LLM)
# ============================================================
"""

from __future__ import annotations

import time
import logging
from typing import Any

from src.retrieval.base import RetrievalStrategy
from src.types import AgentState, Document, SearchResult, RetrievalStrategy as StrategyType

logger = logging.getLogger(__name__)


class NoRetrievalStrategy(RetrievalStrategy):
    """
    无检索策略 — 直接返回空结果，让 LLM 凭自身知识回答
    ← Adaptive-RAG 论文: simple 查询路由到此策略
    """

    def __init__(self):
        super().__init__(name="无检索 (Direct Answer)")
        self.strategy_type = StrategyType.NO_RETRIEVAL

    async def retrieve(self, query: str, state: AgentState, **kwargs) -> SearchResult:
        """
        不检索任何文档，直接返回空结果

        面试可讲:
        "对于被分类器判定为 simple 的查询，我们跳过检索，
        直接让 LLM 凭常识回答。这样做有两个好处:
        1. 节省成本: 不需要调用 Embedding + 向量检索 + Rerank
        2. 避免噪音: 简单的常识问题不需要外部文档，
           引入不相关的检索结果反而会干扰 LLM 的判断"
        """
        logger.info("NoRetrieval: 跳过检索 (query=%s...)", query[:80])

        return SearchResult(
            query=query,
            documents=[],
            strategy=self.strategy_type,
            total_found=0,
            search_time_ms=0.0,
        )
