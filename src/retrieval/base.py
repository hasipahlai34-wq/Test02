"""
# ============================================================
# ★ 检索策略抽象基类 (策略模式)
# ← 本项目设计: 4 种策略可插拔，运行时动态切换
#
#   WeKnora 的检索方式是固定的 Pipeline:
#   CHUNK_SEARCH → CHUNK_RERANK → CHUNK_MERGE → INTO_CHAT_MESSAGE
#   没有策略选择概念，始终使用同样的检索流程
#
#   我们通过策略模式实现:
#   - 简单查询 → NoRetrieval (直接 LLM 回答)
#   - 中等查询 → SingleStep (BM25 + Dense + Rerank)
#   - 复杂查询 → MultiStep (迭代检索 + HyDE + 改写)
#   - 自适应   → Adaptive (分类器 → 动态委托)
# ============================================================

本模块定义了检索策略的统一接口，
所有具体策略必须继承 RetrievalStrategy 并实现 retrieve() 方法。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from src.types import (
    AgentState,
    Document,
    RetrievalStrategy as StrategyType,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """
    ★ 检索策略抽象基类
    定义统一接口: retrieve(query, state) → SearchResult

    面试可讲: "我用策略模式让检索策略可插拔，
    AdaptiveStrategy 运行时会根据查询复杂度动态选择合适的策略。"
    """

    def __init__(self, name: str):
        self.name = name
        self.strategy_type = StrategyType.SINGLE_STEP  # 子类覆盖

    @abstractmethod
    async def retrieve(self, query: str, state: AgentState, **kwargs) -> SearchResult:
        """
        执行检索

        Args:
            query: 用户查询文本 (已改写后的)
            state: LangGraph AgentState (包含会话上下文和元信息)

        Returns:
            SearchResult: 包含检索到的文档列表和元信息

        Raises:
            NotImplementedError: 子类必须实现
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ================================================================
# 策略注册表 (用于 AdaptiveStrategy 的动态路由)
# ================================================================


class StrategyRegistry:
    """
    策略注册表
    AdaptiveStrategy 通过此注册表找到对应复杂度的策略实例
    """

    def __init__(self):
        self._strategies: dict[str, RetrievalStrategy] = {}

    def register(self, strategy: RetrievalStrategy) -> None:
        """注册策略"""
        self._strategies[strategy.strategy_type.value] = strategy
        logger.debug("注册检索策略: %s → %s", strategy.strategy_type.value, strategy.name)

    def get(self, strategy_type: str) -> Optional[RetrievalStrategy]:
        """获取策略"""
        return self._strategies.get(strategy_type)

    def list_all(self) -> list[str]:
        """列出所有已注册的策略类型"""
        return list(self._strategies)

    def clear(self) -> None:
        """清空所有注册"""
        self._strategies.clear()
