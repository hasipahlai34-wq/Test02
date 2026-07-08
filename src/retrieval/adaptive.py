"""
# ============================================================
# ★ 自适应检索策略 (AdaptiveStrategy)
# ← Adaptive-RAG 论文 (NAACL 2024): 核心创新
#
# 核心思想:
#   不是所有查询都需要同样复杂的检索策略。
#   - 简单问题 (什么是AI) → 直接 LLM 回答即可，不需要检索
#   - 中等问题 (Q3营收多少) → 单步检索即可找到答案
#   - 复杂问题 (分析增长驱动因素) → 需要多步迭代检索+HyDE+改写
#
#   用一个小的分类器判断查询复杂度，然后动态路由到对应策略。
#   这就是 "Adaptive" 的含义——检索策略自适应查询的复杂程度。
#
# ← WeKnora: 没有此概念，始终使用固定检索管道
# ============================================================

本模块是 Adaptive-RAG 的核心:
- LLM 对查询进行复杂度三元分类 (simple / medium / complex)
- 根据分类结果动态委托给对应策略
- 策略注册表管理所有可用策略
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from src.retrieval.base import RetrievalStrategy, StrategyRegistry
from src.types import (
    AgentState,
    Document,
    QueryComplexity,
    RetrievalStrategy as StrategyType,
    SearchResult,
)
from config.settings import get_settings
from src.models.llm import LLMClient
from src.utils.json_parser import parse_llm_json_response
from src.utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


class AdaptiveStrategy(RetrievalStrategy):
    """
    ★ 自适应检索策略 — Adaptive-RAG 核心
    查询复杂度分类 → 动态委托到对应策略

    面试可讲 (这是 12 大核心亮点 #1):
    "基于 NAACL 2024 的 Adaptive-RAG 论文，我实现了一个查询复杂度分类器。
    它将用户查询分为 simple/medium/complex 三类:
    - simple → 不检索，直接 LLM 回答 (节省成本)
    - medium → 单步 BM25+Dense+Rerank (标准 RAG)
    - complex → 多步迭代检索+HyDE+查询改写 (深度检索)
    分类器是一个轻量 LLM 调用 (用 gpt-4o-mini)，耗时约 200ms，
    但能显著优化后续管道——避免简单问题走复杂流程造成的浪费。"
    """

    def __init__(
        self,
        registry: Optional[StrategyRegistry] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(name="自适应检索 (Adaptive-RAG)")
        self.strategy_type = StrategyType.ADAPTIVE
        self._registry = registry or StrategyRegistry()
        self._llm = llm_client or LLMClient()

    # ----------------------------------------------------------------
    # 查询复杂度分类 (★ Adaptive-RAG 核心)
    # ----------------------------------------------------------------

    async def classify(
        self,
        query: str,
        conversation: str = "",
    ) -> tuple[QueryComplexity, float, str]:
        """
        使用 LLM 对查询进行复杂度三元分类
        ← Adaptive-RAG 论文 Section 3.1: Query Complexity Classifier

        Args:
            query: 用户查询
            conversation: 对话历史 (可选)

        Returns:
            (复杂度, 置信度, 判断理由)
        """
        from src.graph.router import (
            is_aggregate_query,
            is_complex_diagnostic_query,
            is_implicit_inference_query,
            is_list_aggregation_query,
            is_single_fact_query,
        )

        if is_aggregate_query(query):
            return QueryComplexity.MEDIUM, 1.0, "aggregate_rule"
        if is_implicit_inference_query(query):
            return QueryComplexity.COMPLEX, 1.0, "implicit_inference_rule"
        if is_list_aggregation_query(query):
            return QueryComplexity.MEDIUM, 1.0, "list_aggregation_rule"
        if is_single_fact_query(query):
            return QueryComplexity.MEDIUM, 1.0, "single_fact_rule"
        if is_complex_diagnostic_query(query):
            return QueryComplexity.COMPLEX, 1.0, "diagnostic_rule"

        prompt = load_prompt(
            "complexity_classifier",
            filename="classification",
            query=query,
            conversation=conversation,
            language=get_settings().default_language,
        )

        try:
            response = await self._llm.ask(
                prompt=prompt,
                model_name=get_settings().llm_simple_model,  # 分类器用便宜模型
            )

            result = parse_llm_json_response(response)
            complexity_str = result.get("complexity", "medium")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            complexity = QueryComplexity(complexity_str)
            logger.info(
                "查询分类: complexity=%s confidence=%.2f reason='%s'",
                complexity.value, confidence, reasoning,
            )
            return complexity, confidence, reasoning

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("查询分类 JSON 解析失败: %s，默认 medium", e)
            return QueryComplexity.MEDIUM, 0.5, f"JSON解析失败: {e}"
        except (ConnectionError, TimeoutError) as e:
            # 网络/LLM 服务不可用 → 降级（预期内的故障）
            logger.warning("查询分类网络异常: %s，默认 medium", e)
            return QueryComplexity.MEDIUM, 0.5, f"网络异常降级: {e}"
        except Exception as e:
            # 未预期的代码逻辑错误 → 记录 critical 并降级
            logger.critical("查询分类未预期异常: %s", e, exc_info=True)
            return QueryComplexity.MEDIUM, 0.5, f"内部错误: {type(e).__name__}"

    # ----------------------------------------------------------------
    # 动态路由
    # ----------------------------------------------------------------

    def _route(self, complexity: QueryComplexity) -> RetrievalStrategy:
        """
        根据复杂度路由到对应策略

        Args:
            complexity: 查询复杂度

        Returns:
            对应策略实例

        Raises:
            ValueError: 未找到对应策略
        """
        route_map = {
            QueryComplexity.SIMPLE: StrategyType.NO_RETRIEVAL,
            QueryComplexity.MEDIUM: StrategyType.SINGLE_STEP,
            QueryComplexity.COMPLEX: StrategyType.MULTI_STEP,
        }

        target_type = route_map.get(complexity)
        if target_type is None:
            raise ValueError(f"未知的复杂度类型: {complexity}")

        strategy = self._registry.get(target_type.value)
        if strategy is None:
            raise ValueError(
                f"未注册的策略: {target_type.value}。"
                f"请先注册 NoRetrieval / SingleStep / MultiStep 策略。"
            )

        logger.info(
            "自适应路由: %s → %s",
            complexity.value, strategy.name,
        )
        return strategy

    # ----------------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------------

    async def retrieve(self, query: str, state: AgentState, **kwargs) -> SearchResult:
        """
        自适应检索主入口:
        1. 分类查询复杂度 (如 Graph 路由器已分类则复用)
        2. 更新 state 中的分类信息
        3. 动态委托给对应策略
        """
        start_time = time.perf_counter()

        # Step 1: 分类 (★ C4 修复: Graph 路由器已分类则复用，避免重复 LLM 调用)
        if state.classification_reasoning:
            complexity = state.complexity
            confidence = state.complexity_confidence
            reasoning = state.classification_reasoning
            logger.info("复用上游分类: %s (conf=%.2f)", complexity.value, confidence)
        else:
            complexity, confidence, reasoning = await self.classify(query)
            state.complexity = complexity
            state.complexity_confidence = confidence
            state.classification_reasoning = reasoning

        if kwargs.get("retrieval_filter") and complexity == QueryComplexity.SIMPLE:
            logger.info("active retrieval_filter present; upgrading simple query to medium retrieval")
            complexity = QueryComplexity.MEDIUM
            state.complexity = complexity

        # Step 2: 路由到具体策略
        strategy = self._route(complexity)
        state.selected_strategy = strategy.strategy_type

        from src.graph.router import is_list_aggregation_query, is_low_cost_retrieval_query

        if (
            strategy.strategy_type == StrategyType.SINGLE_STEP
            and is_low_cost_retrieval_query(query)
            and "top_k" not in kwargs
        ):
            kwargs["top_k"] = 5 if is_list_aggregation_query(query) else 3

        # Step 4: 执行检索
        result = await strategy.retrieve(query, state, **kwargs)

        # 保留自适应策略的元信息
        from src.graph.router import is_aggregate_query
        from src.retrieval.single_step import calculate_markdown_table_aggregation

        if is_aggregate_query(query):
            agg_text = calculate_markdown_table_aggregation(
                query,
                [doc.content for doc in result.documents],
            )
            if agg_text:
                result.documents.insert(0, Document(
                    content=agg_text,
                    score=1.0,
                    source="deterministic_table_aggregation",
                    metadata={
                        "aggregate_result": "true",
                        "query_type": "markdown_table_aggregation",
                    },
                ))
                result.total_found = len(result.documents)

        result.strategy = self.strategy_type

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Adaptive 完成: %s (conf=%.2f) → %s → %d docs (%.0fms)",
            complexity.value, confidence, strategy.name,
            len(result.documents), elapsed_ms,
        )

        return result


# ================================================================
# 便捷工厂函数
# ================================================================


def create_adaptive_chain(
    llm_client: Optional[LLMClient] = None,
) -> tuple[AdaptiveStrategy, StrategyRegistry]:
    """
    创建完整的自适应检索链:
    注册 NoRetrieval + SingleStep + MultiStep → 返回 AdaptiveStrategy

    用法:
        adaptive, registry = create_adaptive_chain()
        result = await adaptive.retrieve(query, state)

    Returns:
        (AdaptiveStrategy 实例, StrategyRegistry 实例)
    """
    from src.retrieval.no_retrieval import NoRetrievalStrategy
    from src.retrieval.single_step import SingleStepStrategy
    from src.retrieval.multi_step import MultiStepStrategy

    client = llm_client or LLMClient()

    # 创建各策略
    no_retrieval = NoRetrievalStrategy()
    single_step = SingleStepStrategy()
    multi_step = MultiStepStrategy(single_step_strategy=single_step, llm_client=client)

    # 注册
    registry = StrategyRegistry()
    registry.register(no_retrieval)
    registry.register(single_step)
    registry.register(multi_step)

    # 创建自适应策略
    adaptive = AdaptiveStrategy(registry=registry, llm_client=client)

    logger.info(
        "自适应检索链已创建: %d 个策略已注册 (%s)",
        len(registry.list_all()), ", ".join(registry.list_all()),
    )
    return adaptive, registry
