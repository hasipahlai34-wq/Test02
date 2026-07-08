"""
# ============================================================
# ★ Graph 工作流端到端集成测试 (Mock LLM + Mock 检索)
#
# 验证目标:
# 1. 完整 Graph 执行: classify → route → retrieve → generate → review → guard
# 2. GraphState 字段在各节点间的传递完整性 (C3 回归测试)
# 3. simple/medium/complex 三种路由路径
# 4. quality_passed 对最终输出的影响
#
# 所有 LLM 调用均使用 Mock，可在离线环境运行。
# ============================================================
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.types import Document, SearchResult


# ================================================================
# Mock 工具
# ================================================================


def _make_doc(content: str = "测试文档内容", score: float = 0.95) -> Document:
    """创建真实 Document 对象 (可被 msgpack 序列化)"""
    return Document(
        content=content,
        score=score,
        source="test_source",
        metadata={"source": "test_source"},
    )


def _make_mock_llm():
    """
    创建 Mock LLMClient, 预置 classify → review → safety 三次 ask() 返回值。

    调用顺序:
      1. router.classify_query → ask() 返回复杂度分类 JSON
      2. reviewer.review_answer → ask() 返回质量评估 JSON
      3. safety.check_output_safety → ask() 返回安全检测 JSON
    """
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.ask = AsyncMock(
        side_effect=[
            # (1) classify: medium 复杂度
            '{"complexity": "medium", "confidence": 0.85, "reasoning": "中等复杂度查询"}',
            '{"safe": true, "risk_level": "low", "detected_issues": []}',
            # (2) review: 质量通过，无幻觉
            '{"faithfulness": 0.9, "relevance": 0.9, "completeness": 0.8,'
            ' "overall_score": 0.85, "passed": true, "has_hallucination": false,'
            ' "suggestion": ""}',
            # (3) safety output guard: 安全通过
            '{"safe": true, "risk_level": "low", "needs_human_review": false,'
            ' "review_reason": "", "suggested_action": "return"}',
        ]
    )
    mock_llm.generate = AsyncMock(
        return_value="这是 Mock LLM 生成的测试答案。基于检索到的文档内容，相关性强。"
    )
    mock_llm.generate_stream = MagicMock()
    return mock_llm


# ================================================================
# 测试 1: 端到端 Graph 流程 (medium 路径)
# ================================================================


@pytest.mark.asyncio
async def test_graph_workflow_medium_path():
    """
    ★ 完整 Graph 执行流程 — medium 查询路径:
    classify → (route=medium) → single_step → generate → review → guard → END

    验证:
    - generated_answer 非空且格式正确
    - quality_score / quality_passed 正确传递
    - complexity 分类结果影响后续节点
    - hyde_hypothesis 字段传递 (应不存在)
    """
    mock_llm = _make_mock_llm()

    # Mock 检索结果 (使用真实对象, 可被 msgpack 序列化)
    mock_result = SearchResult(
        query="测试查询",
        documents=[_make_doc("相关知识内容", 0.92)],
        total_found=1,
    )

    mock_strategy = MagicMock()
    mock_strategy.retrieve = AsyncMock(return_value=mock_result)

    # 在所有 LLM 调用点和检索入口打 patch
    with patch(
        "src.graph.router.get_llm_client", return_value=mock_llm
    ), patch(
        "src.agents.generator.get_llm_client", return_value=mock_llm
    ), patch(
        "src.agents.reviewer.get_llm_client", return_value=mock_llm
    ), patch(
        "src.safety.content_guard.get_llm_client", return_value=mock_llm
    ), patch(
        "src.retrieval.single_step.get_single_step",
        AsyncMock(return_value=mock_strategy),
    ), patch(
        "src.evaluation.online_evaluator.ragas_evaluate_node",
        AsyncMock(return_value={
            "ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.9},
            "ragas_eval_error": None,
            "ragas_review_failed": False,
        }),
    ), patch(
        "src.graph.hitl.hitl_gate_node",
        AsyncMock(return_value={
            "hitl_status": "none",
            "hitl_decision": "",
        }),
    ):
        from src.graph.workflow import build_adaptive_rag_graph

        graph = build_adaptive_rag_graph()

        initial_state = {
            "query": "测试查询: Python 有什么优势?",
            "session_id": "test_session_001",
            "complexity": "medium",
            "complexity_confidence": 0.5,
            "retrieved_docs": [],
            "completed": False,
        }

        config = {"configurable": {"thread_id": "test_thread_001"}}
        result = await graph.ainvoke(initial_state, config)

        # ---- 断言: 核心字段传递完整性 (C3 回归) ----
        assert result["completed"] is True, "工作流应完整结束"
        assert "测试答案" in result["generated_answer"], "应包含 Mock LLM 生成的答案"
        assert len(result["generated_answer"]) > 20, "生成的答案不应为空"

        # 路由结果
        assert result.get("complexity") == "medium", "复杂度应保持 medium"

        # 检索结果传递
        assert "retrieved_docs" in result, "retrieved_docs 应在 state 中"
        assert len(result["retrieved_docs"]) == 1, "应有 1 个检索到的文档"

        # 质量审核 (review node 应返回质量评分)
        assert result.get("quality_score") == 0.85, "质量评分应为 0.85"
        assert result.get("quality_passed") is True, "质量审核应通过"

        # 安全检查 (guard node 应返回安全结果)
        assert result.get("safety_risk_level") == "low", "安全风险应为 low"
        assert result.get("needs_human_review") is False, "不应触发人工审核"


@pytest.mark.asyncio
async def test_scoped_document_query_skips_global_cache():
    """当前上传文档问答必须走检索，不能被全局缓存命中短路。"""
    from src.graph.workflow import _cache_lookup_node, _cache_store_node

    cache = MagicMock()
    cache.lookup_exact.return_value = "旧文档缓存答案"

    state = {
        "query": "我简历中有几个项目",
        "generated_answer": "ReflexRAG 和 TriAgent",
        "retrieval_filter": {
            "session_id": "session-a",
            "document_ids": ["doc-active"],
        },
    }

    with patch("src.cache.semantic_cache.get_semantic_cache", return_value=cache):
        lookup_result = await _cache_lookup_node(state)
        store_result = await _cache_store_node(state)

    assert lookup_result == {"cache_hit": False, "from_cache": False}
    assert store_result == {}
    cache.lookup_exact.assert_not_called()
    cache.store_exact.assert_not_called()


@pytest.mark.asyncio
async def test_csv_aggregation_uses_active_scope_when_retrieval_empty():
    """CSV 聚合查询不应依赖普通检索先召回 CSV chunk。"""
    mock_result = SearchResult(
        query="薪资最高是谁",
        documents=[],
        total_found=0,
    )
    mock_strategy = MagicMock()
    mock_strategy.retrieve = AsyncMock(return_value=mock_result)

    class FakeIndexer:
        async def _ensure_initialized(self):
            return None

        def get_all_documents(self):
            return [
                {
                    "content": "csv rows",
                    "metadata": {
                        "session_id": "session-csv",
                        "document_id": "doc-csv",
                        "source": "test_data/employees.csv",
                    },
                }
            ]

    mock_strategy._indexer = FakeIndexer()

    with patch(
        "src.retrieval.single_step.get_single_step",
        AsyncMock(return_value=mock_strategy),
    ):
        from src.graph.workflow import _single_step_retrieve_node

        result = await _single_step_retrieve_node({
            "query": "薪资最高是谁",
            "session_id": "session-csv",
            "retrieval_filter": {
                "session_id": "session-csv",
                "document_ids": ["doc-csv"],
            },
        })

    assert result["retrieved_docs"]
    assert "王五" in result["retrieved_docs"][0].content
    assert "45000" in result["retrieved_docs"][0].content


# ================================================================
# 测试 2: simple 路径 (跳过检索)
# ================================================================


@pytest.mark.asyncio
async def test_graph_workflow_simple_path():
    """
    simple 查询路径:
    classify → (route=simple) → no_retrieval → generate → review → guard → END

    验证:
    - simple 路由正确跳过检索 (retrieved_docs 为空)
    - hyde_hypothesis 不出现 (只有 complex 路径产生)
    - 最终答案仍能生成（不依赖检索内容）
    """
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.ask = AsyncMock(
        side_effect=[
            # classify: simple
            '{"complexity": "simple", "confidence": 0.95, "reasoning": "简单事实查询"}',
            # review
            '{"faithfulness": 0.9, "relevance": 0.85, "completeness": 0.8,'
            ' "overall_score": 0.8, "passed": true, "has_hallucination": false,'
            ' "suggestion": ""}',
            # safety
            '{"safe": true, "risk_level": "low", "needs_human_review": false,'
            ' "review_reason": "", "suggested_action": "return"}',
        ]
    )
    mock_llm.generate = AsyncMock(
        return_value="AI 是 Artificial Intelligence 的缩写。"
    )
    mock_llm.generate_stream = MagicMock()

    with patch(
        "src.graph.router.get_llm_client", return_value=mock_llm
    ), patch(
        "src.agents.generator.get_llm_client", return_value=mock_llm
    ), patch(
        "src.agents.reviewer.get_llm_client", return_value=mock_llm
    ), patch(
        "src.safety.content_guard.get_llm_client", return_value=mock_llm
    ), patch(
        "src.evaluation.online_evaluator.ragas_evaluate_node",
        AsyncMock(return_value={
            "ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.9},
            "ragas_eval_error": None,
            "ragas_review_failed": False,
        }),
    ), patch(
        "src.graph.hitl.hitl_gate_node",
        AsyncMock(return_value={
            "hitl_status": "none",
            "hitl_decision": "",
        }),
    ):
        from src.graph.workflow import build_adaptive_rag_graph

        graph = build_adaptive_rag_graph()

        initial_state = {
            "query": "什么是 AI?",
            "session_id": "test_session_002",
            "complexity": "simple",
            "complexity_confidence": 0.5,
            "retrieved_docs": [],
            "completed": False,
        }

        config = {"configurable": {"thread_id": "test_thread_002"}}
        result = await graph.ainvoke(initial_state, config)

        # ---- 断言 ----
        assert result["completed"] is True
        assert "AI" in result["generated_answer"]

        # simple 路径 → 不检索
        assert result.get("search_count") == 0, "simple 查询应跳过检索"
        assert result["retrieved_docs"] == [], "检索文档列表应为空"

        # 分类正确传递
        assert result.get("complexity") == "simple"


# ================================================================
# 测试 3: 状态字段传递完整性 (C3 回归测试)
# ================================================================


@pytest.mark.asyncio
async def test_state_field_propagation():
    """
    ★ C3 修复回归: 验证 GraphState 关键字段在各节点间正确传递。

    重点验证:
    - hyde_hypothesis 字段从 multi_step 传递到 generate
    - conversation_context 在节点间保持
    - complexity 分类结果影响路由决策
    - quality_passed 从 review 传递到 guard
    """
    mock_llm = _make_mock_llm()

    mock_result = SearchResult(
        query="请根据上下文回答问题",
        documents=[_make_doc("字段传递测试文档")],
        total_found=1,
    )

    mock_strategy = MagicMock()
    mock_strategy.retrieve = AsyncMock(return_value=mock_result)

    with patch(
        "src.graph.router.get_llm_client", return_value=mock_llm
    ), patch(
        "src.agents.generator.get_llm_client", return_value=mock_llm
    ), patch(
        "src.agents.reviewer.get_llm_client", return_value=mock_llm
    ), patch(
        "src.safety.content_guard.get_llm_client", return_value=mock_llm
    ), patch(
        "src.retrieval.single_step.get_single_step",
        AsyncMock(return_value=mock_strategy),
    ), patch(
        "src.evaluation.online_evaluator.ragas_evaluate_node",
        AsyncMock(return_value={
            "ragas_scores": {"faithfulness": 0.9, "answer_relevancy": 0.9},
            "ragas_eval_error": None,
            "ragas_review_failed": False,
        }),
    ), patch(
        "src.graph.hitl.hitl_gate_node",
        AsyncMock(return_value={
            "hitl_status": "none",
            "hitl_decision": "",
        }),
    ):
        from src.graph.workflow import build_adaptive_rag_graph

        graph = build_adaptive_rag_graph()

        initial_state = {
            "query": "请根据上下文回答问题",
            "session_id": "test_session_003",
            "complexity": "medium",
            "complexity_confidence": 0.8,
            "classification_reasoning": "上游已分类: 中等复杂度",
            "conversation_context": "历史对话: 用户询问过 Python 相关问题",
            "retrieved_docs": [],
            "completed": False,
        }

        config = {"configurable": {"thread_id": "test_thread_003"}}
        result = await graph.ainvoke(initial_state, config)

        # ---- 断言: C3 回归关键字段 ----
        assert result["completed"] is True

        # 分类字段 → 应被 router 覆盖为 "medium"（从 mock LLM 返回）
        assert result.get("complexity") == "medium"

        # conversation_context → 在 state 中保留
        assert "conversation_context" in result

        # 质量字段 → 从 review node 传递
        assert "quality_score" in result
        assert "quality_passed" in result
        assert result.get("quality_score") == 0.85

        # 安全字段 → 从 guard node 传递
        assert "safety_risk_level" in result
        assert result.get("safety_risk_level") == "low"

        # 检索字段 → 从 single_step node 传递
        assert "search_count" in result
        assert result.get("search_count") == 1
