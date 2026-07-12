"""
# ============================================================
# 检索策略测试
# ============================================================
"""

import pytest
from src.types import AgentState, QueryComplexity
from langchain_core.documents import Document as LCDocument


class TestRetrievalStrategies:
    """检索策略基础测试"""

    def test_no_retrieval_returns_empty(self):
        """NoRetrieval 应返回空结果"""
        import asyncio
        from src.retrieval.no_retrieval import NoRetrievalStrategy

        strategy = NoRetrievalStrategy()
        state = AgentState(query="什么是AI")
        result = asyncio.run(strategy.retrieve("什么是AI", state))

        assert result.total_found == 0
        assert len(result.documents) == 0

    def test_strategy_registry(self):
        """策略注册表的基础功能"""
        from src.retrieval.base import StrategyRegistry
        from src.retrieval.no_retrieval import NoRetrievalStrategy

        registry = StrategyRegistry()
        no_ret = NoRetrievalStrategy()
        registry.register(no_ret)

        retrieved = registry.get("no_retrieval")
        assert retrieved is not None
        assert retrieved.strategy_type.value == "no_retrieval"

    def test_adaptive_chain_creation(self):
        """自适应检索链应正确创建三种策略"""
        from unittest.mock import MagicMock, patch
        from src.retrieval.adaptive import create_adaptive_chain

        mock_llm = MagicMock()
        mock_llm.model_name = "mock-model"
        mock_llm.ask = MagicMock()

        with patch("src.retrieval.adaptive.LLMClient", return_value=mock_llm):
            adaptive, registry = create_adaptive_chain()

        assert "no_retrieval" in registry.list_all()
        assert "single_step" in registry.list_all()
        assert "multi_step" in registry.list_all()

    def test_single_step_retrieval_filter_keeps_active_document_scope(self):
        """Filtered retrieval must not mix historical documents into current-session RAG."""
        import asyncio
        from types import MethodType
        from src.retrieval.single_step import SingleStepStrategy

        active_meta = {
            "session_id": "session-a",
            "document_id": "doc-active",
            "source": "active.pdf",
            "source_name": "active.pdf",
            "chunk_index": 0,
        }
        history_meta = {
            "session_id": "session-old",
            "document_id": "doc-history",
            "source": "history.pdf",
            "source_name": "history.pdf",
            "chunk_index": 0,
        }

        class FakeIndexer:
            async def _ensure_initialized(self):
                return None

            def get_all_documents(self):
                return [
                    {
                        "content": "ReflexRAG TriAgent current resume projects",
                        "metadata": active_meta,
                    },
                    {
                        "content": "Enterprise education microservice Spring Cloud history project",
                        "metadata": history_meta,
                    },
                ]

            async def search(self, query, top_k=10, filter_dict=None, score_threshold=None):
                assert filter_dict == {
                    "$and": [
                        {"session_id": "session-a"},
                        {"document_id": "doc-active"},
                    ]
                }
                return [
                    (
                        LCDocument(
                            page_content="ReflexRAG TriAgent current resume projects",
                            metadata=active_meta,
                        ),
                        0.9,
                    )
                ]

        strategy = SingleStepStrategy(indexer=FakeIndexer(), rerank_threshold=0)

        async def fake_rerank(self, query, documents, top_k=5, threshold=0.3):
            return [(doc, 0.9) for doc in documents[:top_k]]

        strategy._rerank = MethodType(fake_rerank, strategy)
        result = asyncio.run(
            strategy.retrieve(
                "ReflexRAG projects",
                AgentState(query="ReflexRAG projects"),
                retrieval_filter={
                    "session_id": "session-a",
                    "document_ids": ["doc-active"],
                },
            )
        )

        assert result.documents
        combined = "\n".join(doc.content for doc in result.documents)
        assert "ReflexRAG" in combined
        assert "Enterprise education" not in combined
        assert {doc.metadata.get("document_id") for doc in result.documents} == {"doc-active"}
        assert {doc.metadata.get("source_name") for doc in result.documents} == {"active.pdf"}

    def test_csv_aggregation_detects_salary_max_query(self):
        """CSV 聚合应识别薪资最高类查询并返回对应人员。"""
        from src.retrieval.csv_aggregator import (
            execute_csv_aggregation,
            is_aggregate_query,
        )

        query = "薪资最高的是谁"

        assert is_aggregate_query(query)
        assert not is_aggregate_query("文档内容是什么")

        result = execute_csv_aggregation("test_data/employees.csv", query)

        assert result is not None
        assert "王五" in result
        assert "45000" in result

    def test_single_step_retrieve_accepts_optional_state_and_top_k(self):
        """诊断脚本可直接按 query/top_k 调用单步检索。"""
        import asyncio
        from types import MethodType
        from src.retrieval.single_step import SingleStepStrategy

        class FakeIndexer:
            async def _ensure_initialized(self):
                return None

            def get_all_documents(self):
                return [
                    {"content": "项目经历 ReflexRAG", "metadata": {"source": "resume.pdf"}},
                    {"content": "项目经历 TriAgent", "metadata": {"source": "resume.pdf"}},
                ]

            async def search(self, query, top_k=10, filter_dict=None, score_threshold=None):
                return [
                    (
                        LCDocument(
                            page_content="项目经历 ReflexRAG",
                            metadata={"source": "resume.pdf"},
                        ),
                        0.9,
                    ),
                    (
                        LCDocument(
                            page_content="项目经历 TriAgent",
                            metadata={"source": "resume.pdf"},
                        ),
                        0.8,
                    ),
                ]

        strategy = SingleStepStrategy(indexer=FakeIndexer(), rerank_threshold=0)

        async def fake_rerank(self, query, documents, top_k=5, threshold=0.3):
            return [(doc, 0.9) for doc in documents[:top_k]]

        strategy._rerank = MethodType(fake_rerank, strategy)
        result = asyncio.run(strategy.retrieve("项目经历", top_k=1))

        assert result.documents
        assert len(result.documents) == 1

    def test_markdown_table_aggregation_calculates_budget_remaining(self):
        from src.retrieval.single_step import calculate_markdown_table_aggregation

        context = """
| 项目 | 预算（万元） | Q1 支出 | Q2 支出 | 剩余 |
|------|------------|--------|--------|------|
| 天枢（智能客服3.0） | 320 | 85 | 125 | 110 |
| 开阳（数据分析平台） | 280 | 70 | 90 | 120 |
| 玉衡（自动化运维） | 150 | 40 | 55 | 55 |
| **合计** | **750** | **195** | **270** | **285** |
"""

        result = calculate_markdown_table_aggregation(
            "公司总预算和总支出分别是多少？哪个项目剩余预算最多？",
            [context],
        )

        assert result is not None
        assert "总预算: 750万元" in result
        assert "总支出: 465万元" in result
        assert "开阳" in result
        assert "120万元" in result


    @pytest.mark.asyncio
    async def test_multi_step_scoped_retrieval_uses_multi_query_without_hyde(self):
        """Uploaded-document scoped retrieval should not use HyDE as the first hop."""
        from unittest.mock import AsyncMock, MagicMock

        from src.retrieval.multi_step import MultiStepStrategy
        from src.types import AgentState, Document

        calls = []

        class FakeSingleStep:
            async def retrieve(self, query, state, **kwargs):
                calls.append((query, kwargs))
                return type("Result", (), {
                    "documents": [
                        Document(
                            content=f"evidence for {query}",
                            score=0.9,
                            metadata={"document_id": "doc-active", "chunk_index": query},
                        )
                    ]
                })()

            async def _rerank(self, query, documents, top_k=5, threshold=0.3):
                return [(doc, 1.0 - index * 0.01) for index, doc in enumerate(documents[:top_k])]

        strategy = MultiStepStrategy(single_step_strategy=FakeSingleStep(), llm_client=MagicMock())
        strategy._rewriter.generate_multi_queries = AsyncMock(
            return_value=["天枢项目 延期 原因", "天枢项目 风险 时间线"]
        )
        strategy._hyde.generate = AsyncMock(return_value="hypothetical answer")
        strategy._evaluate_sufficiency = AsyncMock(return_value=(True, ""))

        retrieval_filter = {"session_id": "session-a", "document_ids": ["doc-active"]}
        result = await strategy.retrieve(
            "为什么天枢项目可能延期？",
            AgentState(query="为什么天枢项目可能延期？"),
            retrieval_filter=retrieval_filter,
        )

        called_queries = [query for query, _ in calls]
        assert "为什么天枢项目可能延期？" in called_queries
        assert "天枢项目 延期 原因" in called_queries
        assert "天枢项目 风险 时间线" in called_queries
        assert any("相关证据" in query for query in called_queries)
        assert all(kwargs.get("retrieval_filter") == retrieval_filter for _, kwargs in calls)
        strategy._hyde.generate.assert_not_called()
        assert result.documents

    @pytest.mark.asyncio
    async def test_multi_step_non_scoped_empty_recall_uses_hyde_fallback(self):
        """HyDE remains available as a fallback for non-scoped retrieval misses."""
        from unittest.mock import AsyncMock, MagicMock

        from src.retrieval.multi_step import MultiStepStrategy
        from src.types import AgentState, Document

        calls = []

        class FakeSingleStep:
            async def retrieve(self, query, state, **kwargs):
                calls.append(query)
                docs = []
                if query == "hyde fallback query":
                    docs = [Document(content="hyde fallback evidence", score=0.8)]
                return type("Result", (), {"documents": docs})()

            async def _rerank(self, query, documents, top_k=5, threshold=0.3):
                return [(doc, 0.9) for doc in documents[:top_k]]

        agent_state = AgentState(query="分析增长原因")
        strategy = MultiStepStrategy(single_step_strategy=FakeSingleStep(), llm_client=MagicMock())
        strategy._rewriter.generate_multi_queries = AsyncMock(return_value=[])
        strategy._hyde.generate = AsyncMock(return_value="hyde fallback query")
        strategy._evaluate_sufficiency = AsyncMock(return_value=(True, ""))

        result = await strategy.retrieve("分析增长原因", agent_state)

        assert "hyde fallback query" in calls
        strategy._hyde.generate.assert_awaited_once()
        assert agent_state.hyde_hypothesis == "hyde fallback query"
        assert result.documents


class TestQueryClassification:
    """查询复杂度分类测试"""

    def test_simple_query_detection(self):
        """简单查询应被正确分类"""
        import asyncio
        from src.retrieval.adaptive import AdaptiveStrategy

        # 注意: 这需要 API Key，在 CI 中可以 mock
        pass  # 实际测试需要 mock LLMClient

    def test_complexity_values(self):
        """复杂度枚举值的正确性"""
        assert QueryComplexity.SIMPLE.value == "simple"
        assert QueryComplexity.MEDIUM.value == "medium"
        assert QueryComplexity.COMPLEX.value == "complex"

    def test_aggregate_query_routes_to_medium(self):
        from src.graph.router import is_aggregate_query, quick_classify

        query = "公司总预算和总支出分别是多少？哪个项目剩余预算最多？"

        assert is_aggregate_query(query)
        assert quick_classify(query) == "medium"

    def test_list_aggregation_is_not_single_fact_low_cost(self):
        from src.graph.router import is_list_aggregation_query, is_low_cost_retrieval_query, quick_classify

        assert is_list_aggregation_query(
            "星穹科技目前一共有多少个正在进行的项目？它们分别属于哪些部门？"
        )
        list_all_query = "\u5217\u51fa\u6240\u6709\u9879\u76ee\u53ca\u5176\u6240\u5c5e\u90e8\u95e8"

        assert is_list_aggregation_query(list_all_query)
        assert is_low_cost_retrieval_query(list_all_query)
        assert quick_classify(list_all_query) == "medium"

    def test_implicit_inference_routes_to_complex(self):
        from src.graph.router import (
            is_implicit_inference_query,
            is_low_cost_retrieval_query,
            quick_classify,
        )

        query = (
            "\u5982\u679c\u5929\u67a2\u9879\u76ee 10 \u6708\u53d1\u5e03\u524d"
            "\u9700\u8981\u7d27\u6025\u52a0\u4eba\uff0c\u8c01\u6700\u6709\u53ef\u80fd"
            "\u88ab\u62bd\u8c03\u8fc7\u53bb\u5e2e\u5fd9\uff1f\u4e3a\u4ec0\u4e48\uff1f"
        )

        assert is_implicit_inference_query(query)
        assert not is_low_cost_retrieval_query(query)
        assert quick_classify(query) == "complex"


class TestEmbeddingModel:
    """Embedding 初始化测试"""

    def test_hf_model_cache_detection(self, tmp_path, monkeypatch):
        from src.models.embeddings import _hf_model_cache_exists

        cache_dir = (
            tmp_path
            / "hub"
            / "models--sentence-transformers--demo-model"
            / "snapshots"
            / "abc123"
        )
        cache_dir.mkdir(parents=True)
        (cache_dir / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        assert _hf_model_cache_exists("sentence-transformers/demo-model")
        assert not _hf_model_cache_exists("sentence-transformers/missing-model")
