"""
# ============================================================
# 工作流端到端测试
# ============================================================
"""

import pytest
from src.graph.state import GraphState


class TestGraphState:
    """GraphState 数据模型测试"""

    def test_default_state(self):
        """默认 State 的字段应有合理的默认值"""
        state: GraphState = {
            "query": "测试查询",
            "session_id": "test_session",
        }

        assert state["query"] == "测试查询"
        assert state["session_id"] == "test_session"

    def test_state_has_required_fields(self):
        """State 应包含所有必要的字段"""
        required_fields = {
            "query", "session_id", "messages",
            "complexity", "complexity_confidence", "selected_strategy",
            "retrieved_docs", "generated_answer",
            "quality_score", "quality_passed",
            "completed",
        }
        state_fields = set(GraphState.__annotations__.keys())
        missing = required_fields - state_fields
        assert not missing, f"缺少字段: {missing}"

    def test_agent_state_from_graph_state_ignores_none_placeholders(self):
        """UI initial_state 的 None 占位不应破坏检索状态转换。"""
        from src.types import AgentState

        state = AgentState.from_graph_state({
            "query": "我简历中有几个项目",
            "quality_passed": None,
            "needs_human_review": None,
            "completed": None,
        })

        assert state.quality_passed is True
        assert state.needs_human_review is False
        assert state.completed is False

    def test_public_import_contract(self):
        """核心模块公共导入契约应保持可用。"""
        from src.types import AgentState, GraphState
        from src.graph.hitl import HITLGate, hitl_gate_node
        from src.ingestion.chunker import chunk_csv, chunk_docx, chunk_pdf, chunk_txt
        from src.utils import llm_factory

        assert GraphState
        assert AgentState
        assert HITLGate
        assert hitl_gate_node
        assert chunk_csv
        assert chunk_docx
        assert chunk_pdf
        assert chunk_txt
        assert llm_factory.get_llm_client


class TestGraphRouter:
    """路由逻辑测试"""

    def test_route_simple(self):
        """simple 查询应路由到 no_retrieval"""
        from src.graph.router import route_by_complexity

        state: GraphState = {
            "query": "hello",
            "complexity": "simple",
            "complexity_confidence": 0.9,
        }

        target = route_by_complexity(state)
        assert target == "no_retrieval"

    def test_route_medium(self):
        """medium 查询应路由到 single_step"""
        from src.graph.router import route_by_complexity

        state: GraphState = {
            "query": "营收是多少",
            "complexity": "medium",
            "complexity_confidence": 0.85,
        }

        target = route_by_complexity(state)
        assert target == "single_step"

    def test_route_complex(self):
        """complex 查询应路由到 multi_step"""
        from src.graph.router import route_by_complexity

        state: GraphState = {
            "query": "分析增长因素",
            "complexity": "complex",
            "complexity_confidence": 0.9,
        }

        target = route_by_complexity(state)
        assert target == "multi_step"

    def test_route_unknown_defaults_to_single_step(self):
        """未知复杂度应默认走 single_step"""
        from src.graph.router import route_by_complexity

        state: GraphState = {
            "query": "test",
            "complexity": "unknown_type",
        }

        target = route_by_complexity(state)
        assert target == "single_step"  # 安全默认值


class TestSemanticCache:
    """语义缓存兼容接口测试"""

    def test_legacy_set_stores_answer_payload(self):
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.set("测试查询", "测试答案", ["上下文1"])

        result = cache.lookup_exact("测试查询")

        assert result == {"answer": "测试答案", "contexts": ["上下文1"]}

    def test_store_exact_keeps_string_payload(self):
        from src.cache.semantic_cache import SemanticCache

        cache = SemanticCache()
        cache.store_exact("测试查询", "测试答案")

        assert cache.lookup_exact("测试查询") == "测试答案"
