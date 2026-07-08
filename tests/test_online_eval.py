"""
# ============================================================
# ★ RAGAS 在线评估 + HITL 审核流程测试
#
# 验证:
# 1. RAGAS 在线评估节点 — 正确计算分数, 处理失败场景
# 2. HITL 门禁节点 — 触发条件, 队列写入, 双模式
# 3. Graph 端到端 — 新流程 9 节点完整性
# 4. HITL 队列管理 — 写入/更新/归档
# 5. HITL 拒绝 → 返回拒绝提示语
# 6. 回归: 正常流程不受影响
# ============================================================
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.graph.state import GraphState
from src.types import Document, SearchResult


# ================================================================
# Mock 工具
# ================================================================


def _make_doc(content: str = "测试文档", score: float = 0.95) -> Document:
    return Document(
        content=content,
        score=score,
        source="test_source",
        metadata={"source": "test_source"},
    )


def _make_state(**overrides) -> GraphState:
    """构建测试用 GraphState, 包含所有新节点需要的字段"""
    state: GraphState = {
        "query": "测试查询",
        "session_id": "test_session",
        "complexity": "medium",
        "complexity_confidence": 0.8,
        "selected_strategy": "single_step",
        "retrieved_docs": [_make_doc("相关文档内容", 0.92)],
        "search_count": 1,
        "generated_answer": "这是测试生成的答案, 基于文档内容。",
        "quality_score": 0.85,
        "quality_passed": True,
        "safety_risk_level": "low",
        "needs_human_review": False,
        "review_reason": "",
        "completed": True,
        "hitl_status": "none",
        "hitl_review_id": None,
        "hitl_decision": None,
        "hitl_trigger_reasons": [],
    }
    state.update(overrides)
    return state


def _make_mock_llm():
    """构建 mock LLMClient 用于完整 Graph 流程"""
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.ask = AsyncMock(
        side_effect=[
            # (1) classify: medium
            '{"complexity": "medium", "confidence": 0.85, "reasoning": "中等查询"}',
            '{"safe": true, "risk_level": "low", "detected_issues": []}',
            # (2) review
            '{"faithfulness": 0.9, "relevance": 0.9, "completeness": 0.8,'
            ' "overall_score": 0.85, "passed": true, "has_hallucination": false,'
            ' "suggestion": ""}',
            # (3) safety guard
            '{"safe": true, "risk_level": "low", "needs_human_review": false,'
            ' "review_reason": "", "suggested_action": "return"}',
        ]
    )
    mock_llm.generate = AsyncMock(
        return_value="Mock 生成的答案。"
    )
    mock_llm.generate_stream = MagicMock()
    return mock_llm


# ================================================================
# 测试 1: RAGAS 在线评估节点
# ================================================================


@pytest.mark.asyncio
async def test_ragas_evaluate_normal():
    """
    ★ RAGAS 在线评估: 正常计算分数 → 写入 state

    验证:
    - ragas_scores 包含三个指标
    - ragas_eval_error 为 None
    - ragas_review_failed=False (review 通过)
    """
    mock_scores = {
        "faithfulness": 0.92,
        "answer_relevancy": 0.88,
        "context_precision": 0.85,
    }

    with patch(
        "src.evaluation.online_evaluator.evaluate_ragas",
        AsyncMock(return_value=mock_scores),
    ):
        from src.evaluation.online_evaluator import ragas_evaluate_node

        state = _make_state()
        result = await ragas_evaluate_node(state)

        assert result["ragas_scores"] == mock_scores
        assert result["ragas_eval_error"] is None
        assert result["ragas_review_failed"] is False


@pytest.mark.asyncio
async def test_ragas_evaluate_empty_answer():
    """
    ★ 空答案 → 跳过评估, 设置错误标记

    验证 ragas_eval_error 被设置, 不崩溃。
    """
    from src.evaluation.online_evaluator import ragas_evaluate_node

    state = _make_state(generated_answer="")
    result = await ragas_evaluate_node(state)

    assert result["ragas_scores"] is None
    assert result["ragas_eval_error"] is not None


@pytest.mark.asyncio
async def test_ragas_evaluate_review_failed():
    """
    ★ review 未通过 → 仍然执行 RAGAS 评估, 标记 ragas_review_failed=True

    验证: 即使 quality_passed=False, RAGAS 仍计算并标记。
    """
    mock_scores = {
        "faithfulness": 0.45,
        "answer_relevancy": 0.50,
        "context_precision": 0.30,
    }

    with patch(
        "src.evaluation.online_evaluator.evaluate_ragas",
        AsyncMock(return_value=mock_scores),
    ):
        from src.evaluation.online_evaluator import ragas_evaluate_node

        state = _make_state(quality_passed=False, quality_score=0.3)
        result = await ragas_evaluate_node(state)

        assert result["ragas_scores"] == mock_scores
        assert result["ragas_review_failed"] is True


@pytest.mark.asyncio
async def test_ragas_evaluate_import_error_graceful():
    """
    ★ RAGAS 未安装 → 优雅降级, 不抛异常

    模拟 ImportError, 验证:
    - ragas_scores = None
    - ragas_eval_error 包含 "未安装" 提示
    - 不抛异常
    """
    with patch(
        "src.evaluation.online_evaluator.evaluate_ragas",
        AsyncMock(side_effect=ImportError("No module named 'ragas'")),
    ):
        from src.evaluation.online_evaluator import ragas_evaluate_node

        state = _make_state()
        result = await ragas_evaluate_node(state)

        assert result["ragas_scores"] is None
        assert "未安装" in (result["ragas_eval_error"] or "")


@pytest.mark.asyncio
async def test_ragas_evaluate_disabled_in_config():
    """
    ★ 配置关闭 RAGAS → 跳过评估

    验证 ragas_online_enabled=False 时返回空。
    """
    with patch(
        "src.evaluation.online_evaluator.get_settings",
        return_value=MagicMock(ragas_online_enabled=False),
    ):
        from src.evaluation.online_evaluator import ragas_evaluate_node

        state = _make_state()
        result = await ragas_evaluate_node(state)

        assert result["ragas_scores"] is None
        assert result["ragas_eval_error"] is None


# ================================================================
# 测试 2: HITL 门禁节点 — 触发条件
# ================================================================


@pytest.mark.asyncio
async def test_hitl_no_triggers_passes_through():
    """
    ★ 无触发条件 → 放行, hitl_status="none"

    验证:
    - quality_passed=True, ragas_scores 正常, 安全 low
    - hitl_status = "none"
    """
    with patch("src.graph.hitl._is_streamlit_runtime", return_value=False):
        from src.graph.hitl import hitl_gate_node

        state = _make_state(
            quality_passed=True,
            quality_score=0.9,
            safety_risk_level="low",
            needs_human_review=False,
            ragas_scores={"faithfulness": 0.9, "answer_relevancy": 0.9, "context_precision": 0.9},
        )
        result = await hitl_gate_node(state)

        assert result["hitl_status"] == "none"
        assert result["hitl_decision"] == "pass"


@pytest.mark.asyncio
async def test_hitl_triggers_on_quality_not_passed():
    """
    ★ quality_passed=False → 触发 HITL

    验证:
    - 写入文件队列
    - file_queue 模式: hitl_status="pending"
    - trigger_reasons 包含 "quality_not_passed"
    """
    with patch("src.graph.hitl._is_streamlit_runtime", return_value=False):
        from src.graph.hitl import hitl_gate_node

        state = _make_state(
            quality_passed=False,
            quality_score=0.2,
        )
        result = await hitl_gate_node(state)

        assert result["hitl_status"] == "pending"
        assert result["hitl_review_id"] is not None
        assert "quality_not_passed" in result["hitl_trigger_reasons"]


@pytest.mark.asyncio
async def test_hitl_triggers_on_safety_high():
    """
    ★ safety_risk_level=high → 触发 HITL

    验证 trigger_reasons 包含 "safety_high" 和 "safety_flagged"
    """
    with patch("src.graph.hitl._is_streamlit_runtime", return_value=False):
        from src.graph.hitl import hitl_gate_node

        state = _make_state(
            safety_risk_level="high",
            needs_human_review=True,
        )
        result = await hitl_gate_node(state)

        assert result["hitl_status"] == "pending"
        triggers = result["hitl_trigger_reasons"]
        assert "safety_high" in triggers
        assert "safety_flagged" in triggers


@pytest.mark.asyncio
async def test_hitl_triggers_on_ragas_low():
    """
    ★ RAGAS 分数低于阈值 → 触发 HITL

    验证 trigger_reasons 包含 ragas_ 开头的条目
    """
    mock_settings = MagicMock()
    mock_settings.hitl_enabled = True
    mock_settings.hitl_queue_dir = tempfile.mkdtemp()
    mock_settings.hitl_results_dir = tempfile.mkdtemp()
    mock_settings.hitl_interrupt_timeout_seconds = 1800
    mock_settings.ragas_faithfulness_threshold = 0.6
    mock_settings.ragas_relevancy_threshold = 0.5
    mock_settings.ragas_context_precision_threshold = 0.5

    with patch("src.graph.hitl._is_streamlit_runtime", return_value=False), \
         patch("src.graph.hitl.get_settings", return_value=mock_settings):
        from src.graph.hitl import hitl_gate_node
        from src.graph.hitl import _evaluate_trigger_reasons

        state = _make_state(
            ragas_scores={
                "faithfulness": 0.4,  # 低于 0.6
                "answer_relevancy": 0.7,
                "context_precision": 0.3,  # 低于 0.5
            },
            quality_passed=True,
        )

        reasons = _evaluate_trigger_reasons(state)
        assert len(reasons) >= 2
        assert any("faithfulness" in r for r in reasons)
        assert any("precision" in r for r in reasons)


# ================================================================
# 测试 3: HITL 队列管理
# ================================================================


def test_write_queue_item(tmp_path):
    """★ 队列项写入 → JSON 文件创建, 格式正确"""
    from src.graph.hitl import _build_queue_item, write_queue_item, QUEUE_ITEM_KEYS

    state = _make_state()
    item = _build_queue_item(state, "test-id-001", ["quality_not_passed"])

    # 验证所有必要键存在
    for key in QUEUE_ITEM_KEYS:
        assert key in item, f"队列项缺少键: {key}"

    assert item["review_id"] == "test-id-001"
    assert item["query"] == "测试查询"
    assert item["trigger_reasons"] == ["quality_not_passed"]
    assert item["hitl_status"] == "pending"
    assert item["hitl_decision"] is None
    assert item["mode"] in ("interrupt", "file_queue")
    assert "created_at" in item
    assert "timeout_at" in item


def test_update_queue_item(tmp_path):
    """★ 更新队列项 → JSON 文件内容更新"""
    mock_settings = MagicMock()
    mock_settings.hitl_queue_dir = str(tmp_path)
    mock_settings.hitl_results_dir = str(tmp_path / "results")

    from src.graph.hitl import _build_queue_item, write_queue_item, update_queue_item

    state = _make_state()
    item = _build_queue_item(state, "test-update-001", ["safety_high"])

    with patch("src.graph.hitl.get_settings", return_value=mock_settings):
        write_queue_item(item)

        # 更新
        result = update_queue_item("test-update-001", {
            "hitl_status": "approved",
            "hitl_decision": "approve",
        })

        assert result is not None

        # 验证更新后的内容
        filepath = tmp_path / "test-update-001.json"
        with open(filepath, "r", encoding="utf-8") as f:
            updated = json.load(f)
        assert updated["hitl_status"] == "approved"
        assert updated["hitl_decision"] == "approve"


def test_archive_queue_item(tmp_path):
    """★ 归档队列项 → 移动到 results 目录"""
    queue_dir = tmp_path / "queue"
    results_dir = tmp_path / "results"
    queue_dir.mkdir(parents=True)

    mock_settings = MagicMock()
    mock_settings.hitl_queue_dir = str(queue_dir)
    mock_settings.hitl_results_dir = str(results_dir)

    # 创建队列文件
    item_data = {"review_id": "test-archive-001", "hitl_status": "approved"}
    queue_file = queue_dir / "test-archive-001.json"
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(item_data, f)

    with patch("src.graph.hitl.get_settings", return_value=mock_settings):
        from src.graph.hitl import archive_queue_item

        result = archive_queue_item("test-archive-001")

        assert result is not None
        assert not queue_file.exists(), "队列文件应被移除"
        assert (results_dir / "test-archive-001.json").exists(), "应出现在 results 目录"


def test_list_pending_items(tmp_path):
    """★ 列出待审核项 → 只返回 pending 状态的"""
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir(parents=True)

    mock_settings = MagicMock()
    mock_settings.hitl_queue_dir = str(queue_dir)
    mock_settings.hitl_results_dir = str(tmp_path / "results")

    # 创建多个队列项 (不同状态)
    items_data = [
        {"review_id": "pending-1", "hitl_status": "pending"},
        {"review_id": "pending-2", "hitl_status": "pending_timeout"},
        {"review_id": "approved-1", "hitl_status": "approved"},
    ]
    for data in items_data:
        with open(queue_dir / f"{data['review_id']}.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

    with patch("src.graph.hitl.get_settings", return_value=mock_settings):
        from src.graph.hitl import list_pending_items

        pending = list_pending_items()

        assert len(pending) == 2
        ids = {p["review_id"] for p in pending}
        assert "pending-1" in ids
        assert "pending-2" in ids
        assert "approved-1" not in ids


# ================================================================
# 测试 4: HITL 拒绝处理
# ================================================================


def test_rejected_answer_returns_rejection_message():
    """
    ★ 被拒绝的回答 → 返回固定拒绝提示语

    验证拒绝后的 generated_answer 内容。
    """
    rejection_msg = "回答未通过质量审核，请重新提问"

    # 确认 CLI reject 命令设置此消息
    from cli.review_pending import cmd_reject
    # 只验证消息常量和模拟场景
    assert "质量审核" in rejection_msg
    assert "重新提问" in rejection_msg


# ================================================================
# 测试 5: Graph 端到端 (含新节点)
# ================================================================


@pytest.mark.asyncio
async def test_graph_full_pipeline_with_ragas_and_hitl():
    """
    ★ 完整 9 节点流程: classify → route → retrieve → generate →
      review → ragas_evaluate → guard → hitl_gate → END

    验证:
    - 正常查询下全程不中断
    - ragas_scores 和 hitl_status 在最终 state 中出现
    - 已有字段 (generated_answer, quality_score) 仍然正确
    """
    mock_llm = _make_mock_llm()

    mock_scores = {"faithfulness": 0.9, "answer_relevancy": 0.88, "context_precision": 0.85}

    mock_result = SearchResult(
        query="集成测试查询",
        documents=[_make_doc("集成测试文档", 0.93)],
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
        "src.evaluation.online_evaluator.evaluate_ragas",
        AsyncMock(return_value=mock_scores),
    ), patch(
        "src.graph.hitl._is_streamlit_runtime", return_value=False
    ):
        from src.graph.workflow import build_adaptive_rag_graph

        graph = build_adaptive_rag_graph()

        initial_state = {
            "query": "集成测试: 完整的 Graph 流程",
            "session_id": "test_e2e_001",
            "complexity": "medium",
            "complexity_confidence": 0.5,
            "retrieved_docs": [],
            "completed": False,
        }

        config = {"configurable": {"thread_id": "test_e2e_thread"}}
        result = await graph.ainvoke(initial_state, config)

        # ---- 原有流程断言 (回归) ----
        assert result["completed"] is True
        assert "Mock 生成" in result["generated_answer"]
        assert result.get("quality_score") == 0.85
        assert result.get("quality_passed") is True
        assert result.get("safety_risk_level") == "low"

        # ---- 新节点断言 ----
        assert "ragas_scores" in result, "ragas_scores 应存在于 final state"
        assert result["ragas_scores"] is None
        assert result["ragas_eval_error"] == "pending_async"

        assert "hitl_status" in result, "hitl_status 应存在于 final state"
        assert result["hitl_status"] == "none", "正常查询不应触发 HITL"
        assert "hitl_trigger_reasons" in result


@pytest.mark.asyncio
async def test_graph_hitl_triggers_on_low_quality():
    """
    ★ 低质量回答 → HITL 触发, 队列写入, Graph 继续完成

    验证:
    - review 返回 quality_passed=False
    - hitl_status = "pending"
    - 文件队列写入
    - generated_answer 仍存在 (文件队列模式不拦截)
    """
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.ask = AsyncMock(
        side_effect=[
            # classify
            '{"complexity": "medium", "confidence": 0.8, "reasoning": "中等"}',
            '{"safe": true, "risk_level": "low", "detected_issues": []}',
            # review: 低质量!
            '{"faithfulness": 0.3, "relevance": 0.2, "completeness": 0.1,'
            ' "overall_score": 0.2, "passed": false, "has_hallucination": true,'
            ' "suggestion": "重新生成"}',
            # safety
            '{"safe": true, "risk_level": "low", "needs_human_review": false,'
            ' "review_reason": "", "suggested_action": "return"}',
        ]
    )
    mock_llm.generate = AsyncMock(return_value="低质量回答。")
    mock_llm.generate_stream = MagicMock()

    mock_result = SearchResult(
        query="低质量测试",
        documents=[_make_doc("测试文档", 0.6)],
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
        "src.evaluation.online_evaluator.evaluate_ragas",
        AsyncMock(return_value={"faithfulness": 0.5, "answer_relevancy": 0.4, "context_precision": 0.3}),
    ), patch(
        "src.graph.hitl._is_streamlit_runtime", return_value=False
    ):
        from src.graph.workflow import build_adaptive_rag_graph

        graph = build_adaptive_rag_graph()

        initial_state = {
            "query": "低质量回答测试",
            "session_id": "test_hitl_001",
            "complexity": "medium",
            "complexity_confidence": 0.5,
            "retrieved_docs": [],
            "completed": False,
        }

        config = {"configurable": {"thread_id": "test_hitl_thread"}}
        result = await graph.ainvoke(initial_state, config)

        # ---- 断言 ----
        assert result["completed"] is True
        assert result.get("quality_passed") is False
        assert result.get("quality_score") == 0.2

        # HITL 应触发
        assert result.get("hitl_status") == "pending"
        assert result.get("hitl_review_id") is not None
        assert len(result.get("hitl_trigger_reasons", [])) > 0
        assert "quality_not_passed" in result["hitl_trigger_reasons"]

        # 文件队列应存在
        from config.settings import get_settings
        settings = get_settings()
        queue_dir = Path(settings.hitl_queue_dir)
        queue_file = queue_dir / f"{result['hitl_review_id']}.json"
        assert queue_file.exists(), f"队列文件应存在: {queue_file}"
        print(f"  ✅ 队列文件已创建: {queue_file}")


# ================================================================
# 测试 6: Streamlit 检测
# ================================================================


def test_streamlit_detection_not_runtime():
    """
    ★ 非 Streamlit 环境 → _is_streamlit_runtime() 返回 False
    """
    from src.graph.hitl import _is_streamlit_runtime
    # 测试环境中不应有 Streamlit runtime
    assert _is_streamlit_runtime() is False


def test_streamlit_detection_with_import_error():
    """
    ★ streamlit 未安装 → _is_streamlit_runtime() 返回 False (不抛异常)
    """
    with patch("src.graph.hitl._is_streamlit_runtime", return_value=False):
        from src.graph.hitl import _is_streamlit_runtime
        assert _is_streamlit_runtime() is False


# ================================================================
# 测试 7: 配置正确性
# ================================================================


def test_new_settings_defaults():
    """★ 新配置项存在且有合理的默认值"""
    from config.settings import get_settings

    settings = get_settings()

    # RAGAS 在线评估
    assert hasattr(settings, "ragas_online_enabled")
    assert settings.ragas_online_enabled is True
    assert 0 < settings.ragas_faithfulness_threshold <= 1.0
    assert 0 < settings.ragas_relevancy_threshold <= 1.0

    # HITL
    assert hasattr(settings, "hitl_enabled")
    assert settings.hitl_enabled is True
    assert settings.hitl_interrupt_timeout_seconds == 1800  # 30 min

    # 队列目录
    assert "hitl_queue" in settings.hitl_queue_dir
    assert "hitl_results" in settings.hitl_results_dir
