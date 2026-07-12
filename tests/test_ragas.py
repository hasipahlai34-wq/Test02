"""
# ============================================================
# RAGAS 评估指标测试
# ============================================================
"""

import pandas as pd
import pytest
from types import SimpleNamespace


class TestRAGASEvaluation:
    """RAGAS 评估基础测试"""

    def test_evaluate_empty_contexts(self):
        """空上下文的评估应返回零分"""
        import asyncio
        from src.evaluation.ragas_eval import evaluate_ragas

        scores = asyncio.run(evaluate_ragas(
            query="test query",
            answer="test answer",
            contexts=[],
        ))

        # 应该返回零分但不崩溃
        assert isinstance(scores, dict)
        assert "faithfulness" in scores
        assert "context_precision" not in scores

    def test_extract_scores_from_ragas_dataframe_result(self):
        """RAGAS EvaluationResult.to_pandas() values should be preserved."""
        from src.evaluation.ragas_eval import _extract_ragas_scores

        class FakeEvaluationResult:
            def to_pandas(self):
                return pd.DataFrame(
                    [{"faithfulness": 0.8, "answer_relevancy": 0.7}]
                )

            def __contains__(self, key):
                return False

        scores = _extract_ragas_scores(
            FakeEvaluationResult(),
            ["faithfulness", "answer_relevancy", "context_precision"],
        )

        assert scores == {"faithfulness": 0.8, "answer_relevancy": 0.7}
        assert "context_precision" not in scores

    def test_extract_scores_omits_invalid_values(self):
        """Invalid metric values must not be silently converted to zero."""
        from src.evaluation.ragas_eval import _extract_ragas_scores

        scores = _extract_ragas_scores(
            {
                "faithfulness": None,
                "answer_relevancy": float("nan"),
                "context_precision": 0.0,
            },
            ["faithfulness", "answer_relevancy", "context_precision"],
        )

        assert scores == {"context_precision": 0.0}

    def test_fallback_scores_include_required_metrics(self):
        """RAGAS 外部依赖不可用时应返回离线保守分数。"""
        from src.evaluation.ragas_eval import _fallback_ragas_scores

        scores = _fallback_ragas_scores(ground_truth="标准答案")

        assert scores == {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }

    def test_fallback_scores_fill_missing_metric(self):
        """缺失指标应能用保守降级值补齐。"""
        from src.evaluation.ragas_eval import _fallback_ragas_scores

        scores = {"answer_relevancy": 0.4}
        fallback = _fallback_ragas_scores()
        for name in ["faithfulness"]:
            scores[name] = fallback[name]

        assert scores == {"answer_relevancy": 0.4, "faithfulness": 0.0}

    def test_build_ragas_metrics_binds_llm_and_embeddings(self):
        from src.evaluation.ragas_eval import _build_ragas_metrics

        llm = object()
        embeddings = object()

        first = _build_ragas_metrics(llm, embeddings, "reference", 1)
        second = _build_ragas_metrics(llm, embeddings, "reference", 1)

        assert [metric.name for metric in first] == [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        assert all(metric.llm is llm for metric in first)
        answer_relevancy = first[1]
        assert answer_relevancy.embeddings is embeddings
        assert all(left is not right for left, right in zip(first, second))

    def test_safe_evaluate_reports_missing_metrics(self, monkeypatch):
        import asyncio
        from src.evaluation import compare

        async def fake_evaluate_ragas(query, answer, contexts, ground_truth):
            return {"faithfulness": 0.8}

        monkeypatch.setattr(compare, "evaluate_ragas", fake_evaluate_ragas)

        scores, eval_error = asyncio.run(compare._safe_evaluate_ragas(
            "query",
            "answer",
            ["context"],
            "ground truth",
        ))

        assert scores == {"faithfulness": 0.8}
        assert eval_error == (
            "Missing metrics: answer_relevancy, context_precision, context_recall"
        )

    def test_benchmark_csv_leaves_missing_metrics_blank(self):
        import csv
        import io
        from test_data.run_full_benchmark import BenchmarkResult, csv_summary

        result = BenchmarkResult("QX", "L1", "query")
        result.direct_answer = {"time_ms": 1}
        result.standard_rag = {
            "time_ms": 2,
            "docs_count": 1,
            "scores": {"faithfulness": 0.5},
            "eval_error": "Missing metrics: answer_relevancy",
        }
        result.adaptive_rag = {
            "time_ms": 3,
            "docs_count": 1,
            "complexity": "medium",
            "strategy": "single_step",
            "scores": {"faithfulness": 0.25},
        }

        rows = list(csv.DictReader(io.StringIO(csv_summary([result]))))

        assert rows[0]["StdRAG_Faithfulness"] == "0.500"
        assert rows[0]["StdRAG_Relevancy"] == ""
        assert rows[0]["StdRAG_Precision"] == ""
        assert rows[0]["Relevancy_Delta%"] == "N/A"
        assert "standard_rag: Missing metrics" in rows[0]["Error"]

    def test_ragas_eval_key_prefers_dedicated_env(self, monkeypatch):
        from src.evaluation.ragas_eval import _resolve_ragas_eval_api_key

        monkeypatch.setenv("RAGAS_EVAL_API_KEY", "deepseek-key")
        settings = SimpleNamespace(
            llm_api_key="main-key",
            llm_base_url="https://jojocode.com/v1",
        )

        assert (
            _resolve_ragas_eval_api_key("https://api.deepseek.com/v1", settings)
            == "deepseek-key"
        )

    def test_ragas_eval_key_rejects_cross_provider_fallback(self, monkeypatch):
        from src.evaluation.ragas_eval import _resolve_ragas_eval_api_key

        monkeypatch.delenv("RAGAS_EVAL_API_KEY", raising=False)
        settings = SimpleNamespace(
            llm_api_key="main-key",
            llm_base_url="https://jojocode.com/v1",
        )

        with pytest.raises(RuntimeError, match="RAGAS_EVAL_API_KEY"):
            _resolve_ragas_eval_api_key("https://api.deepseek.com/v1", settings)

    def test_ragas_eval_key_uses_settings_field(self, monkeypatch):
        from src.evaluation.ragas_eval import _resolve_ragas_eval_api_key

        monkeypatch.delenv("RAGAS_EVAL_API_KEY", raising=False)
        settings = SimpleNamespace(
            llm_api_key="main-key",
            llm_base_url="https://jojocode.com/v1",
            ragas_eval_api_key="settings-deepseek-key",
        )

        assert (
            _resolve_ragas_eval_api_key("https://api.deepseek.com/v1", settings)
            == "settings-deepseek-key"
        )

    def test_ragas_eval_key_allows_same_provider_fallback(self, monkeypatch):
        from src.evaluation.ragas_eval import _resolve_ragas_eval_api_key

        monkeypatch.delenv("RAGAS_EVAL_API_KEY", raising=False)
        settings = SimpleNamespace(
            llm_api_key="main-key",
            llm_base_url="https://api.deepseek.com/v1",
        )

        assert (
            _resolve_ragas_eval_api_key("https://api.deepseek.com/v1", settings)
            == "main-key"
        )

    def test_numeric_validator_catches_wrong_budget_answer(self):
        from src.evaluation.ragas_eval import validate_numeric_answer

        context = """
| 项目 | 预算（万元） | Q1 支出 | Q2 支出 | 剩余 |
|------|------------|--------|--------|------|
| 天枢（智能客服3.0） | 320 | 85 | 125 | 110 |
| 开阳（数据分析平台） | 280 | 70 | 90 | 120 |
| 玉衡（自动化运维） | 150 | 40 | 55 | 55 |
| **合计** | **750** | **195** | **270** | **285** |
"""

        result = validate_numeric_answer(
            "公司总预算为750万元，总支出为465万元，剩余最多的是天枢，110万元。",
            "公司总预算和总支出分别是多少？哪个项目剩余预算最多？",
            [context],
        )

        assert result["numeric_match"] is False
        assert 120 in result["correct_numbers"]

    def test_numeric_validator_accepts_correct_budget_answer(self):
        from src.evaluation.ragas_eval import validate_numeric_answer

        context = """
| 项目 | 预算（万元） | Q1 支出 | Q2 支出 | 剩余 |
|------|------------|--------|--------|------|
| 天枢（智能客服3.0） | 320 | 85 | 125 | 110 |
| 开阳（数据分析平台） | 280 | 70 | 90 | 120 |
| 玉衡（自动化运维） | 150 | 40 | 55 | 55 |
| **合计** | **750** | **195** | **270** | **285** |
"""

        result = validate_numeric_answer(
            "公司总预算为750万元，总支出为465万元，剩余预算最多的是开阳，120万元。",
            "公司总预算和总支出分别是多少？哪个项目剩余预算最多？",
            [context],
        )

        assert result["numeric_match"] is True

    def test_compare_result_structure(self):
        """CompareResult 数据结构测试"""
        from src.types import CompareResult

        result = CompareResult(query="test")
        assert result.query == "test"
        assert result.direct_answer == {}
        assert result.standard_rag == {}
        assert result.adaptive_rag == {}

    def test_token_estimation(self):
        """Token 估算不超过合理范围"""
        from src.utils.token_manager import estimate_tokens

        short_text = "hello"
        short_tokens = estimate_tokens(short_text)
        assert short_tokens > 0
        assert short_tokens < 20  # "hello" 应该很少 token

        long_text = "这是中文" * 100
        long_tokens = estimate_tokens(long_text)
        assert long_tokens > short_tokens

    def test_model_pricing(self):
        """模型定价信息应完整"""
        from src.utils.token_manager import MODEL_PRICING
        assert "gpt-4o-mini" in MODEL_PRICING
        assert "gpt-4o" in MODEL_PRICING
        assert "input" in MODEL_PRICING["gpt-4o-mini"]
        assert "output" in MODEL_PRICING["gpt-4o-mini"]
