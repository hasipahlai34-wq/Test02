"""
# ============================================================
# RAGAS 评估指标测试
# ============================================================
"""

import pandas as pd
import pytest


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
