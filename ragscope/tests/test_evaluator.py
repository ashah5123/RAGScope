# Tests for RAGAS evaluator and metric computation.

from unittest.mock import MagicMock, patch

import pytest

from ragscope.evaluators.ragas_evaluator import RAGASEvaluator


@pytest.fixture
def pipeline_results() -> list[dict]:
    """Minimal pipeline_results as returned by RAGPipeline.run_batch()."""
    return [
        {
            "question": "Who created Python?",
            "answer": "Guido van Rossum",
            "contexts": ["Python was created by Guido van Rossum."],
            "latency_ms": 100.0,
        },
        {
            "question": "What is Python?",
            "answer": "A programming language.",
            "contexts": ["Python is a programming language."],
            "latency_ms": 200.0,
        },
    ]


@pytest.fixture
def ground_truths() -> list[str]:
    """Ground truths matching pipeline_results length."""
    return ["Guido van Rossum", "A programming language created by Guido van Rossum."]


def test_evaluate_returns_dict_with_required_keys(
    pipeline_results: list[dict],
    ground_truths: list[str],
) -> None:
    """RAGASEvaluator.evaluate() returns dict with faithfulness, answer_relevancy, context_recall, context_precision, overall_score, avg_latency_ms."""
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = None
    fake_result.faithfulness = 0.9
    fake_result.answer_relevancy = 0.85
    fake_result.context_recall = 0.88
    fake_result.context_precision = 0.82

    with (
        patch("ragscope.evaluators.ragas_evaluator._get_chat_ollama"),
        patch("ragas.evaluate", return_value=fake_result),
    ):
        evaluator = RAGASEvaluator()
        out = evaluator.evaluate(pipeline_results, ground_truths)

    assert "faithfulness" in out
    assert "answer_relevancy" in out
    assert "context_recall" in out
    assert "context_precision" in out
    assert "overall_score" in out
    assert "avg_latency_ms" in out


def test_evaluate_overall_score_is_mean_of_four_metrics(
    pipeline_results: list[dict],
    ground_truths: list[str],
) -> None:
    """overall_score equals the mean of faithfulness, answer_relevancy, context_recall, context_precision."""
    f, a, c_r, c_p = 0.9, 0.8, 0.85, 0.75
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = None
    fake_result.faithfulness = f
    fake_result.answer_relevancy = a
    fake_result.context_recall = c_r
    fake_result.context_precision = c_p

    with (
        patch("ragscope.evaluators.ragas_evaluator._get_chat_ollama"),
        patch("ragas.evaluate", return_value=fake_result),
    ):
        evaluator = RAGASEvaluator()
        out = evaluator.evaluate(pipeline_results, ground_truths)

    expected_mean = (f + a + c_r + c_p) / 4.0
    assert out["overall_score"] == pytest.approx(expected_mean)
    assert out["faithfulness"] == f
    assert out["answer_relevancy"] == a
    assert out["context_recall"] == c_r
    assert out["context_precision"] == c_p


def test_evaluate_fallback_returns_zeros_but_all_keys(
    pipeline_results: list[dict],
    ground_truths: list[str],
) -> None:
    """When RAGAS raises, evaluate() returns dict with all keys; metrics are 0, avg_latency_ms from input."""
    with patch("ragscope.evaluators.ragas_evaluator._get_chat_ollama"):
        with patch("ragas.evaluate", side_effect=RuntimeError("RAGAS failed")):
            evaluator = RAGASEvaluator()
            out = evaluator.evaluate(pipeline_results, ground_truths)

    assert out["faithfulness"] == 0.0
    assert out["answer_relevancy"] == 0.0
    assert out["context_recall"] == 0.0
    assert out["context_precision"] == 0.0
    assert out["overall_score"] == 0.0
    assert out["avg_latency_ms"] == pytest.approx(150.0)  # (100 + 200) / 2
