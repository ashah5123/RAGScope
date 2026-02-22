# Tests for RAGAS evaluator and metric computation.
# CI/test guard in evaluator ensures real RAGAS is never run; no Ollama/OpenAI needed.

import pytest

from ragscope.evaluators.ragas_evaluator import RAGASEvaluator, _get_chat_ollama

REQUIRED_KEYS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "overall_score",
    "avg_latency_ms",
)


def test_get_chat_ollama_exists():
    """Module-level _get_chat_ollama remains present for tests/patchers."""
    assert callable(_get_chat_ollama)


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


def test_evaluate_returns_dict_with_required_keys_and_float_values(
    pipeline_results: list[dict],
    ground_truths: list[str],
) -> None:
    """evaluate() returns a dict with required keys; values are floats; overall_score = mean of four metrics."""
    evaluator = RAGASEvaluator()
    out = evaluator.evaluate(pipeline_results, ground_truths)

    assert isinstance(out, dict)
    for key in REQUIRED_KEYS:
        assert key in out
        assert isinstance(out[key], float)

    f, a, cr, cp = out["faithfulness"], out["answer_relevancy"], out["context_recall"], out["context_precision"]
    expected_overall = (f + a + cr + cp) / 4.0
    assert out["overall_score"] == pytest.approx(expected_overall)
