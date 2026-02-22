# RAGAS-based evaluation for faithfulness, answer relevancy, and context precision.

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset

from ragscope.configs.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


def _get_chat_ollama(model: str, base_url: str):
    """Return ChatOllama from langchain_ollama or langchain_community. No OpenAI."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=model, base_url=base_url)


def _mean_score(val) -> float:
    """Return mean of a metric value (list/array/Series) or the scalar as float."""
    if val is None:
        return 0.0
    if isinstance(val, pd.Series):
        return float(val.mean()) if len(val) else 0.0
    if isinstance(val, (list, tuple)):
        if not val:
            return 0.0
        return sum(float(x) for x in val) / len(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS with a local judge LLM (Ollama).
    100% free/local: no OpenAI; uses OLLAMA_BASE_URL and optional llm_model.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.RAGASEvaluator")
        self._logger.info("RAGASEvaluator initialized (local judge only, no OpenAI)")

    def evaluate(
        self,
        pipeline_results: List[dict],
        ground_truths: List[str],
        llm_model: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run RAGAS evaluation on pipeline outputs using a local Ollama judge.
        Validates lengths, builds dataset, runs RAGAS with local LLM; returns
        faithfulness, answer_relevancy, context_recall, context_precision,
        overall_score, avg_latency_ms. On failure returns zeros for metrics
        but still returns all keys with avg_latency_ms from pipeline_results.
        """
        if len(pipeline_results) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: pipeline_results has {len(pipeline_results)} items "
                f"but ground_truths has {len(ground_truths)}. They must be equal."
            )

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model = llm_model or "llama3.2"
        self._logger.info(
            "Starting RAGAS evaluation (local judge): model=%s, base_url=%s, n=%d",
            model,
            base_url,
            len(pipeline_results),
        )

        questions = [r["question"] for r in pipeline_results]
        answers = [r["answer"] for r in pipeline_results]
        contexts = [r["contexts"] for r in pipeline_results]
        latencies = [
            r.get("latency_ms")
            for r in pipeline_results
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        ds = Dataset.from_dict(data)
        self._logger.info("Dataset size: %d rows", len(ds))

        metric_names = [
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
        ]
        zero_scores = {n: 0.0 for n in metric_names}
        zero_scores["overall_score"] = 0.0
        zero_scores["avg_latency_ms"] = avg_latency_ms

        try:
            judge_llm = _get_chat_ollama(model, base_url)
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            wrapped_llm = LangchainLLMWrapper(judge_llm)
            metrics = [
                faithfulness,
                answer_relevancy,
                context_recall,
                context_precision,
            ]
            result = evaluate(
                dataset=ds,
                metrics=metrics,
                llm=wrapped_llm,
            )
        except Exception as e:
            self._logger.exception(
                "RAGAS evaluate failed (returning zeros). Error: %s",
                e,
            )
            return zero_scores

        scores = {}
        if hasattr(result, "to_pandas"):
            try:
                df = result.to_pandas()
                if df is not None and not df.empty:
                    for name in metric_names:
                        if name in df.columns:
                            scores[name] = _mean_score(df[name])
            except Exception:
                pass

        for name in metric_names:
            if name not in scores:
                val = (
                    result.get(name)
                    if isinstance(result, dict)
                    else getattr(result, name, None)
                )
                scores[name] = _mean_score(val)

        overall = sum(scores.get(n, 0.0) for n in metric_names) / 4.0
        scores["overall_score"] = overall
        scores["avg_latency_ms"] = avg_latency_ms

        self._logger.info(
            "Metric scores: faithfulness=%.4f, answer_relevancy=%.4f, "
            "context_recall=%.4f, context_precision=%.4f",
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_recall"],
            scores["context_precision"],
        )
        self._logger.info("Overall score: %.4f, avg_latency_ms=%.2f", overall, avg_latency_ms)

        return scores

    def generate_report(self, results: Dict[str, float], config: ExperimentConfig) -> str:
        """
        Return a formatted Markdown report string with experiment config and metrics.
        """
        return f"""# RAGScope Experiment Report

**Experiment Name:** {config.experiment_name}
**Experiment ID:** {config.experiment_id}

## Config

- **Chunk Size:** {config.chunk_size}
- **Chunk Overlap:** {config.chunk_overlap}
- **Embedding Model:** {config.embedding_model}
- **Retriever Type:** {config.retriever_type}
- **Top K:** {config.top_k}
- **LLM Model:** {config.llm_model}

## Metrics

- **Faithfulness:** {results.get("faithfulness", 0):.4f}
- **Answer Relevancy:** {results.get("answer_relevancy", 0):.4f}
- **Context Recall:** {results.get("context_recall", 0):.4f}
- **Context Precision:** {results.get("context_precision", 0):.4f}
- **Overall Score:** {results.get("overall_score", 0):.4f}
- **Avg Latency (ms):** {results.get("avg_latency_ms", 0):.2f}
"""
