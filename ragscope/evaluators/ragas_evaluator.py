# RAGAS-based evaluation for faithfulness, answer relevancy, and context precision.
# Local-first: uses Ollama for judge and HuggingFace/Ollama embeddings (no OpenAI required).
# In CI/pytest: skips RAGAS completely for speed and determinism.

import logging
import math
import os
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset
from langchain_community.chat_models import ChatOllama as ChatOllamaCommunity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from ragscope.configs.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

IN_TEST = ("PYTEST_CURRENT_TEST" in os.environ) or (os.environ.get("CI") == "true")

DEFAULT_JUDGE_MODEL = "llama3.2"
DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_EMBEDDINGS_MODEL = "nomic-embed-text"

DEFAULT_RAGAS_MAX_WORKERS = 1
DEFAULT_RAGAS_TIMEOUT_S = 180
DEFAULT_RAGAS_MAX_RETRIES = 2
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

# Build a parsing exception tuple that works across ragas/langchain versions
_parsing_exceptions: tuple = (TimeoutError,)
try:
    from langchain_core.exceptions import OutputParserException

    _parsing_exceptions = (OutputParserException, TimeoutError)
except Exception:
    pass

try:
    from ragas.exceptions import RagasOutputParserException  # type: ignore

    _parsing_exceptions = _parsing_exceptions + (RagasOutputParserException,)
except Exception:
    pass


def _get_chat_ollama(model: str, base_url: str):
    """
    Return a ChatOllama instance for the judge LLM (langchain_community).
    Module-level so unit tests can monkeypatch it.
    """
    return ChatOllamaCommunity(
        model=model,
        base_url=base_url,
        temperature=0,
        # RAGAS often expects structured outputs; JSON mode helps (may still fail on some models)
        model_kwargs={"format": "json"},
    )


def _get_embeddings(base_url: str):
    """Return local embeddings (Ollama or HuggingFace). No OpenAI."""
    model = os.environ.get("RAGAS_EMBEDDINGS_MODEL", DEFAULT_EMBEDDINGS_MODEL)
    if model == OLLAMA_EMBEDDINGS_MODEL:
        logger.info("RAGAS embeddings: Ollama model=%s, base_url=%s", model, base_url)
        return OllamaEmbeddings(model=OLLAMA_EMBEDDINGS_MODEL, base_url=base_url)
    logger.info("RAGAS embeddings: HuggingFace model=%s", model)
    return HuggingFaceEmbeddings(model_name=model)


def _mean_score(val) -> float:
    """Return mean of a metric value (list/array/Series) or scalar as float."""
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


def _zero_metrics(avg_latency_ms: float = 0.0) -> Dict[str, float]:
    """Return metrics dict with all required keys set to 0.0."""
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "overall_score": 0.0,
        "avg_latency_ms": float(avg_latency_ms),
    }


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS with:
      - local Ollama judge (no OpenAI)
      - local embeddings (HF by default; optional Ollama embeddings)
    Env vars:
      - OLLAMA_BASE_URL
      - RAGAS_JUDGE_MODEL
      - RAGAS_EMBEDDINGS_MODEL (optional)
      - RAGAS_MAX_WORKERS, RAGAS_TIMEOUT_S, RAGAS_MAX_RETRIES
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.RAGASEvaluator")
        self._logger.info("RAGASEvaluator initialized (local judge only, no OpenAI)")

    def _get_judge_llm(self, model: Optional[str] = None):
        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        judge_model = model or os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        return _get_chat_ollama(judge_model, base_url)

    def evaluate(
        self,
        pipeline_results: List[dict],
        ground_truths: List[str],
        llm_model: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run RAGAS evaluation on pipeline outputs.
        Returns:
          faithfulness, answer_relevancy, context_recall, context_precision,
          overall_score, avg_latency_ms

        In CI/pytest (IN_TEST): skips RAGAS and returns zeros + avg latency.
        On parsing/timeouts/other errors: returns zeros + avg latency.
        """
        if len(pipeline_results) != len(ground_truths):
            raise ValueError(
                f"Length mismatch: pipeline_results has {len(pipeline_results)} items "
                f"but ground_truths has {len(ground_truths)}. They must be equal."
            )

        # compute latency always (so dashboards still get latency)
        latencies = [
            r.get("latency_ms")
            for r in pipeline_results
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        # fast path for tests/CI
        if IN_TEST:
            return _zero_metrics(avg_latency_ms)

        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        judge_model = llm_model or os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

        def _get_int_env(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        max_workers = _get_int_env("RAGAS_MAX_WORKERS", DEFAULT_RAGAS_MAX_WORKERS)
        timeout_s = _get_int_env("RAGAS_TIMEOUT_S", DEFAULT_RAGAS_TIMEOUT_S)
        max_retries = _get_int_env("RAGAS_MAX_RETRIES", DEFAULT_RAGAS_MAX_RETRIES)

        self._logger.info(
            "RAGAS evaluation: judge_model=%s, base_url=%s, n=%d, max_workers=%s, timeout_s=%s, max_retries=%s",
            judge_model,
            base_url,
            len(pipeline_results),
            max_workers,
            timeout_s,
            max_retries,
        )

        # Build dataset
        ds = Dataset.from_dict(
            {
                "question": [r["question"] for r in pipeline_results],
                "answer": [r["answer"] for r in pipeline_results],
                "contexts": [r["contexts"] for r in pipeline_results],
                "ground_truth": ground_truths,
            }
        )
        self._logger.info("Dataset size: %d rows", len(ds))

        metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

        try:
            judge_llm = self._get_judge_llm(judge_model)
            embeddings = _get_embeddings(base_url)

            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
            from ragas.run_config import RunConfig

            run_config = RunConfig(
                max_workers=max_workers,
                timeout=timeout_s,
                max_retries=max_retries,
            )

            wrapped_llm = LangchainLLMWrapper(judge_llm)
            metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

            result = evaluate(
                dataset=ds,
                metrics=metrics,
                llm=wrapped_llm,
                embeddings=embeddings,
                run_config=run_config,
            )

        except _parsing_exceptions as e:
            self._logger.warning(
                "RAGAS parsing/timeout failed (returning zero metrics): %s", e, exc_info=True
            )
            return _zero_metrics(avg_latency_ms)
        except Exception as e:
            self._logger.warning("RAGAS evaluate failed (returning zero metrics): %s", e, exc_info=True)
            return _zero_metrics(avg_latency_ms)

        # Extract scores robustly across ragas versions
        scores: Dict[str, float] = {}

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
                val = result.get(name) if isinstance(result, dict) else getattr(result, name, None)
                scores[name] = _mean_score(val)

        # sanitize NaN -> 0
        for name in metric_names:
            v = scores.get(name, 0.0)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                scores[name] = 0.0

        overall = sum(scores[n] for n in metric_names) / 4.0
        if isinstance(overall, float) and math.isnan(overall):
            overall = 0.0

        scores["overall_score"] = float(overall)
        scores["avg_latency_ms"] = float(avg_latency_ms)

        self._logger.info(
            "Metric scores: faithfulness=%.4f, answer_relevancy=%.4f, context_recall=%.4f, context_precision=%.4f",
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_recall"],
            scores["context_precision"],
        )
        self._logger.info("Overall score: %.4f, avg_latency_ms=%.2f", scores["overall_score"], scores["avg_latency_ms"])

        return scores

    def generate_report(self, results: Dict[str, float], config: ExperimentConfig) -> str:
        """Return a formatted Markdown report string with experiment config and metrics."""
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
