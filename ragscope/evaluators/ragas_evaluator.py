from __future__ import annotations

import logging
import math
import os
from typing import Dict, List, Optional


# ---- Dataset import: prefer HF datasets; fallback to ragas' schema ----
try:
    from datasets import Dataset  # type: ignore
except Exception:  # pragma: no cover
    from ragas.dataset_schema import Dataset  # type: ignore


# ---- LLM + embeddings imports: prefer modern packages; fallback gracefully ----
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "llama3.2"
DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Can be overridden by env vars
JUDGE_REQUEST_TIMEOUT_S = int(os.environ.get("RAGAS_TIMEOUT_S", "180"))
RAGAS_MAX_WORKERS = int(os.environ.get("RAGAS_MAX_WORKERS", "1"))
RAGAS_MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "1"))

# If set to "0", we won't pass format="json" to ChatOllama
RAGAS_FORCE_JSON = os.environ.get("RAGAS_FORCE_JSON", "1") != "0"


def _get_metrics():
    """
    RAGAS v0.1+ expects *initialized* metric objects (instances).
    Some older versions expose pre-built metric objects. We support both.
    """
    # Newer API (preferred)
    try:
        from ragas.metrics.collections import (  # type: ignore
            Faithfulness,
            AnswerRelevancy,
            ContextRecall,
            ContextPrecision,
        )

        return [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
    except Exception:
        # Older API: these are typically already metric objects
        from ragas.metrics import (  # type: ignore
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        return [faithfulness, answer_relevancy, context_recall, context_precision]


def _get_chat_ollama(model: str, base_url: str):
    kwargs = dict(
        model=model,
        base_url=base_url,
        temperature=0,
        request_timeout=JUDGE_REQUEST_TIMEOUT_S,
    )
    # Helps many local Ollama models comply with RAGAS JSON parsing.
    if RAGAS_FORCE_JSON:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def _get_embeddings(_: str):
    # Force CPU to avoid MPS OOM on Apple Silicon during evaluation.
    device = os.environ.get("RAGAS_EMBED_DEVICE", "cpu")
    return HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDINGS_MODEL,
        model_kwargs={"device": device},
    )


def _zero(avg_latency_ms: float, notes: str = "") -> Dict[str, float]:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "overall_score": 0.0,
        "avg_latency_ms": float(avg_latency_ms or 0.0),
        "total_cost_usd": 0.0,
        "notes": notes,
    }


def _sanitize_float(x) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return 0.0
        return v
    except Exception:
        return 0.0


class RAGASEvaluator:
    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.RAGASEvaluator")

    def evaluate(
        self,
        pipeline_results: List[dict],
        ground_truths: List[str],
        llm_model: Optional[str] = None,
    ) -> Dict[str, float]:
        # avg latency
        latencies = [
            r.get("latency_ms")
            for r in pipeline_results
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0

        # CI safeguard: GitHub runners won't have Ollama running.
        # Returning deterministic zeros keeps tests stable.
        base_url = os.environ.get("OLLAMA_BASE_URL")
        if not base_url:
            return _zero(avg_latency_ms, notes="ragas_skipped: missing OLLAMA_BASE_URL")

        judge_model = llm_model or os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

        try:
            from ragas import evaluate as ragas_evaluate  # type: ignore
            from ragas.llms import LangchainLLMWrapper  # type: ignore
            from ragas.run_config import RunConfig  # type: ignore

            judge_llm = _get_chat_ollama(judge_model, base_url)
            wrapped_llm = LangchainLLMWrapper(judge_llm)
            embeddings = _get_embeddings(base_url)

            ds = Dataset.from_dict(
                {
                    "user_input": [r["question"] for r in pipeline_results],
                    "retrieved_contexts": [r["contexts"] for r in pipeline_results],
                    "response": [r["answer"] for r in pipeline_results],
                    "reference": ground_truths,
                }
            )

            run_config = RunConfig(
                max_workers=RAGAS_MAX_WORKERS,
                timeout=JUDGE_REQUEST_TIMEOUT_S,
                max_retries=RAGAS_MAX_RETRIES,
            )

            result = ragas_evaluate(
                dataset=ds,
                metrics=_get_metrics(),
                llm=wrapped_llm,
                embeddings=embeddings,
                run_config=run_config,
            )

            # EvaluateResult -> pandas; compute means robustly
            df = result.to_pandas()

            f = _sanitize_float(df["faithfulness"].mean()) if "faithfulness" in df.columns else 0.0
            a = _sanitize_float(df["answer_relevancy"].mean()) if "answer_relevancy" in df.columns else 0.0
            cr = _sanitize_float(df["context_recall"].mean()) if "context_recall" in df.columns else 0.0
            cp = _sanitize_float(df["context_precision"].mean()) if "context_precision" in df.columns else 0.0

            overall = (f + a + cr + cp) / 4.0

            return {
                "faithfulness": float(f),
                "answer_relevancy": float(a),
                "context_recall": float(cr),
                "context_precision": float(cp),
                "overall_score": float(_sanitize_float(overall)),
                "avg_latency_ms": float(avg_latency_ms or 0.0),
                "total_cost_usd": 0.0,
                "notes": "",
            }

        except Exception as e:
            # If Ollama isn't reachable or RAGAS parsing fails, don't crash tests.
            self._logger.exception("RAGAS evaluation failed")
            return _zero(avg_latency_ms, notes=f"ragas_failed: {type(e).__name__}: {e}")
