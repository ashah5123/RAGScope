from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional


def _get_metrics():
    # Ragas expects initialised metric objects (instances).
    # Prefer newer API; fall back to older metric objects.
    try:
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
        return [Faithfulness(), AnswerRelevancy(), ContextRecall(), ContextPrecision()]
    except Exception:
        from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
        return [faithfulness, answer_relevancy, context_recall, context_precision]

try:
    from datasets import Dataset
except Exception:
    from ragas.dataset_schema import Dataset

# Use modern langchain packages if available; fall back gracefully.
try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover
    from langchain_community.chat_models import ChatOllama  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_JUDGE_MODEL = "llama3.2"
DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Can be overridden by env vars
JUDGE_REQUEST_TIMEOUT_S = int(os.environ.get("RAGAS_TIMEOUT_S", "180"))
RAGAS_MAX_WORKERS = int(os.environ.get("RAGAS_MAX_WORKERS", "1"))
RAGAS_MAX_RETRIES = int(os.environ.get("RAGAS_MAX_RETRIES", "1"))


def _get_chat_ollama(model: str, base_url: str):
    # Critical: force JSON mode for RAGAS prompts.
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        format="json",
        request_timeout=JUDGE_REQUEST_TIMEOUT_S,
    )


def _get_embeddings(base_url: str):
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
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


class RAGASEvaluator:
    def __init__(self) -> None:
        self._logger = logging.getLogger(f"{__name__}.RAGASEvaluator")

    def evaluate(
        self,
        pipeline_results: List[dict],
        ground_truths: List[str],
        llm_model: Optional[str] = None,
    ) -> Dict[str, float]:
        latencies = [
            r.get("latency_ms")
            for r in pipeline_results
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0

        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

        # Important: judge model comes from env by default.
        # We only use llm_model if explicitly passed.
        judge_model = llm_model or os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.llms import LangchainLLMWrapper
            from ragas.run_config import RunConfig

            # Prefer collections imports (your environment shows these work)
            try:
                from ragas.metrics.collections import (
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                )
            except Exception:
                from ragas.metrics import (  # type: ignore
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                    context_precision,
                )

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

            # Robust: compute means from the evaluation dataframe
            df = result.to_pandas()
            scores = {
                "faithfulness": float(df["faithfulness"].mean()),
                "answer_relevancy": float(df["answer_relevancy"].mean()),
                "context_recall": float(df["context_recall"].mean()),
                "context_precision": float(df["context_precision"].mean()),
            }
            scores["overall_score"] = sum(scores.values()) / 4.0
            scores["avg_latency_ms"] = float(avg_latency_ms or 0.0)
            scores["total_cost_usd"] = 0.0
            scores["notes"] = ""
            return scores

        except Exception as e:
            self._logger.exception("RAGAS evaluation failed")
            return _zero(avg_latency_ms, notes=f"ragas_failed: {type(e).__name__}: {e}")
