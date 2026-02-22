# RAGAS-based evaluation (local Ollama judge)

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

FAST_TEST = os.environ.get("RAGSCOPE_FAST_TEST", "0") == "1"

DEFAULT_JUDGE_MODEL = "llama3.2"
DEFAULT_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def _get_chat_ollama(model: str, base_url: str):
    """Module-level so tests can patch it."""
    return ChatOllamaCommunity(
        model=model,
        base_url=base_url,
        temperature=0,
        model_kwargs={"format": "json"},
    )


def _get_embeddings(base_url: str):
    return HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDINGS_MODEL)


def _zero_metrics(avg_latency_ms: float = 0.0) -> Dict[str, float]:
    return {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_recall": 0.0,
        "context_precision": 0.0,
        "overall_score": 0.0,
        "avg_latency_ms": avg_latency_ms,
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
        avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

        if FAST_TEST:
            return _zero_metrics(avg_latency_ms)

        base_url = os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        judge_model = llm_model or os.environ.get("RAGAS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)

        try:
            judge_llm = _get_chat_ollama(judge_model, base_url)
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

            run_config = RunConfig(max_workers=1, timeout=120, max_retries=1)
            wrapped_llm = LangchainLLMWrapper(judge_llm)

            ds = Dataset.from_dict(
                {
                    "question": [r["question"] for r in pipeline_results],
                    "answer": [r["answer"] for r in pipeline_results],
                    "contexts": [r["contexts"] for r in pipeline_results],
                    "ground_truth": ground_truths,
                }
            )

            result = evaluate(
                dataset=ds,
                metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                llm=wrapped_llm,
                embeddings=embeddings,
                run_config=run_config,
            )

        except Exception:
            return _zero_metrics(avg_latency_ms)

        scores = {
            "faithfulness": float(getattr(result, "faithfulness", 0.0) or 0.0),
            "answer_relevancy": float(getattr(result, "answer_relevancy", 0.0) or 0.0),
            "context_recall": float(getattr(result, "context_recall", 0.0) or 0.0),
            "context_precision": float(getattr(result, "context_precision", 0.0) or 0.0),
        }

        overall = sum(scores.values()) / 4.0
        scores["overall_score"] = overall
        scores["avg_latency_ms"] = avg_latency_ms

        return scores

    def generate_report(self, results: Dict[str, float], config: ExperimentConfig) -> str:
        return f"Experiment: {config.experiment_name}"
