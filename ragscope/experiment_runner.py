from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ragscope.configs.experiment_config import ExperimentConfig
from ragscope.evaluators.ragas_evaluator import RAGASEvaluator
from ragscope.pipelines.rag_pipeline import RAGPipeline


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _jsonable(x: Any) -> Any:
    # Make nested objects JSON-safe (datetime, pydantic, dataclasses, etc.)
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        return _jsonable(x.model_dump())
    if is_dataclass(x):
        return _jsonable(asdict(x))
    return str(x)


class ExperimentRunner:
    """
    Runs one or more ExperimentConfig configs against a fixed benchmark (questions + ground_truths),
    persists results to experiments/results_<timestamp>.json and .csv, and returns the in-memory results.
    """

    def __init__(
        self,
        configs: List[ExperimentConfig],
        questions: List[str],
        ground_truths: List[str],
        data_path: str,
        results_dir: str = "experiments",
    ) -> None:
        self.configs = configs
        self.questions = questions
        self.ground_truths = ground_truths
        self.data_path = data_path
        self.results_dir = results_dir

        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

    def run_single(self, config: ExperimentConfig) -> Dict[str, Any]:
        pipeline = RAGPipeline(config)

        documents = pipeline.load_and_chunk_documents(self.data_path)
        pipeline.build_index(documents)

        pipeline_results = pipeline.run_batch(self.questions)

        evaluator = RAGASEvaluator()
        metrics = evaluator.evaluate(
            pipeline_results=pipeline_results,
            ground_truths=self.ground_truths,
            llm_model=None,
        )

        faith = float(metrics.get("faithfulness", 0.0) or 0.0)
        rel = float(metrics.get("answer_relevancy", 0.0) or 0.0)
        rec = float(metrics.get("context_recall", 0.0) or 0.0)
        prec = float(metrics.get("context_precision", 0.0) or 0.0)
        overall = (faith + rel + rec + prec) / 4.0

        row: Dict[str, Any] = {
            "experiment_id": getattr(config, "experiment_id", None),
            "experiment_name": getattr(config, "experiment_name", None),
            "chunk_size": getattr(config, "chunk_size", None),
            "chunk_overlap": getattr(config, "chunk_overlap", None),
            "embedding_model": getattr(config, "embedding_model", None),
            "retriever_type": getattr(config, "retriever_type", None),
            "top_k": getattr(config, "top_k", None),
            "llm_model": getattr(config, "llm_model", None),
            "dataset_path": getattr(config, "dataset_path", None),
            "faithfulness": faith,
            "answer_relevancy": rel,
            "context_recall": rec,
            "context_precision": prec,
            "overall_score": overall,
            "avg_latency_ms": float(metrics.get("avg_latency_ms", 0.0) or 0.0),
            "total_cost_usd": float(metrics.get("total_cost_usd", 0.0) or 0.0),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notes": metrics.get("notes", "") or "",
        }

        return _jsonable(row)

    def run_batch(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for cfg in self.configs:
            results.append(self.run_single(cfg))
        self.save_results(results)
        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        ts = _utc_ts()
        json_path = Path(self.results_dir) / f"results_{ts}.json"
        csv_path = Path(self.results_dir) / f"results_{ts}.csv"

        json_path.write_text(json.dumps(results, indent=2))

        # Stable CSV columns for your leaderboard/dashboard
        fieldnames = [
            "experiment_id",
            "experiment_name",
            "chunk_size",
            "chunk_overlap",
            "embedding_model",
            "retriever_type",
            "top_k",
            "llm_model",
            "dataset_path",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "overall_score",
            "avg_latency_ms",
            "total_cost_usd",
            "timestamp",
            "notes",
        ]

        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k) for k in fieldnames})

        # Also keep a convenient latest leaderboard file
        latest_path = Path(self.results_dir) / "latest_leaderboard.csv"
        latest_path.write_text(csv_path.read_text())

        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")
