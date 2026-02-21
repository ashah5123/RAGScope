# Orchestrates running RAG experiments with config, pipeline, evaluator, and tracking.

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from ragscope.configs.experiment_config import ExperimentConfig, ExperimentResult
from ragscope.evaluators.ragas_evaluator import RAGASEvaluator
from ragscope.pipelines.rag_pipeline import RAGPipeline

if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO)


class ExperimentRunner:
    """Orchestrates running RAG experiments across configs with pipeline, evaluator, and result persistence."""

    def __init__(
        self,
        configs: List[ExperimentConfig],
        questions: List[str],
        ground_truths: List[str],
        data_path: str,
        results_dir: str = "experiments/",
    ) -> None:
        if len(questions) != len(ground_truths):
            raise ValueError(
                f"questions and ground_truths length mismatch: "
                f"{len(questions)} vs {len(ground_truths)}"
            )

        self.configs = configs
        self.questions = questions
        self.ground_truths = ground_truths
        self.data_path = data_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def run_single(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment for the given config; returns ExperimentResult or fallback on error."""
        self.logger.info(
            "Starting experiment: name=%s, id=%s",
            config.experiment_name,
            config.experiment_id,
        )
        try:
            pipeline = RAGPipeline(config)
            documents = pipeline.load_and_chunk_documents(self.data_path)
            pipeline.build_index(documents)
            pipeline_results = pipeline.run_batch(self.questions)

            latencies = [
                r.get("latency_ms")
                for r in pipeline_results
                if isinstance(r.get("latency_ms"), (int, float))
            ]
            avg_latency_ms = (
                sum(latencies) / len(latencies) if latencies else 0.0
            )

            evaluator = RAGASEvaluator()
            metrics = evaluator.evaluate(pipeline_results, self.ground_truths)

            return ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                faithfulness=metrics["faithfulness"],
                answer_relevancy=metrics["answer_relevancy"],
                context_recall=metrics["context_recall"],
                context_precision=metrics["context_precision"],
                avg_latency_ms=avg_latency_ms,
                total_cost_usd=None,
                timestamp=datetime.now(timezone.utc),
                notes="",
            )
        except Exception as e:
            self.logger.exception(
                "Experiment failed: name=%s, id=%s",
                config.experiment_name,
                config.experiment_id,
            )
            return ExperimentResult(
                experiment_id=config.experiment_id,
                config=config,
                faithfulness=0.0,
                answer_relevancy=0.0,
                context_recall=0.0,
                context_precision=0.0,
                avg_latency_ms=0.0,
                total_cost_usd=None,
                timestamp=datetime.now(timezone.utc),
                notes=str(e),
            )

    def run_all(self) -> List[ExperimentResult]:
        """Run all configured experiments and return a list of ExperimentResult."""
        n = len(self.configs)
        self.logger.info("Starting batch run: %d experiment(s)", n)
        results: List[ExperimentResult] = []
        for i, config in enumerate(self.configs, start=1):
            print(f"Running experiment {i}/{n}: {config.experiment_name}")
            results.append(self.run_single(config))
        self.logger.info("Batch run complete: %d result(s)", len(results))
        return results

    def save_results(self, results: List[ExperimentResult]) -> None:
        """Save results to JSON and CSV in self.results_dir. Logs paths; swallows errors."""
        try:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
            data = [r.to_dict() for r in results]

            json_path = self.results_dir / f"results_{timestamp_str}.json"
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            self.logger.info("Saved results JSON: %s", json_path)

            csv_path = self.results_dir / f"results_{timestamp_str}.csv"
            pd.DataFrame(data).to_csv(csv_path, index=False)
            self.logger.info("Saved results CSV: %s", csv_path)
        except Exception:
            self.logger.exception("Failed to save results")

    def get_leaderboard(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Return a DataFrame of results sorted by overall_score, with columns in fixed order."""
        if not results:
            return pd.DataFrame(
                columns=[
                    "experiment_name", "chunk_size", "embedding_model", "retriever_type",
                    "llm_model", "faithfulness", "answer_relevancy", "context_recall",
                    "context_precision", "overall_score", "avg_latency_ms",
                ]
            )
        df = pd.DataFrame([r.to_dict() for r in results])

        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        if "overall_score" not in df.columns:
            present = [c for c in metric_cols if c in df.columns]
            df["overall_score"] = df[present].mean(axis=1) if present else 0.0

        config_map = {
            "experiment_name": "config_experiment_name",
            "chunk_size": "config_chunk_size",
            "embedding_model": "config_embedding_model",
            "retriever_type": "config_retriever_type",
            "llm_model": "config_llm_model",
        }
        out = pd.DataFrame()
        for out_col in [
            "experiment_name", "chunk_size", "embedding_model", "retriever_type", "llm_model",
            "faithfulness", "answer_relevancy", "context_recall", "context_precision",
            "overall_score", "avg_latency_ms",
        ]:
            src = config_map.get(out_col, out_col)
            if src in df.columns:
                out[out_col] = df[src]
            elif out_col in df.columns:
                out[out_col] = df[out_col]
            else:
                out[out_col] = None
        return out.sort_values("overall_score", ascending=False).reset_index(drop=True)
