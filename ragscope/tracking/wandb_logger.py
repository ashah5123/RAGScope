# Weights & Biases logger for experiment metrics and artifacts.

import logging
import os
from typing import List, Optional

import wandb

from ragscope.configs.experiment_config import ExperimentResult

logger = logging.getLogger(__name__)


class WandbLogger:
    """
    Integrates RAGScope experiment results with Weights & Biases.
    Disables gracefully if WANDB_API_KEY is not set; never crashes the program.
    """

    def __init__(
        self,
        project_name: str = "ragscope",
        entity: Optional[str] = None,
    ) -> None:
        self.project_name = project_name
        self.entity = entity if entity is not None else os.environ.get("WANDB_ENTITY")
        self._disabled = not os.environ.get("WANDB_API_KEY")
        if self._disabled:
            print(
                "WandbLogger: WANDB_API_KEY is not set. W&B logging is disabled. "
                "Set WANDB_API_KEY to enable experiment tracking."
            )
            logger.warning("W&B logging disabled: WANDB_API_KEY not set")
        else:
            logger.info(
                "WandbLogger initialized: project=%s, entity=%s",
                self.project_name,
                self.entity,
            )

    def log_experiment(self, result: ExperimentResult) -> None:
        """Log a single experiment result to W&B. No-op if API key missing; never raises."""
        if self._disabled:
            logger.debug("Skipping W&B log (disabled): %s", result.experiment_id)
            return
        try:
            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=result.experiment_id,
                reinit=True,
            )
            if run is None:
                logger.warning("wandb.init returned None for %s", result.experiment_id)
                return

            c = result.config
            wandb.config.update({
                "experiment_name": c.experiment_name,
                "chunk_size": c.chunk_size,
                "chunk_overlap": c.chunk_overlap,
                "embedding_model": c.embedding_model,
                "retriever_type": c.retriever_type,
                "top_k": c.top_k,
                "llm_model": c.llm_model,
            })

            overall_score = (
                result.faithfulness
                + result.answer_relevancy
                + result.context_recall
                + result.context_precision
            ) / 4.0
            wandb.log({
                "faithfulness": result.faithfulness,
                "answer_relevancy": result.answer_relevancy,
                "context_recall": result.context_recall,
                "context_precision": result.context_precision,
                "overall_score": overall_score,
                "avg_latency_ms": result.avg_latency_ms,
                "total_cost_usd": result.total_cost_usd if result.total_cost_usd is not None else 0.0,
            })

            wandb.finish()
            logger.info("Logged experiment to W&B: %s", result.experiment_id)
        except Exception as e:
            logger.exception("W&B log failed for %s: %s", result.experiment_id, e)
            try:
                wandb.finish(quiet=True)
            except Exception:
                pass

    def log_batch(self, results: List[ExperimentResult]) -> None:
        """Log each experiment result to W&B; print confirmation after each."""
        if self._disabled:
            logger.debug("Skipping W&B batch log (disabled), %d results", len(results))
            return
        for i, result in enumerate(results, start=1):
            self.log_experiment(result)
            print(f"Logged experiment {i}/{len(results)} to W&B: {result.config.experiment_name}")
