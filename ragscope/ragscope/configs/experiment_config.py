# Pydantic models and loaders for experiment and run configuration.

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_name: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: Literal[
        "all-MiniLM-L6-v2",
        "nomic-embed-text",
        "all-mpnet-base-v2",
    ]
    retriever_type: Literal["dense", "sparse", "hybrid"]
    top_k: int = 5
    llm_model: Literal["llama3.2", "mistral", "gemma2"]
    dataset_path: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentResult(BaseModel):
    experiment_id: str
    config: ExperimentConfig
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    context_precision: float
    avg_latency_ms: float
    total_cost_usd: float | None = None
    timestamp: datetime
    notes: str = ""

    def to_dict(self) -> dict:
        """Return a flat dictionary of all fields for logging and dataframes."""
        config_flat = {
            f"config_{k}": (
                v.isoformat() if isinstance(v, datetime) else v
            )
            for k, v in self.config.model_dump().items()
        }
        return {
            "experiment_id": self.experiment_id,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "context_precision": self.context_precision,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
            **config_flat,
        }
