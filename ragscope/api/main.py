# FastAPI app entrypoint and route definitions for runs and results.

import glob
import json
import logging
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ragscope.configs.experiment_config import ExperimentConfig, ExperimentResult
from ragscope.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

app = FastAPI(title="RAGScope API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_RESULTS_DIR = "experiments/"


def _load_all_results(results_dir: str) -> List[dict]:
    """Load and combine all results_*.json files from results_dir. Returns list of dicts."""
    path = os.path.join(results_dir, "results_*.json")
    files = sorted(glob.glob(path))
    combined: List[dict] = []
    for f in files:
        try:
            with open(f) as fp:
                data = json.load(fp)
            if isinstance(data, list):
                combined.extend(data)
            else:
                combined.append(data)
        except Exception as e:
            logger.warning("Failed to load %s: %s", f, e)
    return combined


class RunExperimentsRequest(BaseModel):
    configs: List[ExperimentConfig]
    questions: List[str]
    ground_truths: List[str]
    data_path: str
    results_dir: str = "experiments/"


@app.get("/health")
async def health() -> dict:
    """Health check."""
    logger.info("GET /health")
    return {"status": "ok", "version": "0.1.0"}


@app.post("/experiments/run")
async def run_experiments(body: RunExperimentsRequest) -> List[dict]:
    """Run experiments, save results, return leaderboard as JSON records."""
    logger.info("POST /experiments/run: %d configs", len(body.configs))
    if len(body.configs) == 0:
        raise HTTPException(status_code=400, detail="configs must be non-empty")
    if len(body.questions) != len(body.ground_truths):
        raise HTTPException(
            status_code=400,
            detail=f"questions and ground_truths length mismatch: {len(body.questions)} vs {len(body.ground_truths)}",
        )
    try:
        runner = ExperimentRunner(
            configs=body.configs,
            questions=body.questions,
            ground_truths=body.ground_truths,
            data_path=body.data_path,
            results_dir=body.results_dir,
        )
        results = runner.run_all()
        runner.save_results(results)
        df = runner.get_leaderboard(results)
        records = df.to_dict(orient="records")
        logger.info("Experiments run complete: %d results", len(records))
        return records
    except Exception as e:
        logger.exception("Experiment run failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/experiments/results")
async def get_results(results_dir: Optional[str] = None) -> List[dict]:
    """Return combined list of all saved experiment result dicts from results_dir."""
    dir_path = results_dir or DEFAULT_RESULTS_DIR
    logger.info("GET /experiments/results dir=%s", dir_path)
    combined = _load_all_results(dir_path)
    return combined


@app.get("/experiments/leaderboard")
async def get_leaderboard(results_dir: Optional[str] = None) -> List[dict]:
    """Return leaderboard (list of dicts) sorted by overall_score descending."""
    dir_path = results_dir or DEFAULT_RESULTS_DIR
    logger.info("GET /experiments/leaderboard dir=%s", dir_path)
    combined = _load_all_results(dir_path)
    if not combined:
        return []
    df = pd.DataFrame(combined)
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
    out_cols = [
        "experiment_name", "chunk_size", "embedding_model", "retriever_type", "llm_model",
        "faithfulness", "answer_relevancy", "context_recall", "context_precision",
        "overall_score", "avg_latency_ms",
    ]
    out = pd.DataFrame()
    for col in out_cols:
        src = config_map.get(col, col)
        if src in df.columns:
            out[col] = df[src]
        elif col in df.columns:
            out[col] = df[col]
        else:
            out[col] = None
    out = out.sort_values("overall_score", ascending=False).reset_index(drop=True)
    return out.to_dict(orient="records")


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, results_dir: Optional[str] = None) -> dict:
    """Return full result dict for the given experiment_id. 404 if not found."""
    dir_path = results_dir or DEFAULT_RESULTS_DIR
    logger.info("GET /experiments/%s", experiment_id)
    combined = _load_all_results(dir_path)
    for rec in combined:
        if rec.get("experiment_id") == experiment_id:
            return rec
    raise HTTPException(status_code=404, detail=f"Experiment {experiment_id!r} not found")
