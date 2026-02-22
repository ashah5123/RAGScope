# FastAPI app entrypoint and route definitions for runs and results.

import glob
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ragscope.configs.experiment_config import ExperimentConfig, ExperimentResult
from ragscope.experiment_runner import ExperimentRunner

logger = logging.getLogger(__name__)

app = FastAPI(title="RAGScope API", version="0.1.0")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / 'experiments'

def _resolve_results_dir(results_dir: str | None) -> Path:
    # If caller passes absolute path, use it. Otherwise resolve relative to repo root.
    if results_dir:
        rd = _resolve_results_dir(results_dir)
        return rd if rd.is_absolute() else (PROJECT_ROOT / rd)
    return DEFAULT_RESULTS_DIR


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_RESULTS_DIR = "experiments/"

METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")


def is_valid_result(r: ExperimentResult | dict) -> bool:
    """Return False if overall_score is missing/<=0 or all four metrics are zero."""
    if isinstance(r, dict):
        overall = r.get("overall_score")
        if overall is None:
            vals = [r.get(k) for k in METRIC_KEYS]
            if None in vals:
                return False
            overall = sum(vals) / 4
        if overall is None or overall <= 0:
            return False
        if all(r.get(k) == 0 for k in METRIC_KEYS):
            return False
        return True
    # ExperimentResult: no overall_score field, compute from metrics
    metrics = (r.faithfulness, r.answer_relevancy, r.context_recall, r.context_precision)
    if all(m == 0 for m in metrics):
        return False
    overall = sum(metrics) / 4
    return overall > 0


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
    results_dir: str = "experiments/"  # kept for request model default


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
async def get_results(
    results_dir: Optional[str] = None,
    include_invalid: bool = False,
) -> List[dict]:
    """Return combined list of all saved experiment result dicts from results_dir."""
    dir_path = results_dir or DEFAULT_RESULTS_DIR
    logger.info("GET /experiments/results dir=%s include_invalid=%s", dir_path, include_invalid)
    combined = _load_all_results(dir_path)
    total = len(combined)
    if not include_invalid:
        combined = [r for r in combined if is_valid_result(r)]
        filtered = total - len(combined)
        logger.info("Results: loaded=%d filtered_out=%d returned=%d", total, filtered, len(combined))
    else:
        logger.info("Results: loaded=%d returned=%d (include_invalid=True)", total, total)
    return combined


@app.get("/experiments/leaderboard")
async def get_leaderboard(results_dir: Optional[str] = None) -> List[dict]:
    """Return leaderboard (list of dicts) sorted by overall_score descending. Invalid results are excluded."""
    dir_path = results_dir or DEFAULT_RESULTS_DIR
    logger.info("GET /experiments/leaderboard dir=%s", dir_path)
    combined = _load_all_results(dir_path)
    total = len(combined)
    combined = [r for r in combined if is_valid_result(r)]
    filtered = total - len(combined)
    logger.info("Leaderboard: loaded=%d filtered_out=%d returned=%d", total, filtered, len(combined))
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