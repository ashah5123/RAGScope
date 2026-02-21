# RAGScope

Automated benchmarking and evaluation framework for RAG pipelines.

## Setup

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and `WANDB_API_KEY`.
2. Install: `pip install -e .`

## Structure

- `ragscope/` — Core package (pipelines, evaluators, retrievers, configs, tracking).
- `api/` — FastAPI app for runs and results.
- `dashboard/` — Dashboard app for result visualization.
- `tests/` — Test suite.
- `notebooks/` — Analysis notebooks.
- `data/raw/`, `data/processed/` — Datasets.
- `experiments/` — Experiment configs and outputs.
- `docs/` — Documentation.
