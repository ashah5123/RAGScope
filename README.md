# RAGScope

Automated Benchmarking & Evaluation Framework for Retrieval-Augmented Generation Systems

---

## Executive Summary

Evaluating retrieval-augmented generation systems is difficult: quality depends on retrieval accuracy, chunking strategy, embedding choice, and the language model, and these factors interact in non-linear ways. Naive chatbot demos and ad-hoc prompts do not provide comparable metrics or reproducible baselines. Without controlled experiments, teams cannot isolate which configuration changes improve or degrade performance.

RAGScope addresses this by providing a systematic benchmarking pipeline. Users define experiment configurations (chunk size, retriever type, embedding model, LLM), run batch jobs over a fixed question set with ground truths, and obtain RAGAS-derived scores (faithfulness, answer relevancy, context recall, context precision) plus latency. All evaluation runs are local (Ollama as judge, no OpenAI dependency), and results are persisted so that leaderboards and trade-off analyses are reproducible.

The framework emphasizes engineering rigor: deterministic evaluation, modular retriever and pipeline components, and a clear separation between orchestration (ExperimentRunner), evaluation (RAGAS with local judge), and presentation (FastAPI + Streamlit dashboard). Reproducibility is achieved through versioned configs, saved result files, and documented run commands.

---

## System Architecture

```
Streamlit Dashboard
        |
        v
FastAPI Backend
        |
        v
ExperimentRunner
        |
        v
RAGPipeline
        |
        v
Retriever Layer (Dense / Sparse / Hybrid)
        |
        v
Qdrant + Ollama
```

- **Streamlit Dashboard:** Consumes the FastAPI JSON API to display leaderboard, per-experiment details, and latency-vs-score trade-offs. No direct access to data stores or runners.

- **FastAPI Backend:** Exposes `/experiments/run`, `/experiments/results`, `/experiments/leaderboard`, and `/experiments/{id}`. Validates requests, delegates execution to ExperimentRunner, and returns JSON-serializable records. Invalid or zero-score results are filtered from the leaderboard and optionally from results.

- **ExperimentRunner:** Orchestrates the full loop: for each ExperimentConfig, builds a RAGPipeline, runs the pipeline over all questions, invokes the RAGAS evaluator, aggregates metrics, and persists results to `experiments/results_<timestamp>.json` and CSV.

- **RAGPipeline:** Loads documents, chunks by configured size/overlap, builds the index in Qdrant (and optional sparse index for hybrid), and answers queries via the retriever plus LLM. Encapsulates indexing and query logic so the runner stays agnostic to retrieval implementation.

- **Retriever Layer:** Dense (vector-only), sparse (keyword/BM25-style), or hybrid (combined). Implemented via a factory; each retriever type uses the same Qdrant client and optional sparse backend. Embedding model is configurable (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2, nomic-embed-text).

- **Qdrant + Ollama:** Qdrant serves vector search; Ollama serves the generative LLM and the RAGAS judge model. No cloud APIs required for core benchmarking.

---

## Evaluation Methodology

RAGScope uses RAGAS-style metrics computed with a local judge (Ollama) and local embeddings:

- **Faithfulness:** Measures whether the generated answer is grounded in the retrieved context. The judge evaluates factual consistency between answer and context; scores are aggregated per run.

- **Answer Relevancy:** Measures how well the answer addresses the question. Typically computed via judge or embedding-based similarity between question and answer.

- **Context Recall:** Measures how much of the ground-truth information is present in the retrieved context. Higher values indicate the retriever is surfacing the right evidence.

- **Context Precision:** Measures whether the retrieved context is focused on what is needed to answer the question, reducing noise and irrelevant chunks.

**Overall score** is the arithmetic mean of these four metrics. This gives a single comparable number across experiments without weighting; teams can still inspect individual metrics for trade-offs (e.g. high recall but lower precision).

The **local judge** is an Ollama-served model (e.g. llama3.2). RAGAS uses this model for faithfulness and answer relevancy style judgments. Embedding-based similarity is used where applicable (e.g. context recall/precision). There is no OpenAI or other paid API dependency for evaluation.

---

## Experimental Design

Experiments sweep key parameters that affect RAG behavior:

- **Chunk size:** Smaller chunks (e.g. 128) can improve recall by aligning boundaries with answer-relevant spans but may fragment context and hurt faithfulness; larger chunks (e.g. 512) provide more context per retrieval but can dilute precision. Overlap is fixed per run to keep comparisons clean.

- **Retriever type:** Dense retrieval uses vector similarity only (embedding model + Qdrant). Sparse retrieval uses lexical/keyword matching. Hybrid combines both (e.g. reciprocal rank fusion) to improve recall and robustness to phrasing.

- **Embedding model:** Choices such as all-MiniLM-L6-v2 (faster, smaller) vs all-mpnet-base-v2 (higher quality, slower) affect retrieval quality and latency. Nomic-embed-text is supported for Ollama-hosted embeddings.

- **LLM model:** The generative model (e.g. llama3.2, mistral, gemma2) is fixed per experiment; the same judge model is used for fairness across runs.

The portfolio script runs a curated subset (e.g. 6 or 18 configurations) to balance coverage and runtime. Full grid sweeps are available for exhaustive comparison.

---

## Real Results

Representative non-zero results from a single run (format matches the API leaderboard). Columns: experiment name, chunk size, embedding model, retriever type, overall score (mean of four RAGAS metrics), and average latency in milliseconds.

| experiment_name        | chunk_size | embedding_model   | retriever_type | overall_score | avg_latency_ms |
|------------------------|------------|-------------------|----------------|---------------|----------------|
| chunk256_hybrid_mpnet  | 256        | all-mpnet-base-v2 | hybrid         | 0.905         | 420            |
| chunk512_hybrid_mpnet  | 512        | all-mpnet-base-v2 | hybrid         | 0.880         | 380            |
| chunk128_hybrid_MiniLM | 128        | all-MiniLM-L6-v2  | hybrid         | 0.858         | 350            |
| chunk256_dense_mpnet   | 256        | all-mpnet-base-v2 | dense          | 0.850         | 410            |
| chunk128_dense_MiniLM  | 128        | all-MiniLM-L6-v2  | dense          | 0.835         | 320            |
| chunk512_sparse_MiniLM | 512        | all-MiniLM-L6-v2  | sparse         | 0.803         | 180            |

---

## Dashboard

### Leaderboard

![Leaderboard](docs/assets/dashboard_leaderboard.png)

The leaderboard view lists experiments sorted by overall score (descending). Each row summarizes configuration (chunk size, embedding model, retriever type, LLM) and the four RAGAS metrics plus mean overall score and average latency. From an engineering perspective this allows quick comparison of which configurations meet quality thresholds and how retrieval and model choices interact.

### Latency vs Score Tradeoff

![Latency vs Score](docs/assets/latency_vs_score.png)

This plot shows average latency (ms) versus overall score. It highlights the cost-quality trade-off: higher-scoring setups often incur more latency (e.g. hybrid retrieval, larger embeddings). Teams can use it to choose configurations that meet both latency SLOs and minimum quality bars.

---

## Reproducibility

Exact steps to reproduce the environment and runs:

```bash
# Clone repository
git clone <repo_url>
cd RAGScope

# Virtual environment and package
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# Start Qdrant (Docker)
docker run -p 6333:6333 qdrant/qdrant

# Start Ollama (LLM and RAGAS judge)
ollama serve
ollama pull llama3.2

# Environment
export QDRANT_URL=http://localhost:6333
export OLLAMA_BASE_URL=http://localhost:11434

# Batch experiments (writes to experiments/)
python ragscope/scripts/run_portfolio_experiments.py --data_path ragscope/data/sample.txt
# Full grid: add --full

# API and dashboard
uvicorn ragscope.api.main:app --reload --host 0.0.0.0 --port 8000
# In another terminal:
streamlit run ragscope/dashboard/app.py
```

Point the dashboard at `http://localhost:8000` for the API. Results are under `experiments/` as `results_<timestamp>.json` and CSV.

---

## Engineering Highlights

- **Modular retriever factory:** Dense, sparse, and hybrid retrievers are registered in a single factory; the pipeline requests a retriever by type and config. New backends can be added without changing the runner or API.

- **Local-first architecture:** No OpenAI or paid APIs for core benchmarking. Ollama for generation and judge; Qdrant for vectors; HuggingFace or Ollama for embeddings.

- **Batch orchestration engine:** ExperimentRunner executes a list of ExperimentConfigs sequentially, reusing the same question set and ground truths. Each run produces one ExperimentResult with metrics and latency; results are appended to timestamped files.

- **Deterministic evaluation pipeline:** Fixed judge model, fixed evaluation code path, and persisted inputs/outputs. Filtering of invalid or zero-score results is explicit in the API so leaderboards reflect only valid runs.

- **CI-tested codebase:** Tests cover evaluator behavior, retriever factory, and API endpoints so that changes to metrics or configuration do not regress without detection.

---

## Roadmap

- **Larger benchmark datasets:** Move beyond small sample sets to standardized RAG benchmarks (e.g. BEIR-style or domain-specific QA) for more generalizable conclusions.

- **Multi-LLM evaluation:** Compare multiple judge and generator models in the same run to quantify model-dependent variance in scores.

- **Cost modeling:** Track token usage and approximate cost when using billable APIs (optional), and report cost-per-query alongside latency.

- **Distributed evaluation support:** Parallelize experiment runs across machines or processes to reduce wall-clock time for large parameter sweeps.
