# RAGScope — Automated Benchmarking & Evaluation Framework for RAG Pipelines

RAGScope runs reproducible RAG experiments, evaluates them with RAGAS metrics, and surfaces results in a dashboard. The stack is **local and free**: Ollama for the LLM and judge, Qdrant for vector search, and optional W&B for tracking.

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Dashboard      │────▶│  API (FastAPI)  │────▶│  ExperimentRunner│
│  (Streamlit)    │     │  /experiments/* │     │  run_all / save   │
└─────────────────┘     └─────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Qdrant          │◀───▶│  RAGPipeline    │◀────│  RAGASEvaluator  │
│  (vector store)  │     │  load / index   │     │  (local judge)    │
└─────────────────┘     │  query / batch  │     └──────────────────┘
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  Ollama (LLM +   │
                        │  judge model)    │
                        └──────────────────┘
```

---

## Quickstart (local, free)

1. **Create a venv and install the package**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -e .
   ```

2. **Start Qdrant** (Docker)
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Start Ollama** (LLM + RAGAS judge)
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

4. **Set environment variables**
   ```bash
   export QDRANT_URL=http://localhost:6333
   export OLLAMA_BASE_URL=http://localhost:11434
   ```

5. **Run the API**
   ```bash
   uvicorn ragscope.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Run the dashboard**
   ```bash
   streamlit run ragscope/dashboard/app.py
   ```
   Open the URL shown (e.g. http://localhost:8501) and point the API URL to `http://localhost:8000`.

---

## Run portfolio experiments

From the repo root (with `QDRANT_URL` and `OLLAMA_BASE_URL` set):

```bash
python ragscope/scripts/run_portfolio_experiments.py --data_path ragscope/data/sample.txt
```

- **Default:** 6 experiments (curated chunk size × retriever × embedding combos).
- **Full grid:** add `--full` for 18 runs (3 chunk sizes × 3 retriever types × 2 embedding models).

Results are written to `experiments/` as:
- `results_<timestamp>.json` — full result records.
- `results_<timestamp>.csv` — same data for analysis.

---

## Sample results

Example leaderboard (columns match the dashboard):

| experiment_name        | chunk_size | embedding_model   | retriever_type | llm_model | faithfulness | answer_relevancy | context_recall | context_precision | overall_score | avg_latency_ms |
|------------------------|------------|-------------------|----------------|-----------|--------------|------------------|---------------|-------------------|---------------|----------------|
| chunk256_hybrid_mpnet  | 256        | all-mpnet-base-v2 | hybrid         | llama3.2  | 0.89         | 0.91             | 0.94          | 0.88              | 0.905         | 420             |
| chunk512_hybrid_mpnet  | 512        | all-mpnet-base-v2 | hybrid         | llama3.2  | 0.87         | 0.88             | 0.92          | 0.85              | 0.880         | 380             |
| chunk128_hybrid_MiniLM | 128        | all-MiniLM-L6-v2  | hybrid         | llama3.2  | 0.85         | 0.86             | 0.90          | 0.82              | 0.858         | 350             |
| chunk256_dense_mpnet   | 256        | all-mpnet-base-v2 | dense          | llama3.2  | 0.84         | 0.87             | 0.88          | 0.81              | 0.850         | 410             |
| chunk128_dense_MiniLM  | 128        | all-MiniLM-L6-v2  | dense          | llama3.2  | 0.82         | 0.84             | 0.89          | 0.79              | 0.835         | 320             |
| chunk512_sparse_MiniLM | 512        | all-MiniLM-L6-v2  | sparse         | llama3.2  | 0.79         | 0.81             | 0.85          | 0.76              | 0.803         | 180             |

---

## Findings

- **Hybrid retrieval** (dense + sparse) often improves context recall and context precision over dense-only or sparse-only, at the cost of slightly higher latency.
- **Smaller chunk sizes** can improve recall by surfacing more precise snippets, but may reduce faithfulness if the model over-relies on fragmented context.
- **Larger embedding models** (e.g. all-mpnet-base-v2 vs all-MiniLM-L6-v2) can improve answer relevancy and precision but tend to increase indexing and retrieval latency.

---

## Dashboard screenshot

![Dashboard Screenshot](docs/dashboard.png)

*(Add `docs/dashboard.png` to show the Streamlit leaderboard and comparison views.)*

---

## Project structure

| Path              | Description                                      |
|-------------------|--------------------------------------------------|
| `ragscope/`       | Core package: pipelines, evaluators, retrievers, configs, tracking |
| `ragscope/api/`   | FastAPI app for running experiments and serving results |
| `ragscope/dashboard/` | Streamlit dashboard (leaderboard, compare, latency vs score, details) |
| `ragscope/scripts/`   | Portfolio experiment runner                      |
| `ragscope/data/`  | Sample data and processed outputs                |
| `experiments/`    | Saved result JSON/CSV files                      |
| `docs/`           | Documentation and assets                         |
