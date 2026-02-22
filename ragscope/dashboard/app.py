# Dashboard app for visualizing RAG experiment results and comparisons.

import logging
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="RAGScope",
    layout="wide",
    page_icon="🔬",
)

DEFAULT_API_URL = "http://localhost:8000"


def fetch_health(api_url: str) -> bool:
    """Return True if /health succeeds."""
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return r.status_code == 200
    except Exception as e:
        logger.warning("Health check failed for %s: %s", api_url, e)
        return False


@st.cache_data(ttl=30)
def fetch_leaderboard(api_url: str) -> list[dict]:
    """Fetch leaderboard from API. Cache key includes api_url."""
    try:
        r = requests.get(f"{api_url.rstrip('/')}/experiments/leaderboard", timeout=10)
        r.raise_for_status()
        return r.json() if r.json() is not None else []
    except Exception as e:
        logger.warning("Fetch leaderboard failed: %s", e)
        return []


@st.cache_data(ttl=30)
def fetch_results(api_url: str) -> list[dict]:
    """Fetch all results from API. Cache key includes api_url."""
    try:
        r = requests.get(f"{api_url.rstrip('/')}/experiments/results", timeout=10)
        r.raise_for_status()
        return r.json() if r.json() is not None else []
    except Exception as e:
        logger.warning("Fetch results failed: %s", e)
        return []


def _fmt(val: Any) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.4f}" if isinstance(val, float) else str(val)
    return str(val)


# --- Sidebar ---
with st.sidebar:
    st.subheader("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API_URL, key="api_url")
    if st.button("🔄 Refresh Data"):
        fetch_leaderboard.clear()
        fetch_results.clear()
        st.rerun()
    st.divider()
    connected = fetch_health(api_url)
    if connected:
        st.success("Connected")
    else:
        st.warning("Cannot reach API. Check URL and that the server is running.")

# --- Header ---
st.title("🔬 RAGScope")
st.caption("Automated Benchmarking & Evaluation Framework for RAG Pipelines")
st.divider()

# Fetch data (cached)
leaderboard = fetch_leaderboard(api_url)
results = fetch_results(api_url)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Leaderboard",
    "📊 Compare Experiments",
    "⚡ Latency vs Score",
    "📋 Experiment Details",
])

with tab1:
    if not leaderboard:
        st.info("No experiments yet. Run one from the API.")
    else:
        df = pd.DataFrame(leaderboard)
        if "overall_score" not in df.columns and all(c in df.columns for c in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]):
            df["overall_score"] = df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean(axis=1)
        df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)
        display_cols = [c for c in ["experiment_name", "chunk_size", "embedding_model", "retriever_type", "llm_model", "faithfulness", "answer_relevancy", "context_recall", "context_precision", "overall_score", "avg_latency_ms"] if c in df.columns]
        def highlight_top_row(row):
            return ["background-color: #ffd700; font-weight: bold" if row.name == 0 else "" for _ in row]
        st.dataframe(
            df[display_cols].style.apply(highlight_top_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    if not leaderboard:
        st.info("No data yet. Run experiments from the API.")
    else:
        df = pd.DataFrame(leaderboard)
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
        present_metrics = [c for c in metric_cols if c in df.columns]
        if not present_metrics:
            st.info("No metric columns found in leaderboard.")
        else:
            df_melt = df.copy()
            if "experiment_name" in df_melt.columns:
                df_melt["_name"] = df_melt["experiment_name"]
            elif "experiment_id" in df_melt.columns:
                df_melt["_name"] = df_melt["experiment_id"]
            else:
                df_melt["_name"] = df_melt.index.astype(str)
            color_col = "retriever_type" if "retriever_type" in df.columns else None
            id_vars = ["_name"] + ([color_col] if color_col else [])
            long = df_melt.melt(id_vars=id_vars, value_vars=present_metrics, var_name="metric", value_name="score")
            fig = px.bar(long, x="_name", y="score", color=color_col, barmode="group", text_auto=".2f",
                         title="Metrics by Experiment", labels={"score": "Score", "_name": "Experiment"})
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if not leaderboard:
        st.info("No data yet. Run experiments from the API.")
    else:
        df = pd.DataFrame(leaderboard)
        if "overall_score" not in df.columns and all(c in df.columns for c in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]):
            df["overall_score"] = df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean(axis=1)
        need = ["avg_latency_ms", "overall_score"]
        if not all(c in df.columns for c in need):
            st.info("Leaderboard missing avg_latency_ms or overall_score.")
        else:
            df = df.dropna(subset=need)
            name_col = "experiment_name" if "experiment_name" in df.columns else ("experiment_id" if "experiment_id" in df.columns else None)
            hover_data = {}
            if "retriever_type" in df.columns:
                hover_data["retriever_type"] = True
            if "embedding_model" in df.columns:
                hover_data["embedding_model"] = True
            if "llm_model" in df.columns:
                hover_data["llm_model"] = True
            fig = px.scatter(
                df, x="avg_latency_ms", y="overall_score",
                hover_name=name_col,
                hover_data=hover_data if hover_data else None,
                title="Latency vs Overall Score",
                labels={"avg_latency_ms": "Avg Latency (ms)", "overall_score": "Overall Score"},
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    if not results and not leaderboard:
        st.info("No experiments yet. Run one from the API.")
    else:
        # Prefer full results for details (have config_*); fallback to leaderboard
        options = []
        if results:
            for r in results:
                name = r.get("config_experiment_name") or r.get("experiment_name") or r.get("experiment_id", "")
                options.append((name, r.get("experiment_id", name), r))
        elif leaderboard:
            for r in leaderboard:
                name = r.get("experiment_name") or r.get("experiment_id", "")
                options.append((name, r.get("experiment_id", name), r))
        if not options:
            st.info("No experiment records to show.")
        else:
            names = [o[0] for o in options]
            selected_name = st.selectbox("Select experiment", options=names, key="detail_select")
            rec = next(r for n, _, r in options if n == selected_name)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Config")
                st.metric("Chunk size", rec.get("config_chunk_size") or rec.get("chunk_size", "—"))
                st.metric("Chunk overlap", rec.get("config_chunk_overlap", "—"))
                st.metric("Embedding model", rec.get("config_embedding_model") or rec.get("embedding_model", "—"))
                st.metric("Retriever type", rec.get("config_retriever_type") or rec.get("retriever_type", "—"))
                st.metric("Top K", rec.get("config_top_k") or rec.get("top_k", "—"))
                st.metric("LLM model", rec.get("config_llm_model") or rec.get("llm_model", "—"))
            with c2:
                st.subheader("Metrics")
                st.metric("Faithfulness", _fmt(rec.get("faithfulness")))
                st.metric("Answer relevancy", _fmt(rec.get("answer_relevancy")))
                st.metric("Context recall", _fmt(rec.get("context_recall")))
                st.metric("Context precision", _fmt(rec.get("context_precision")))
                overall = rec.get("overall_score")
                if overall is None and all(k in rec for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]):
                    overall = (rec["faithfulness"] + rec["answer_relevancy"] + rec["context_recall"] + rec["context_precision"]) / 4
                st.metric("Overall score", _fmt(overall))
                st.metric("Avg latency (ms)", _fmt(rec.get("avg_latency_ms")))
                st.divider()
                if rec.get("notes"):
                    st.caption("Notes: " + str(rec["notes"]))
