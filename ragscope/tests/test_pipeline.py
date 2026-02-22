# Tests for RAG pipeline behavior and outputs.

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.documents import Document

from ragscope.configs.experiment_config import ExperimentConfig
from ragscope.pipelines.rag_pipeline import RAGPipeline


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """ExperimentConfig with required fields set; defaults for the rest."""
    return ExperimentConfig(
        experiment_name="test_run",
        embedding_model="all-MiniLM-L6-v2",
        retriever_type="dense",
        llm_model="llama3.2",
        dataset_path="/tmp/data.txt",
    )


@pytest.fixture
def temp_txt_file(tmp_path: Path) -> Path:
    """A temporary .txt file with sample content for chunking."""
    p = tmp_path / "sample.txt"
    p.write_text(
        "Python is a programming language created by Guido van Rossum. "
        "It was first released in 1991. Python emphasizes code readability."
    )
    return p


def test_experiment_config_initializes_with_defaults(sample_config: ExperimentConfig) -> None:
    """ExperimentConfig has expected defaults when only required fields are set."""
    assert sample_config.experiment_name == "test_run"
    assert sample_config.chunk_size == 512
    assert sample_config.chunk_overlap == 50
    assert sample_config.top_k == 5
    assert sample_config.llm_model == "llama3.2"
    assert sample_config.embedding_model == "all-MiniLM-L6-v2"
    assert sample_config.retriever_type == "dense"
    assert sample_config.experiment_id != ""
    assert sample_config.created_at is not None


def test_load_and_chunk_documents_chunks_sample_file(
    sample_config: ExperimentConfig,
    temp_txt_file: Path,
) -> None:
    """RAGPipeline.load_and_chunk_documents() returns a list of Document chunks."""
    pipeline = RAGPipeline(sample_config)
    chunks = pipeline.load_and_chunk_documents(str(temp_txt_file))
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, Document)
        assert isinstance(c.page_content, str)
        assert len(c.page_content) > 0


def test_query_returns_expected_dict_keys(
    sample_config: ExperimentConfig,
) -> None:
    """RAGPipeline.query() returns dict with question, answer, contexts, latency_ms (mocked)."""
    fake_docs = [
        Document(page_content="context one"),
        Document(page_content="context two"),
    ]
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = fake_docs

    fake_response = MagicMock()
    fake_response.content = "deterministic answer"

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_response

    with (
        patch("ragscope.pipelines.rag_pipeline.ChatOllama", return_value=fake_llm),
        patch(
            "ragscope.pipelines.rag_pipeline.RetrieverFactory.get_retriever",
            return_value=fake_retriever,
        ),
    ):
        pipeline = RAGPipeline(sample_config)
        pipeline.build_index(fake_docs)
        result = pipeline.query("What is Python?")

    assert result["question"] == "What is Python?"
    assert result["answer"] == "deterministic answer"
    assert result["contexts"] == ["context one", "context two"]
    assert "latency_ms" in result
    assert isinstance(result["latency_ms"], (int, float))
