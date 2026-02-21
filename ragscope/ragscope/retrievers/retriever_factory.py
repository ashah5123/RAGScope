# Factory for creating and configuring retrievers (e.g. Qdrant, in-memory).

import logging
import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings

from ragscope.configs.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


class RetrieverFactoryError(Exception):
    """Raised when retriever creation fails."""

    pass


class RetrieverFactory:
    """Factory for creating retrievers from ExperimentConfig and documents."""

    @staticmethod
    def _get_embedding_model(model_name: str) -> Embeddings:
        """Return the appropriate LangChain embedding model for the given name."""
        if model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
            logger.info("Using HuggingFace embeddings: model_name=%s", model_name)
            return HuggingFaceEmbeddings(model_name=model_name)
        if model_name == "nomic-embed-text":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info(
                "Using Ollama embeddings: model=nomic-embed-text, base_url=%s",
                base_url,
            )
            return OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
        raise RetrieverFactoryError(
            f"Unsupported embedding model: {model_name!r}. "
            "Supported: all-MiniLM-L6-v2, all-mpnet-base-v2, nomic-embed-text."
        )

    @staticmethod
    def get_retriever(
        config: ExperimentConfig,
        documents: list[Document],
    ) -> BaseRetriever:
        """
        Return a retriever based on config.retriever_type.

        - dense: Qdrant vector store with config.embedding_model
        - sparse: BM25Retriever
        - hybrid: EnsembleRetriever (dense 0.6, sparse 0.4)
        """
        if not documents:
            raise RetrieverFactoryError("documents list cannot be empty.")

        retriever_type = config.retriever_type
        k = config.top_k
        collection_name = f"ragscope_{config.experiment_id}"

        if retriever_type == "dense":
            return RetrieverFactory._get_dense_retriever(
                config=config,
                documents=documents,
                k=k,
                collection_name=collection_name,
            )
        if retriever_type == "sparse":
            return RetrieverFactory._get_sparse_retriever(
                documents=documents,
                k=k,
            )
        if retriever_type == "hybrid":
            return RetrieverFactory._get_hybrid_retriever(
                config=config,
                documents=documents,
                k=k,
                collection_name=collection_name,
            )

        raise RetrieverFactoryError(
            f"Unsupported retriever_type: {retriever_type}. "
            "Supported: dense, sparse, hybrid."
        )

    @staticmethod
    def _get_dense_retriever(
        config: ExperimentConfig,
        documents: list[Document],
        k: int,
        collection_name: str,
    ) -> BaseRetriever:
        """Build Qdrant vector store retriever using config embedding model."""
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url:
            raise RetrieverFactoryError(
                "QDRANT_URL environment variable is required for dense retriever. "
                "Set QDRANT_URL to your Qdrant server URL (e.g. http://localhost:6333)."
            )
        if not api_key:
            logger.info(
                "QDRANT_API_KEY not set; connecting to Qdrant without API key (local or unauthenticated)."
            )

        try:
            embeddings = RetrieverFactory._get_embedding_model(config.embedding_model)
        except RetrieverFactoryError:
            raise
        except Exception as e:
            logger.exception("Failed to create embedding model for dense retriever")
            raise RetrieverFactoryError(
                f"Embedding model error: {e!r}. Check embedding_model and env (e.g. OLLAMA_BASE_URL for nomic-embed-text)."
            ) from e

        try:
            vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                url=url,
                api_key=api_key or None,
                collection_name=collection_name,
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            logger.info(
                "Created dense retriever (Qdrant) collection=%s, k=%s, url=%s",
                collection_name,
                k,
                url,
            )
            return retriever
        except Exception as e:
            logger.exception("Failed to create Qdrant vector store")
            raise RetrieverFactoryError(
                f"Qdrant error: {e!r}. Check QDRANT_URL (and QDRANT_API_KEY if required)."
            ) from e

    @staticmethod
    def _get_sparse_retriever(
        documents: list[Document],
        k: int,
    ) -> BaseRetriever:
        """Build BM25 retriever."""
        try:
            retriever = BM25Retriever.from_documents(documents=documents, k=k)
            logger.info("Created sparse retriever (BM25) with k=%s", k)
            return retriever
        except Exception as e:
            logger.exception("Failed to create BM25 retriever")
            raise RetrieverFactoryError(f"BM25 error: {e}") from e

    @staticmethod
    def _get_hybrid_retriever(
        config: ExperimentConfig,
        documents: list[Document],
        k: int,
        collection_name: str,
    ) -> BaseRetriever:
        """Build EnsembleRetriever with dense (0.6) and sparse (0.4)."""
        try:
            dense = RetrieverFactory._get_dense_retriever(
                config=config,
                documents=documents,
                k=k,
                collection_name=collection_name,
            )
            sparse = RetrieverFactory._get_sparse_retriever(documents=documents, k=k)
            retriever = EnsembleRetriever(
                retrievers=[dense, sparse],
                weights=[0.6, 0.4],
            )
            logger.info(
                "Created hybrid retriever (EnsembleRetriever) weights=[0.6, 0.4], k=%s",
                k,
            )
            return retriever
        except RetrieverFactoryError:
            raise
        except Exception as e:
            logger.exception("Failed to create hybrid retriever")
            raise RetrieverFactoryError(f"Hybrid retriever error: {e}") from e
