import logging
import os
from typing import List, Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ragscope.configs.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


class RetrieverFactoryError(Exception):
    pass


class DenseQdrantRetriever(BaseRetriever):
    client: Any
    collection_name: str
    embeddings: Any
    top_k: int

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vec = self.embeddings.embed_query(query)

        # qdrant-client API compatibility:
        # - some versions expose client.search(...)
        # - newer versions use client.query_points(...)
        results = None

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query=query_vec,
                limit=self.top_k,
                with_payload=True,
            )
            points = results  # list of ScoredPoint
        else:
            # Try query_points with common signatures
            try:
                qp = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vec,
                    limit=self.top_k,
                    with_payload=True,
                )
            except TypeError:
                # Alternate signature: query=QueryVector(...)
                qp = self.client.query_points(
                    collection_name=self.collection_name,
                    query=qmodels.QueryVector(vector=query_vec),
                    limit=self.top_k,
                    with_payload=True,
                )

            # Different client builds return different shapes
            points = getattr(qp, "points", None)
            if points is None:
                points = getattr(qp, "result", None)
            if points is None:
                points = qp

        docs: List[Document] = []
        for p in points:
            payload = getattr(p, "payload", None) or {}
            text = payload.get("text", "")
            meta = payload.get("metadata", {}) or {}
            docs.append(Document(page_content=text, metadata=meta))
        return docs


class SimpleHybridRetriever(BaseRetriever):
    dense: Any
    sparse: Any
    top_k: int
    weights: Any = (0.6, 0.4)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        dense_k = max(1, int(round(self.top_k * self.weights[0])))
        sparse_k = max(1, self.top_k - dense_k)

        dense_docs = self.dense.invoke(query)[:dense_k]
        sparse_docs = self.sparse.invoke(query)[:sparse_k]

        seen = set()
        merged: List[Document] = []
        for d in dense_docs + sparse_docs:
            key = (d.page_content or "").strip()
            if key and key not in seen:
                merged.append(d)
                seen.add(key)

        if len(merged) < self.top_k:
            for d in self.dense.invoke(query)[dense_k:] + self.sparse.invoke(query)[sparse_k:]:
                key = (d.page_content or "").strip()
                if key and key not in seen:
                    merged.append(d)
                    seen.add(key)
                if len(merged) >= self.top_k:
                    break

        return merged[: self.top_k]


class RetrieverFactory:
    @staticmethod
    def _get_embedding_model(model_name: str) -> Embeddings:
        if model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
            return HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device":"cpu"})

        if model_name == "nomic-embed-text":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)

        raise RetrieverFactoryError(
            f"Unsupported embedding model: {model_name}. "
            "Supported: all-MiniLM-L6-v2, all-mpnet-base-v2, nomic-embed-text."
        )

    @staticmethod
    def get_retriever(config: ExperimentConfig, documents: List[Document]) -> BaseRetriever:
        if not documents:
            raise RetrieverFactoryError("documents list cannot be empty.")

        k = config.top_k
        collection_name = f"ragscope_{config.experiment_id}"

        if config.retriever_type == "dense":
            return RetrieverFactory._get_dense_retriever(config, documents, k, collection_name)

        if config.retriever_type == "sparse":
            return RetrieverFactory._get_sparse_retriever(documents, k)

        if config.retriever_type == "hybrid":
            return RetrieverFactory._get_hybrid_retriever(config, documents, k, collection_name)

        raise RetrieverFactoryError(f"Unsupported retriever_type: {config.retriever_type}")

    @staticmethod
    def _get_qdrant_client() -> QdrantClient:
        url = os.environ.get("QDRANT_URL", "").strip()
        api_key = os.environ.get("QDRANT_API_KEY", "").strip() or None
        if not url:
            raise RetrieverFactoryError("QDRANT_URL is required (e.g. http://localhost:6333).")
        return QdrantClient(url=url, api_key=api_key, timeout=int(os.environ.get('QDRANT_TIMEOUT_S', '180')))

    @staticmethod
    def _ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
        try:
            client.get_collection(collection_name)
            return
        except Exception:
            logger.info("Creating Qdrant collection=%s size=%s", collection_name, vector_size)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    @staticmethod
    def _index_documents(
        client: QdrantClient,
        collection_name: str,
        embeddings: Embeddings,
        documents: List[Document],
    ) -> None:
        points: List[qmodels.PointStruct] = []
        for d in documents:
            text = d.page_content or ""
            if not text.strip():
                continue
            vec = embeddings.embed_query(text)
            points.append(
                qmodels.PointStruct(
                    id=str(uuid4()),
                    vector=vec,
                    payload={"text": text, "metadata": d.metadata or {}},
                )
            )

        if not points:
            raise RetrieverFactoryError("No non-empty documents to index into Qdrant.")

        client.upsert(collection_name=collection_name, points=points)

    @staticmethod
    def _get_dense_retriever(
        config: ExperimentConfig,
        documents: List[Document],
        k: int,
        collection_name: str,
    ) -> BaseRetriever:
        client = RetrieverFactory._get_qdrant_client()
        embeddings = RetrieverFactory._get_embedding_model(config.embedding_model)

        dim = len(embeddings.embed_query("dimension_probe"))
        RetrieverFactory._ensure_collection(client, collection_name, dim)
        RetrieverFactory._index_documents(client, collection_name, embeddings, documents)

        return DenseQdrantRetriever(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            top_k=k,
        )

    @staticmethod
    def _get_sparse_retriever(documents: List[Document], k: int) -> BaseRetriever:
        retriever = BM25Retriever.from_documents(documents=documents)
        retriever.k = k
        return retriever

    @staticmethod
    def _get_hybrid_retriever(
        config: ExperimentConfig,
        documents: List[Document],
        k: int,
        collection_name: str,
    ) -> BaseRetriever:
        dense = RetrieverFactory._get_dense_retriever(config, documents, k, collection_name)
        sparse = RetrieverFactory._get_sparse_retriever(documents, k)
        return SimpleHybridRetriever(dense=dense, sparse=sparse, top_k=k, weights=(0.6, 0.4))
