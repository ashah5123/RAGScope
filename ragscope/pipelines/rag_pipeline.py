# Core RAG pipeline interface and default implementation.

import logging
import os
import time
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.chat_models import ChatOllama

from ragscope.configs.experiment_config import ExperimentConfig
from ragscope.retrievers.retriever_factory import RetrieverFactory

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".pdf", ".csv")
PROMPT_TEMPLATE = (
    "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
)


class RAGPipeline:
    """
    Production RAG pipeline using only free/local components: Ollama LLM,
    RetrieverFactory (dense/sparse/hybrid), and RecursiveCharacterTextSplitter.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.retriever = None
        self._logger = logging.getLogger(f"{__name__}.RAGPipeline")
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = ChatOllama(
            model=config.llm_model,
            base_url=base_url,
        )
        self._logger.info(
            "RAGPipeline initialized with llm_model=%s, base_url=%s",
            config.llm_model,
            base_url,
        )

    def load_and_chunk_documents(self, file_path: str) -> List[Document]:
        """
        Load documents from a file (.txt, .pdf, or .csv) and chunk them
        using RecursiveCharacterTextSplitter with config chunk_size and chunk_overlap.
        """
        file_path = os.path.abspath(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        self._logger.info("Loading documents from path=%s", file_path)
        if ext == ".txt":
            loader = TextLoader(file_path)
        elif ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = CSVLoader(file_path)

        raw_docs = loader.load()
        self._logger.info("Loaded %d raw document(s) from %s", len(raw_docs), file_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(raw_docs)
        self._logger.info(
            "Chunked into %d chunk(s) (chunk_size=%s, chunk_overlap=%s)",
            len(chunks),
            self.config.chunk_size,
            self.config.chunk_overlap,
        )
        return chunks

    def build_index(self, documents: List[Document]) -> None:
        """
        Build the retriever index from the given documents and store it in self.retriever.
        """
        if not documents:
            self._logger.warning("build_index called with empty documents list")
        self._logger.info(
            "Building index with retriever_type=%s for %d document(s)",
            self.config.retriever_type,
            len(documents),
        )
        self.retriever = RetrieverFactory.get_retriever(self.config, documents)
        self._logger.info("Index built successfully")

    def query(self, question: str) -> dict:
        """
        Run a single RAG query: retrieve top_k docs, build context, invoke LLM,
        and return question, answer, contexts, and latency_ms.
        """
        if self.retriever is None:
            raise RuntimeError(
                "Retriever is not built. Call build_index(documents) before query(question)."
            )

        self._logger.info("Query: retrieving for question (top_k=%s)", self.config.top_k)
        docs = self.retriever.invoke(question)
        docs = docs[: self.config.top_k]
        self._logger.info("Retrieved %d document(s)", len(docs))

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        self._logger.info("Invoking LLM for generation")
        start = time.time()
        response = self.llm.invoke(prompt)
        latency_ms = (time.time() - start) * 1000
        self._logger.info("Generation completed in %.2f ms", latency_ms)

        answer_text = response.content if hasattr(response, "content") else str(response)
        return {
            "question": question,
            "answer": answer_text,
            "contexts": [doc.page_content for doc in docs],
            "latency_ms": latency_ms,
        }

    def run_batch(self, questions: List[str]) -> List[dict]:
        """
        Run query(question) for each question and return a list of result dicts.
        """
        results = []
        for i, q in enumerate(questions):
            self._logger.info("Batch query %d/%d", i + 1, len(questions))
            results.append(self.query(q))
        return results
