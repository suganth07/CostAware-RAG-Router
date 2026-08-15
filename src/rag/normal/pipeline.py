from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Optional

from logger import get_logger
from src.adapters.embedding import EmbeddingAdapter
from src.adapters.llm import LLM
from src.adapters.storage.vector_store import VectorDocument, VectorStore
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.helpers import Document, TextChunker, normalize_documents
from src.rag.context import RAGContextBuilder
from src.rag.types import RAGAnswer, RAGConfig, RAGContext, RAGIndexingReport, RetrievalResult


class NormalRAG:
    """Classic vector RAG: chunk, embed, vector search, generate."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        embedding_adapter: EmbeddingAdapter,
        llm: LLM,
        config: Optional[RAGConfig] = None,
        instrumentation: Optional[ComputeTracker] = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_adapter = embedding_adapter
        self.llm = llm
        self.config = config or RAGConfig()
        self.instrumentation = instrumentation or NoopComputeTracker()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.context_builder = RAGContextBuilder(max_chars=self.config.max_context_chars)
        self._logger = get_logger(__name__)

    def index_documents(self, documents: Iterable[Document | dict | str]) -> RAGIndexingReport:
        normalized_documents = normalize_documents(documents)
        report = RAGIndexingReport(documents=len(normalized_documents))
        if not normalized_documents:
            return report

        with self.instrumentation.track("normal.chunk_documents", {"documents": len(normalized_documents)}):
            chunks = self.chunker.chunk_documents(normalized_documents)
        report.chunks = len(chunks)
        self.instrumentation.add_count("normal.chunks", len(chunks))

        if not chunks:
            return report

        index_batch_size = max(1, int(self.config.index_batch_size))
        started = time.perf_counter()
        self._logger.info(
            "normal_rag indexing embeddings chunks=%s batches=%s batch_size=%s",
            len(chunks),
            (len(chunks) + index_batch_size - 1) // index_batch_size,
            index_batch_size,
        )
        for batch_number, start in enumerate(range(0, len(chunks), index_batch_size), start=1):
            batch = chunks[start : start + index_batch_size]
            with self.instrumentation.track("normal.embed_chunks", {"chunks": len(batch), "batch": batch_number}):
                embedding_result = self.embedding_adapter.embed_texts([chunk.text for chunk in batch])
            vector_documents = [
                VectorDocument(
                    id=chunk.id,
                    vector=vector,
                    text=chunk.text,
                    metadata={
                        **chunk.metadata,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.index,
                    },
                )
                for chunk, vector in zip(batch, embedding_result.vectors)
            ]
            with self.instrumentation.track("normal.upsert_vectors", {"vectors": len(vector_documents), "batch": batch_number}):
                self.vector_store.upsert_many(vector_documents)
            report.vectors += len(vector_documents)
            if batch_number == 1 or start + len(batch) == len(chunks) or batch_number % 25 == 0:
                self._logger.info(
                    "normal_rag indexing progress batches=%s/%s vectors=%s cache_hits=%s cache_misses=%s elapsed_s=%.2f",
                    batch_number,
                    (len(chunks) + index_batch_size - 1) // index_batch_size,
                    report.vectors,
                    embedding_result.usage.get("embedding_cache_hits", 0),
                    embedding_result.usage.get("embedding_cache_misses", 0),
                    time.perf_counter() - started,
                )
        self.instrumentation.add_count("normal.vectors", report.vectors)
        self._logger.info(
            "normal_rag indexed documents=%s chunks=%s vectors=%s",
            report.documents,
            report.chunks,
            report.vectors,
        )
        return report

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> RAGContext:
        limit = top_k or self.config.top_k
        with self.instrumentation.track("normal.embed_query", {"chars": len(question)}):
            query_vector = self.embedding_adapter.embed_text(question)

        with self.instrumentation.track("normal.vector_search", {"top_k": limit}):
            vector_results = self.vector_store.search(query_vector, limit=limit)

        results = [
            RetrievalResult(
                id=result.document.id,
                text=result.document.text,
                score=result.score,
                metadata=result.document.metadata,
                source="vector",
            )
            for result in vector_results
        ]
        self.instrumentation.add_count("normal.retrieved", len(results))

        with self.instrumentation.track("normal.build_context", {"results": len(results)}):
            context_text = self.context_builder.build(results, title="Vector evidence")

        return RAGContext(
            question=question,
            results=results,
            context_text=context_text,
            retriever="normal",
            details={"top_k": limit},
        )

    def answer(self, question: str, *, top_k: Optional[int] = None) -> RAGAnswer:
        context = self.retrieve(question, top_k=top_k)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context.context_text}\n\n"
            "Answer with a concise response grounded in the context."
        )
        with self.instrumentation.track("normal.generate_answer", {"context_chars": len(context.context_text)}):
            response = self.llm.generate(prompt, system_prompt=self.config.system_prompt)

        return RAGAnswer(
            question=question,
            answer=response.text,
            context=context,
            usage={
                "llm": response.usage,
                "instrumentation": self.instrumentation.snapshot(),
            },
        )
