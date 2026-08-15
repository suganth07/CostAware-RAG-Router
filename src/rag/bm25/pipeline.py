from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Optional

from logger import get_logger
from src.adapters.llm import LLM
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.helpers import Document, TextChunker, normalize_documents
from src.rag.bm25.index import BM25Document, BM25Index
from src.rag.context import RAGContextBuilder
from src.rag.types import RAGAnswer, RAGConfig, RAGContext, RAGIndexingReport


class BM25RAG:
    """Lexical RAG using a local Okapi BM25 index."""

    def __init__(
        self,
        *,
        llm: LLM,
        index: Optional[BM25Index] = None,
        config: Optional[RAGConfig] = None,
        instrumentation: Optional[ComputeTracker] = None,
    ) -> None:
        self.llm = llm
        self.index = index or BM25Index()
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

        started = time.perf_counter()
        self._logger.info("bm25_rag indexing started documents=%s", len(normalized_documents))
        with self.instrumentation.track("bm25.chunk_documents", {"documents": len(normalized_documents)}):
            chunks = self.chunker.chunk_documents(normalized_documents)
        report.chunks = len(chunks)
        self.instrumentation.add_count("bm25.chunks", len(chunks))

        bm25_documents = [
            BM25Document(
                id=chunk.id,
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.index,
                },
            )
            for chunk in chunks
        ]
        with self.instrumentation.track("bm25.upsert_index", {"chunks": len(bm25_documents)}):
            self.index.upsert_many(bm25_documents)

        self._logger.info(
            "bm25_rag indexed documents=%s chunks=%s elapsed_s=%.2f",
            report.documents,
            report.chunks,
            time.perf_counter() - started,
        )
        return report

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> RAGContext:
        limit = top_k or self.config.top_k
        with self.instrumentation.track("bm25.search", {"top_k": limit, "chars": len(question)}):
            results = self.index.search(question, limit=limit)
        self.instrumentation.add_count("bm25.retrieved", len(results))

        with self.instrumentation.track("bm25.build_context", {"results": len(results)}):
            context_text = self.context_builder.build(results, title="BM25 evidence")

        return RAGContext(
            question=question,
            results=results,
            context_text=context_text,
            retriever="bm25",
            details={"top_k": limit},
        )

    def answer(self, question: str, *, top_k: Optional[int] = None) -> RAGAnswer:
        context = self.retrieve(question, top_k=top_k)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context.context_text}\n\n"
            "Answer with a concise response grounded in the context."
        )
        with self.instrumentation.track("bm25.generate_answer", {"context_chars": len(context.context_text)}):
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
