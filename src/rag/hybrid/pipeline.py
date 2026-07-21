from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from logger import get_logger
from src.adapters.llm import LLM
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.helpers import Document, normalize_documents
from src.rag.bm25 import BM25RAG
from src.rag.context import RAGContextBuilder
from src.rag.hybrid.fusion import reciprocal_rank_fusion
from src.rag.normal import NormalRAG
from src.rag.types import RAGAnswer, RAGConfig, RAGContext, RAGIndexingReport


class HybridRAG:
    """Hybrid retriever composed from NormalRAG + BM25RAG."""

    def __init__(
        self,
        *,
        normal_rag: NormalRAG,
        bm25_rag: BM25RAG,
        llm: Optional[LLM] = None,
        config: Optional[RAGConfig] = None,
        instrumentation: Optional[ComputeTracker] = None,
        rank_constant: int = 60,
    ) -> None:
        self.normal_rag = normal_rag
        self.bm25_rag = bm25_rag
        self.llm = llm or normal_rag.llm
        self.config = config or normal_rag.config
        self.instrumentation = instrumentation or NoopComputeTracker()
        self.rank_constant = rank_constant
        self.context_builder = RAGContextBuilder(max_chars=self.config.max_context_chars)
        self._logger = get_logger(__name__)

    def index_documents(self, documents: Iterable[Document | dict | str]) -> RAGIndexingReport:
        normalized_documents = normalize_documents(documents)
        with self.instrumentation.track("hybrid.index_normal", {"documents": len(normalized_documents)}):
            normal_report = self.normal_rag.index_documents(normalized_documents)
        with self.instrumentation.track("hybrid.index_bm25", {"documents": len(normalized_documents)}):
            bm25_report = self.bm25_rag.index_documents(normalized_documents)

        report = RAGIndexingReport(
            documents=len(normalized_documents),
            chunks=max(normal_report.chunks, bm25_report.chunks),
            vectors=normal_report.vectors,
            details={
                "normal": normal_report,
                "bm25": bm25_report,
            },
        )
        self._logger.info(
            "hybrid_rag indexed documents=%s chunks=%s vectors=%s",
            report.documents,
            report.chunks,
            report.vectors,
        )
        return report

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> RAGContext:
        limit = top_k or self.config.top_k
        with self.instrumentation.track("hybrid.retrieve_normal", {"top_k": limit}):
            normal_context = self.normal_rag.retrieve(question, top_k=limit)
        with self.instrumentation.track("hybrid.retrieve_bm25", {"top_k": limit}):
            bm25_context = self.bm25_rag.retrieve(question, top_k=limit)

        with self.instrumentation.track(
            "hybrid.fuse_results",
            {
                "normal_results": len(normal_context.results),
                "bm25_results": len(bm25_context.results),
            },
        ):
            fused_results = reciprocal_rank_fusion(
                [normal_context.results, bm25_context.results],
                limit=limit,
                rank_constant=self.rank_constant,
            )
        self.instrumentation.add_count("hybrid.retrieved", len(fused_results))

        with self.instrumentation.track("hybrid.build_context", {"results": len(fused_results)}):
            context_text = self.context_builder.build(fused_results, title="Hybrid evidence")

        return RAGContext(
            question=question,
            results=fused_results,
            context_text=context_text,
            retriever="hybrid",
            details={
                "top_k": limit,
                "normal": normal_context,
                "bm25": bm25_context,
            },
        )

    def answer(self, question: str, *, top_k: Optional[int] = None) -> RAGAnswer:
        context = self.retrieve(question, top_k=top_k)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context.context_text}\n\n"
            "Answer with a concise response grounded in the context."
        )
        with self.instrumentation.track("hybrid.generate_answer", {"context_chars": len(context.context_text)}):
            response = self.llm.generate(prompt, system_prompt=self.config.system_prompt)

        return RAGAnswer(
            question=question,
            answer=response.text,
            context=context,
            usage={
                "llm": response.usage,
                "hybrid": self.instrumentation.snapshot(),
                "normal": self.normal_rag.instrumentation.snapshot(),
                "bm25": self.bm25_rag.instrumentation.snapshot(),
            },
        )
