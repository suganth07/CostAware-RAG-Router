from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult
from src.adapters.llm.adapter import LLM, LLMResponse
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.compute_eval import InMemoryComputeTracker
from src.helpers import Document
from src.rag import BM25RAG, HybridRAG, NormalRAG


class FakeEmbedding(EmbeddingAdapter):
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float(len(text)),
                    float(lower.count("ada")),
                    float(lower.count("graph")),
                    float(lower.count("rag")),
                    float(lower.count("bm25")),
                ]
            )
        return EmbeddingResult(vectors=vectors, model="fake-embedding")


class FakeLLM(LLM):
    def generate(self, prompt, *, system_prompt=None, **kwargs):
        return LLMResponse(
            text="The answer is grounded in the retrieved context.",
            model="fake-llm",
        )


def build_documents():
    return [
        Document(
            id="d1",
            text="Ada Lovelace is useful evidence for graph RAG examples.",
            metadata={"dataset": "smoke"},
        ),
        Document(
            id="d2",
            text="BM25 retrieval is lexical and normal RAG retrieval is vector based.",
            metadata={"dataset": "smoke"},
        ),
    ]


def main() -> None:
    llm = FakeLLM()
    embedding = FakeEmbedding()
    question = "How do graph RAG and BM25 retrieval relate to Ada?"

    normal = NormalRAG(
        vector_store=InMemoryVectorStore(),
        embedding_adapter=embedding,
        llm=llm,
        instrumentation=InMemoryComputeTracker(log_stages=False),
    )
    normal_report = normal.index_documents(build_documents())
    normal_answer = normal.answer(question)
    assert normal_report.vectors == 2
    assert normal_answer.context.results
    assert normal_answer.context.retriever == "normal"

    bm25 = BM25RAG(
        llm=llm,
        instrumentation=InMemoryComputeTracker(log_stages=False),
    )
    bm25_report = bm25.index_documents(build_documents())
    bm25_answer = bm25.answer(question)
    assert bm25_report.chunks == 2
    assert bm25_answer.context.results
    assert bm25_answer.context.retriever == "bm25"

    hybrid_normal = NormalRAG(
        vector_store=InMemoryVectorStore(),
        embedding_adapter=embedding,
        llm=llm,
        instrumentation=InMemoryComputeTracker(log_stages=False),
    )
    hybrid_bm25 = BM25RAG(
        llm=llm,
        instrumentation=InMemoryComputeTracker(log_stages=False),
    )
    hybrid = HybridRAG(
        normal_rag=hybrid_normal,
        bm25_rag=hybrid_bm25,
        instrumentation=InMemoryComputeTracker(log_stages=False),
    )
    hybrid_report = hybrid.index_documents(build_documents())
    hybrid_answer = hybrid.answer(question)
    assert hybrid_report.vectors == 2
    assert hybrid_answer.context.results
    assert hybrid_answer.context.retriever == "hybrid"
    assert hybrid_answer.usage["normal"]["metrics"]
    assert hybrid_answer.usage["bm25"]["metrics"]
    assert hybrid_answer.usage["hybrid"]["metrics"]


if __name__ == "__main__":
    main()
