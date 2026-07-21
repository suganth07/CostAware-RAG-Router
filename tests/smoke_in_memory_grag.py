from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult
from src.adapters.llm.adapter import LLM, LLMResponse
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.g_rag import Document, GraphRAG, InMemoryInstrumentation


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
                    float(lower.count("neo4j")),
                ]
            )
        return EmbeddingResult(vectors=vectors, model="fake-embedding")


class FakeLLM(LLM):
    def generate(self, prompt, *, system_prompt=None, **kwargs):
        return LLMResponse(
            text="Ada is connected to graph retrieval in the supplied context.",
            model="fake-llm",
        )


def main() -> None:
    rag = GraphRAG(
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
        embedding_adapter=FakeEmbedding(),
        llm=FakeLLM(),
        instrumentation=InMemoryInstrumentation(log_stages=False),
    )

    report = rag.index_documents(
        [
            Document(
                id="d1",
                text="Ada Lovelace studied Graph Retrieval. Neo4j stores graph relationships.",
            ),
            Document(id="d2", text="Normal RAG retrieves chunks with vectors."),
        ]
    )
    answer = rag.answer("How is Ada connected to graph retrieval?")

    assert report.documents == 2
    assert report.chunks == 2
    assert report.vectors == 2
    assert report.graph_nodes > 0
    assert answer.answer
    assert answer.context.vector_results
    assert answer.usage["instrumentation"]["metrics"]


if __name__ == "__main__":
    main()
