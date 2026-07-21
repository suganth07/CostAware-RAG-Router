from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters.embedding import EmbeddingAdapter, EmbeddingResult
from src.adapters.embedding import LlamaCppEmbeddingAdapter, LlamaCppEmbeddingConfig
from src.adapters.llm import LLM, LLMResponse
from src.adapters.storage.graph_store import (
    GraphEdge,
    GraphNeighborhood,
    GraphNode,
    InMemoryGraphStore,
)
from src.adapters.storage.vector_store import (
    NumpyVectorStore,
    VectorDocument,
    VectorSearchResult,
)
from src.g_rag import Document, GraphRAG, GraphRAGConfig
from src.g_rag.ranking import ProductionEvidenceRanker


class _KeywordEmbedding(EmbeddingAdapter):
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float("movie" in lower),
                    float("director" in lower),
                    float("actor" in lower or "star" in lower),
                ]
            )
        return EmbeddingResult(vectors=vectors, model="keyword")


class _NoopLLM(LLM):
    def generate(self, prompt, *, system_prompt=None, **kwargs):
        return LLMResponse(text="ok", model="noop")


def test_numpy_vector_store_persists_and_reuses_exact_index(tmp_path) -> None:
    cache = tmp_path / "index"
    store = NumpyVectorStore(cache_directory=cache)
    store.upsert_many(
        [
            VectorDocument(id="a", vector=[1.0, 0.0], text="alpha"),
            VectorDocument(id="b", vector=[0.0, 1.0], text="beta"),
        ]
    )
    assert store.search([1.0, 0.0], limit=1)[0].document.id == "a"
    store.flush()

    restored = NumpyVectorStore(cache_directory=cache)
    assert restored.cache_loaded
    assert restored.existing_ids(["a", "missing"]) == {"a"}
    result = restored.search([0.0, 1.0], limit=1)[0]
    assert result.document.id == "b"
    assert result.score == pytest.approx(1.0)


def test_llama_cpp_concurrent_batches_preserve_input_order() -> None:
    class _FakeLlamaCpp(LlamaCppEmbeddingAdapter):
        def _post(self, path, payload):
            return {
                "data": [
                    {"index": index, "embedding": [float(value)]}
                    for index, value in enumerate(payload["input"])
                ],
                "usage": {"prompt_tokens": len(payload["input"])},
            }

    adapter = _FakeLlamaCpp(
        LlamaCppEmbeddingConfig(batch_size=2, max_concurrent_requests=3)
    )
    result = adapter.embed_texts(["0", "1", "2", "3", "4", "5"])

    assert result.vectors == [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]
    assert result.usage == {"prompt_eval_count": 6}


def test_production_ranker_deduplicates_and_interleaves_depths() -> None:
    nodes = [
        GraphNode(id="seed", properties={"name": "Seed Movie"}),
        GraphNode(id="actor", properties={"name": "Actor One"}),
        GraphNode(id="other", properties={"name": "Other Movie"}),
        GraphNode(id="director", properties={"name": "Director One"}),
    ]
    edges = [
        GraphEdge(id="depth-1", source_id="seed", target_id="actor", type="STARRED_ACTORS"),
        GraphEdge(id="depth-1-copy", source_id="seed", target_id="actor", type="STARRED_ACTORS"),
        GraphEdge(id="depth-2", source_id="other", target_id="actor", type="STARRED_ACTORS"),
        GraphEdge(id="depth-3", source_id="other", target_id="director", type="DIRECTED_BY"),
    ]
    ranked = ProductionEvidenceRanker().rank_graph(
        query="who directed movies with the same actor as [Seed Movie]",
        neighborhood=GraphNeighborhood(
            seed_ids=("seed",),
            nodes=nodes,
            edges=edges,
        ),
        seed_node_ids=["seed"],
        vector_results=[],
        max_depth=3,
        limit=3,
    )

    assert len(ranked.neighborhood.edges) == 3
    assert [ranked.edge_depths[edge.id] for edge in ranked.neighborhood.edges] == [1, 2, 3]
    assert {edge.id for edge in ranked.neighborhood.edges} & {"depth-1", "depth-1-copy"}


def test_native_graph_documents_are_hybrid_indexed_without_duplicate_graph(tmp_path) -> None:
    movie = GraphNode(id="a_movie", properties={"name": "A Movie"})
    director = GraphNode(id="a_director", properties={"name": "A Director"})
    directed = GraphEdge(
        id="directed",
        source_id="a_movie",
        target_id="a_director",
        type="DIRECTED_BY",
    )
    bundle = SimpleNamespace(
        documents=[
            Document(
                id="fact",
                text="A Movie directed by A Director.",
                metadata={"subject": "A Movie", "object": "A Director"},
            )
        ],
        graph_nodes=[movie, director],
        graph_edges=[directed],
    )
    graph = InMemoryGraphStore()
    pipeline = GraphRAG(
        graph_store=graph,
        vector_store=NumpyVectorStore(cache_directory=tmp_path / "vectors"),
        embedding_adapter=_KeywordEmbedding(),
        llm=_NoopLLM(),
        config=GraphRAGConfig(
            top_k=1,
            vector_candidate_k=1,
            graph_depth=1,
            graph_limit=10,
            ranked_graph_limit=10,
        ),
    )

    report = pipeline.index_dataset_bundle(bundle, include_documents=True)
    retrieved = pipeline.retrieve("who directed [A Movie]")

    assert report.graph_edges == 1
    assert report.vectors == report.embedded_vectors == 1
    assert [edge.id for edge in retrieved.graph_neighborhood.edges] == ["directed"]
    assert "A Director" in retrieved.context_text
    assert retrieved.details["retrieval_contract"] == "production_hybrid_graph_v1"
