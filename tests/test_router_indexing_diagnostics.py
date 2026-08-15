from __future__ import annotations

from src.adapters.embedding import CachingEmbeddingAdapter, EmbeddingAdapter, EmbeddingResult
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.helpers import Document
from src.rag.normal import NormalRAG
from src.rag.types import RAGConfig


class CountingEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return EmbeddingResult(
            vectors=[[float(index + 1), 1.0] for index, _ in enumerate(texts)],
            model="test",
        )


def test_cache_inspection_never_calls_delegate(tmp_path) -> None:
    delegate = CountingEmbedding()
    cache = CachingEmbeddingAdapter(
        delegate,
        cache_path=tmp_path / "embeddings.sqlite3",
        namespace="test",
    )
    cache.embed_texts(["already cached"])
    inspection = cache.inspect_texts(["already cached", "missing", "already cached"])

    assert delegate.calls == 1
    assert inspection == {
        "texts": 3,
        "unique_texts": 2,
        "cached_unique_texts": 1,
        "missing_unique_texts": 1,
        "cached_texts": 2,
        "missing_texts": 1,
        "cache_hit_rate": 2 / 3,
    }


def test_normal_rag_indexes_in_cache_checkpoint_batches(tmp_path) -> None:
    delegate = CountingEmbedding()
    cache = CachingEmbeddingAdapter(
        delegate,
        cache_path=tmp_path / "embeddings.sqlite3",
        namespace="test",
    )
    rag = NormalRAG(
        vector_store=InMemoryVectorStore(),
        embedding_adapter=cache,
        llm=object(),
        config=RAGConfig(index_batch_size=1),
    )
    report = rag.index_documents([
        Document(id="one", text="first document"),
        Document(id="two", text="second document"),
    ])

    assert report.vectors == 2
    assert delegate.calls == 2
    assert cache.inspect_texts(["first document", "second document"])["missing_texts"] == 0
