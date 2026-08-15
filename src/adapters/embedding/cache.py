from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Sequence

from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult
from logger import get_logger


def local_embedding_cache_namespace(model_name: str) -> str:
    """Identify local embedding semantics independently of API transport."""
    return f"local:{str(model_name)}:mean_pooling_v1"


class CachingEmbeddingAdapter(EmbeddingAdapter):
    """Persistent content-addressed cache around any embedding adapter."""

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        *,
        cache_path: str | Path = "artifacts/cache/embeddings.sqlite3",
        namespace: str,
        model_name: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.cache_path = Path(cache_path)
        self.namespace = str(namespace)
        self.model_name = model_name or self.namespace
        self._write_lock = threading.Lock()
        self._logger = get_logger(__name__)
        self._initialize()

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        values = [str(text) for text in texts]
        if not values:
            return EmbeddingResult(vectors=[], model=self.model_name)

        keys = [_text_key(text) for text in values]
        cached = self._load_vectors(set(keys))
        cached_before = set(cached)
        missing_by_key: dict[str, str] = {}
        for key, value in zip(keys, values):
            if key not in cached:
                missing_by_key.setdefault(key, value)

        cache_hits = sum(key in cached_before for key in keys)
        self._logger.info(
            "embedding cache lookup namespace=%s texts=%s unique=%s hits=%s misses=%s",
            self.namespace,
            len(values),
            len(set(keys)),
            cache_hits,
            len(missing_by_key),
        )
        delegate_result: EmbeddingResult | None = None
        if missing_by_key:
            missing_keys = list(missing_by_key)
            started = time.perf_counter()
            delegate_result = self.adapter.embed_texts(
                [missing_by_key[key] for key in missing_keys]
            )
            if len(delegate_result.vectors) != len(missing_keys):
                raise ValueError(
                    "Embedding cache delegate returned a different vector count "
                    f"({len(delegate_result.vectors)} != {len(missing_keys)})"
                )
            fresh = {
                key: [float(value) for value in vector]
                for key, vector in zip(missing_keys, delegate_result.vectors)
            }
            self._store_vectors(fresh)
            cached.update(fresh)
            self._logger.info(
                "embedding cache stored namespace=%s fresh=%s elapsed_s=%.2f",
                self.namespace,
                len(fresh),
                time.perf_counter() - started,
            )

        vectors = [cached[key] for key in keys]
        usage = dict(delegate_result.usage) if delegate_result is not None else {}
        usage.update(
            {
                "embedding_cache_hits": cache_hits,
                "embedding_cache_misses": len(missing_by_key),
            }
        )
        return EmbeddingResult(
            vectors=vectors,
            model=delegate_result.model if delegate_result is not None else self.model_name,
            raw={
                "cache": str(self.cache_path),
                "namespace": self.namespace,
            },
            usage=usage,
        )

    def cache_summary(self) -> dict[str, int | str]:
        """Return namespace-level cache counts without calling the embedder."""
        with sqlite3.connect(self.cache_path) as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
        return {
            "cache_path": str(self.cache_path),
            "namespace": self.namespace,
            "entries": int(row[0]) if row else 0,
        }

    def inspect_texts(self, texts: Sequence[str]) -> dict[str, int | float]:
        """Measure cache coverage without generating any embeddings."""
        values = [str(text) for text in texts]
        keys = [_text_key(text) for text in values]
        unique_keys = set(keys)
        cached = self._load_vectors(unique_keys)
        unique_hits = len(cached)
        text_hits = sum(key in cached for key in keys)
        return {
            "texts": len(values),
            "unique_texts": len(unique_keys),
            "cached_unique_texts": unique_hits,
            "missing_unique_texts": len(unique_keys) - unique_hits,
            "cached_texts": text_hits,
            "missing_texts": len(values) - text_hits,
            "cache_hit_rate": text_hits / len(values) if values else 1.0,
        }

    def _initialize(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    namespace TEXT NOT NULL,
                    text_key TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY (namespace, text_key)
                )
                """
            )

    def _load_vectors(self, keys: set[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        result: dict[str, list[float]] = {}
        ordered = sorted(keys)
        with sqlite3.connect(self.cache_path) as connection:
            for start in range(0, len(ordered), 800):
                batch = ordered[start : start + 800]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT text_key, vector_json FROM embeddings "
                    f"WHERE namespace = ? AND text_key IN ({placeholders})",
                    [self.namespace, *batch],
                )
                for key, vector_json in rows:
                    result[str(key)] = [float(value) for value in json.loads(vector_json)]
        return result

    def _store_vectors(self, vectors: dict[str, list[float]]) -> None:
        if not vectors:
            return
        rows = [
            (self.namespace, key, json.dumps(vector, separators=(",", ":")))
            for key, vector in vectors.items()
        ]
        with self._write_lock, sqlite3.connect(self.cache_path) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(namespace, text_key, vector_json) "
                "VALUES (?, ?, ?)",
                rows,
            )


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
