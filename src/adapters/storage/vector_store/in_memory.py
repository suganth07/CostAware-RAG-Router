from __future__ import annotations

import math
from typing import Any, Optional

from logger import get_logger
from src.adapters.storage.vector_store.base import (
    VectorDocument,
    VectorSearchResult,
    VectorStore,
)


class InMemoryVectorStore(VectorStore):
    """Simple cosine-similarity vector store for local development."""

    def __init__(self) -> None:
        self._documents: dict[str, VectorDocument] = {}
        self._dimension: Optional[int] = None
        self._logger = get_logger(__name__)

    def upsert(self, document: VectorDocument) -> None:
        self._validate_dimension(document.vector)
        self._documents[document.id] = document

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        self._validate_dimension(query_vector)
        results: list[VectorSearchResult] = []

        for document in self._documents.values():
            if filters and not _matches_filters(document.metadata, filters):
                continue
            score = _cosine_similarity(query_vector, document.vector)
            results.append(VectorSearchResult(document=document, score=score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:limit]

    def get(self, document_id: str) -> Optional[VectorDocument]:
        return self._documents.get(str(document_id))

    def delete(self, document_id: str) -> None:
        self._documents.pop(str(document_id), None)

    def clear(self) -> None:
        self._documents.clear()
        self._dimension = None

    def _validate_dimension(self, vector: list[float]) -> None:
        if not vector:
            raise ValueError("Vector cannot be empty")
        if self._dimension is None:
            self._dimension = len(vector)
            return
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {len(vector)}"
            )


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
