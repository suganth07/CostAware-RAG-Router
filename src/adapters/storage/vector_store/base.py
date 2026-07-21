from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


@dataclass(slots=True)
class VectorDocument:
    id: str
    vector: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.vector = [float(value) for value in self.vector]
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class VectorSearchResult:
    document: VectorDocument
    score: float


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, document: VectorDocument) -> None:
        raise NotImplementedError

    def upsert_many(self, documents: Iterable[VectorDocument]) -> None:
        for document in documents:
            self.upsert(document)

    def existing_ids(self, document_ids: Iterable[str]) -> set[str]:
        """Return IDs already present in the store.

        Remote and batch-oriented stores should override this method.  The
        conservative default keeps the storage interface backwards compatible
        while allowing indexing pipelines to avoid re-embedding persisted
        documents.
        """
        return {
            document_id
            for document_id in (str(value) for value in document_ids)
            if self.get(document_id) is not None
        }

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: str) -> Optional[VectorDocument]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None

    def flush(self) -> None:
        """Persist buffered writes when the adapter supports it."""
        return None
