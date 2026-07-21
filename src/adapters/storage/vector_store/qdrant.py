from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from logger import get_logger
from src.adapters.storage.vector_store.base import (
    VectorDocument,
    VectorSearchResult,
    VectorStore,
)


@dataclass(slots=True)
class QdrantConfig:
    collection_name: str
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    vector_size: Optional[int] = None
    distance: str = "Cosine"
    recreate_collection: bool = False


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store using qdrant-client."""

    def __init__(self, config: QdrantConfig) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ImportError(
                "QdrantVectorStore requires 'qdrant-client'. Install it with: pip install qdrant-client"
            ) from exc

        self.config = config
        self._models = models
        self._client = QdrantClient(url=config.url, api_key=config.api_key)
        self._logger = get_logger(__name__)

        if config.recreate_collection:
            self._client.delete_collection(collection_name=config.collection_name)
        if config.vector_size is not None:
            self._ensure_collection(config.vector_size)

    def upsert(self, document: VectorDocument) -> None:
        self._ensure_collection(len(document.vector))
        point = self._models.PointStruct(
            id=_qdrant_id(document.id),
            vector=document.vector,
            payload={
                "doc_id": document.id,
                "text": document.text,
                "metadata": document.metadata,
            },
        )
        self._client.upsert(
            collection_name=self.config.collection_name,
            points=[point],
        )

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        query_filter = _build_filter(self._models, filters)

        if hasattr(self._client, "query_points"):
            response = self._client.query_points(
                collection_name=self.config.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        else:
            points = self._client.search(
                collection_name=self.config.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )

        return [_result_from_qdrant_point(point) for point in points]

    def get(self, document_id: str) -> Optional[VectorDocument]:
        points = self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=[_qdrant_id(document_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None

        point = points[0]
        payload = point.payload or {}
        metadata = dict(payload.get("metadata") or {})
        return VectorDocument(
            id=str(payload.get("doc_id") or document_id),
            vector=list(point.vector or []),
            text=str(payload.get("text") or ""),
            metadata=metadata,
        )

    def delete(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self.config.collection_name,
            points_selector=self._models.PointIdsList(points=[_qdrant_id(document_id)]),
        )

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _ensure_collection(self, vector_size: int) -> None:
        exists = self._client.collection_exists(self.config.collection_name)
        if exists:
            return

        distance = getattr(self._models.Distance, self.config.distance.upper())
        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=self._models.VectorParams(
                size=vector_size,
                distance=distance,
            ),
        )


def _qdrant_id(document_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(document_id)))


def _build_filter(models: Any, filters: Optional[dict[str, Any]]) -> Any:
    if not filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"metadata.{key}",
                match=models.MatchValue(value=value),
            )
            for key, value in filters.items()
        ]
    )


def _result_from_qdrant_point(point: Any) -> VectorSearchResult:
    payload = point.payload or {}
    metadata = dict(payload.get("metadata") or {})
    document = VectorDocument(
        id=str(payload.get("doc_id") or point.id),
        vector=list(getattr(point, "vector", None) or []),
        text=str(payload.get("text") or ""),
        metadata=metadata,
    )
    return VectorSearchResult(document=document, score=float(point.score))
