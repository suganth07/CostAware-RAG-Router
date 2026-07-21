from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from src.adapters.storage.vector_store.base import (
    VectorDocument,
    VectorSearchResult,
    VectorStore,
)


_FORMAT_VERSION = 1


class NumpyVectorStore(VectorStore):
    """Vectorized exact-cosine store with an optional persistent local index.

    This adapter is intended for reproducible production-style benchmarks.  It
    keeps vectors in contiguous float32 arrays instead of Python float objects,
    batches matrix construction, memory-maps a completed cached index, and
    retains the same ``VectorStore`` interface used by Qdrant.
    """

    def __init__(self, *, cache_directory: str | Path | None = None) -> None:
        self.cache_directory = Path(cache_directory) if cache_directory else None
        self._documents: list[VectorDocument] = []
        self._positions: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        self._pending_matrices: list[np.ndarray] = []
        self._dimension: int | None = None
        self._dirty = False
        self.cache_loaded = False
        if self.cache_directory is not None:
            self._load_cache()

    def upsert(self, document: VectorDocument) -> None:
        self.upsert_many([document])

    def upsert_many(self, documents: Iterable[VectorDocument]) -> None:
        # Keep the final value for a repeated ID, matching normal upsert
        # semantics without creating duplicate matrix rows.
        unique: dict[str, VectorDocument] = {}
        for document in documents:
            unique[document.id] = document
        values = list(unique.values())
        if not values:
            return
        matrix = np.asarray([document.vector for document in values], dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] == 0:
            raise ValueError("Vectors must be a non-empty two-dimensional matrix")
        self._validate_dimension(int(matrix.shape[1]))
        matrix = _normalize_rows(matrix)

        # Replacements are uncommon for immutable evaluation corpora.  Handle
        # them correctly by materializing once, while retaining append-only
        # batches during the large initial index build.
        replacement_positions = [
            self._positions.get(document.id) for document in values
        ]
        if any(position is not None for position in replacement_positions):
            mutable = self._materialize(copy=True)
            new_documents: list[VectorDocument] = []
            new_rows: list[np.ndarray] = []
            for document, row, position in zip(values, matrix, replacement_positions):
                stored = _without_vector(document)
                if position is None:
                    self._positions[stored.id] = len(self._documents) + len(new_documents)
                    new_documents.append(stored)
                    new_rows.append(row)
                else:
                    self._documents[position] = stored
                    mutable[position] = row
            if new_rows:
                mutable = np.vstack(
                    [mutable, np.asarray(new_rows, dtype=np.float32)]
                )
                self._documents.extend(new_documents)
            self._matrix = np.ascontiguousarray(mutable, dtype=np.float32)
            self._pending_matrices.clear()
        else:
            start = len(self._documents)
            for offset, document in enumerate(values):
                stored = _without_vector(document)
                self._positions[stored.id] = start + offset
                self._documents.append(stored)
            self._pending_matrices.append(
                np.ascontiguousarray(matrix, dtype=np.float32)
            )
        self._dirty = True

    def existing_ids(self, document_ids: Iterable[str]) -> set[str]:
        return {
            document_id
            for document_id in (str(value) for value in document_ids)
            if document_id in self._positions
        }

    def search(
        self,
        query_vector: list[float],
        *,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[VectorSearchResult]:
        if limit <= 0 or not self._documents:
            return []
        matrix = self._materialize()
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.size == 0:
            return []
        self._validate_dimension(int(query.shape[0]))
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        normalized_query = query / norm

        if filters:
            candidate_positions = np.asarray(
                [
                    position
                    for position, document in enumerate(self._documents)
                    if _matches_filters(document.metadata, filters)
                ],
                dtype=np.int64,
            )
            if candidate_positions.size == 0:
                return []
            candidate_scores = matrix[candidate_positions] @ normalized_query
            local_positions = _top_positions(candidate_scores, limit)
            positions = candidate_positions[local_positions]
            scores = candidate_scores[local_positions]
        else:
            all_scores = matrix @ normalized_query
            positions = _top_positions(all_scores, limit)
            scores = all_scores[positions]

        ranked = sorted(
            zip(positions.tolist(), scores.tolist()),
            key=lambda item: (-float(item[1]), self._documents[item[0]].id),
        )
        return [
            VectorSearchResult(
                document=self._documents[position],
                score=float(score),
            )
            for position, score in ranked
        ]

    def get(self, document_id: str) -> Optional[VectorDocument]:
        position = self._positions.get(str(document_id))
        if position is None:
            return None
        document = self._documents[position]
        vector = self._materialize()[position].astype(float).tolist()
        return VectorDocument(
            id=document.id,
            vector=vector,
            text=document.text,
            metadata=document.metadata,
        )

    def delete(self, document_id: str) -> None:
        position = self._positions.get(str(document_id))
        if position is None:
            return
        matrix = self._materialize(copy=True)
        keep = np.ones(len(self._documents), dtype=bool)
        keep[position] = False
        self._matrix = np.ascontiguousarray(matrix[keep], dtype=np.float32)
        del self._documents[position]
        self._positions = {
            document.id: index for index, document in enumerate(self._documents)
        }
        self._pending_matrices.clear()
        self._dirty = True

    def clear(self) -> None:
        self._documents.clear()
        self._positions.clear()
        self._matrix = None
        self._pending_matrices.clear()
        self._dimension = None
        self._dirty = True
        self.cache_loaded = False

    def flush(self) -> None:
        if self.cache_directory is None or not self._dirty:
            return
        matrix = self._materialize()
        directory = self.cache_directory
        directory.mkdir(parents=True, exist_ok=True)
        vectors_path = directory / "vectors.npy"
        documents_path = directory / "documents.jsonl"
        manifest_path = directory / "manifest.json"
        temporary_vectors = directory / "vectors.tmp.npy"
        temporary_documents = directory / "documents.tmp.jsonl"
        temporary_manifest = directory / "manifest.tmp.json"

        np.save(temporary_vectors, np.asarray(matrix, dtype=np.float32))
        with temporary_documents.open("w", encoding="utf-8", newline="\n") as file:
            for document in self._documents:
                file.write(
                    json.dumps(
                        {
                            "id": document.id,
                            "text": document.text,
                            "metadata": document.metadata,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )
        temporary_manifest.write_text(
            json.dumps(
                {
                    "format_version": _FORMAT_VERSION,
                    "documents": len(self._documents),
                    "dimension": self._dimension,
                    "normalized": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_vectors.replace(vectors_path)
        temporary_documents.replace(documents_path)
        temporary_manifest.replace(manifest_path)
        self._dirty = False

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _materialize(self, *, copy: bool = False) -> np.ndarray:
        if self._pending_matrices:
            matrices = [
                *([self._matrix] if self._matrix is not None else []),
                *self._pending_matrices,
            ]
            self._matrix = np.ascontiguousarray(
                np.vstack(matrices), dtype=np.float32
            )
            self._pending_matrices.clear()
        if self._matrix is None:
            dimension = int(self._dimension or 0)
            self._matrix = np.empty((0, dimension), dtype=np.float32)
        return np.array(self._matrix, copy=True) if copy else self._matrix

    def _validate_dimension(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("Vector dimension must be positive")
        if self._dimension is None:
            self._dimension = dimension
            return
        if dimension != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, got {dimension}"
            )

    def _load_cache(self) -> None:
        directory = self.cache_directory
        if directory is None:
            return
        vectors_path = directory / "vectors.npy"
        documents_path = directory / "documents.jsonl"
        manifest_path = directory / "manifest.json"
        if not (vectors_path.exists() and documents_path.exists() and manifest_path.exists()):
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("format_version", -1)) != _FORMAT_VERSION:
                return
            matrix = np.load(vectors_path, mmap_mode="r")
            documents: list[VectorDocument] = []
            with documents_path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    documents.append(
                        VectorDocument(
                            id=str(value["id"]),
                            vector=[],
                            text=str(value.get("text") or ""),
                            metadata=dict(value.get("metadata") or {}),
                        )
                    )
            if matrix.ndim != 2 or matrix.shape[0] != len(documents):
                return
            expected_count = int(manifest.get("documents", -1))
            expected_dimension = int(manifest.get("dimension", -1))
            if matrix.shape != (expected_count, expected_dimension):
                return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        self._documents = documents
        self._positions = {
            document.id: index for index, document in enumerate(documents)
        }
        self._matrix = matrix
        self._dimension = int(matrix.shape[1])
        self._dirty = False
        self.cache_loaded = True


def _without_vector(document: VectorDocument) -> VectorDocument:
    return VectorDocument(
        id=document.id,
        vector=[],
        text=document.text,
        metadata=document.metadata,
    )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _top_positions(scores: np.ndarray, limit: int) -> np.ndarray:
    count = min(max(int(limit), 0), int(scores.shape[0]))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    if count == scores.shape[0]:
        return np.arange(scores.shape[0], dtype=np.int64)
    positions = np.argpartition(scores, scores.shape[0] - count)[-count:]
    return positions.astype(np.int64, copy=False)


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())
