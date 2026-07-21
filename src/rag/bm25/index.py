from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from src.rag.types import RetrievalResult

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class BM25Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """Dependency-free Okapi BM25 index."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._documents: dict[str, BM25Document] = {}
        self._term_freqs: dict[str, Counter[str]] = {}
        self._document_freqs: Counter[str] = Counter()
        self._doc_lengths: dict[str, int] = {}
        self._avg_doc_length = 0.0

    def upsert(self, document: BM25Document) -> None:
        self.delete(document.id)

        tokens = tokenize(document.text)
        term_freq = Counter(tokens)
        self._documents[document.id] = document
        self._term_freqs[document.id] = term_freq
        self._doc_lengths[document.id] = len(tokens)
        for term in term_freq:
            self._document_freqs[term] += 1
        self._recompute_average_length()

    def upsert_many(self, documents: list[BM25Document]) -> None:
        for document in documents:
            self.upsert(document)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        if not query_terms or not self._documents:
            return []

        scores: list[RetrievalResult] = []
        for document_id, document in self._documents.items():
            if filters and not _matches_filters(document.metadata, filters):
                continue
            score = self._score_document(document_id, query_terms)
            if score <= 0.0:
                continue
            scores.append(
                RetrievalResult(
                    id=document.id,
                    text=document.text,
                    score=score,
                    metadata=document.metadata,
                    source="bm25",
                )
            )

        scores.sort(key=lambda result: result.score, reverse=True)
        return scores[:limit]

    def delete(self, document_id: str) -> None:
        document_id = str(document_id)
        term_freq = self._term_freqs.pop(document_id, None)
        if term_freq is not None:
            for term in term_freq:
                self._document_freqs[term] -= 1
                if self._document_freqs[term] <= 0:
                    del self._document_freqs[term]
        self._documents.pop(document_id, None)
        self._doc_lengths.pop(document_id, None)
        self._recompute_average_length()

    def clear(self) -> None:
        self._documents.clear()
        self._term_freqs.clear()
        self._document_freqs.clear()
        self._doc_lengths.clear()
        self._avg_doc_length = 0.0

    def _score_document(self, document_id: str, query_terms: list[str]) -> float:
        total_documents = len(self._documents)
        term_freq = self._term_freqs[document_id]
        doc_length = self._doc_lengths[document_id]
        score = 0.0

        for term in query_terms:
            frequency = term_freq.get(term, 0)
            if frequency == 0:
                continue

            document_frequency = self._document_freqs.get(term, 0)
            idf = math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self._avg_doc_length or 1.0)
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator

        return score

    def _recompute_average_length(self) -> None:
        if not self._doc_lengths:
            self._avg_doc_length = 0.0
            return
        self._avg_doc_length = sum(self._doc_lengths.values()) / len(self._doc_lengths)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def _matches_filters(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())
