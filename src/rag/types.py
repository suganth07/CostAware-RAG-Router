from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.text = str(self.text)
        self.score = float(self.score)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class RAGConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    max_context_chars: int = 12_000
    system_prompt: str = (
        "You answer questions using only the supplied retrieval context. "
        "If the context is insufficient, say what is missing."
    )


@dataclass(slots=True)
class RAGIndexingReport:
    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGContext:
    question: str
    results: list[RetrievalResult]
    context_text: str
    retriever: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RAGAnswer:
    question: str
    answer: str
    context: RAGContext
    usage: dict[str, Any] = field(default_factory=dict)
