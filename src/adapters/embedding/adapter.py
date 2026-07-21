from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


class EmbeddingAdapter(ABC):
    """Common interface for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        raise NotImplementedError

    def embed_text(self, text: str) -> list[float]:
        result = self.embed_texts([text])
        if not result.vectors:
            raise ValueError("Embedding provider returned no vectors")
        return result.vectors[0]
