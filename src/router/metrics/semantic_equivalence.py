"""Plan-aligned interface for a question-conditioned semantic answer model.

The model is intentionally injectable: plan.md leaves the final cross-encoder
or answer-similarity model as an experimental choice. Returning ``None`` for
an unavailable model preserves nullable metric semantics and prevents an
unvalidated heuristic from entering reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


class SemanticEquivalenceScorer(ABC):
    """Scores (question, generated answer, accepted answer) in [0, 1]."""

    @abstractmethod
    def score(self, *, question: str, prediction: str, accepted_answers: Sequence[str]) -> float | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class UnavailableSemanticEquivalenceScorer(SemanticEquivalenceScorer):
    """Explicit null scorer used until a model is selected and validated."""

    reason: str = "No validated semantic-equivalence model is configured"

    def score(self, *, question: str, prediction: str, accepted_answers: Sequence[str]) -> float | None:
        return None


def validate_score(value: float | None) -> float | None:
    """Keep model adapters from silently emitting invalid reward inputs."""
    if value is None:
        return None
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError("Semantic-equivalence score must be within [0, 1]")
    return score
