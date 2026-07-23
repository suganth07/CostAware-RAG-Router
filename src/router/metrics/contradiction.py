"""Configurable contradiction scoring with safe deterministic hard guards."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


class ContradictionScorer(ABC):
    @abstractmethod
    def score(self, *, question: str, prediction: str, accepted_answers: Sequence[str], numeric_conflict: bool, polarity_conflict: bool, entity_conflict: bool) -> float | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HardConflictContradictionScorer(ContradictionScorer):
    """Returns certainty only for deterministic number/date/polarity conflicts."""

    def score(self, *, question: str, prediction: str, accepted_answers: Sequence[str], numeric_conflict: bool, polarity_conflict: bool, entity_conflict: bool) -> float | None:
        return 1.0 if numeric_conflict or polarity_conflict else 0.0


def validate_contradiction(value: float | None) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError("Contradiction probability must be within [0, 1]")
    return score
