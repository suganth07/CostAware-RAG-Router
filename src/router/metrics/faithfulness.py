"""Interfaces for claim-level faithfulness without assuming a fixed judge model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class ClaimExtractor(ABC):
    @abstractmethod
    def extract(self, answer: str) -> list[str]:
        raise NotImplementedError


class ClaimVerifier(ABC):
    @abstractmethod
    def support_scores(self, *, claims: Sequence[str], context: str) -> list[float] | None:
        """Return one [0,1] support score per claim, or None when unavailable."""
        raise NotImplementedError


class SentenceClaimExtractor(ClaimExtractor):
    """Deterministic segmentation only; it does not claim semantic support."""

    def extract(self, answer: str) -> list[str]:
        import re
        return [item.strip() for item in re.split(r"(?<=[.!?])\s+", answer.strip()) if item.strip()]


def claim_faithfulness(*, answer: str, context: str, extractor: ClaimExtractor, verifier: ClaimVerifier) -> float | None:
    claims = extractor.extract(answer)
    if not claims or not context.strip():
        return None
    scores = verifier.support_scores(claims=claims, context=context)
    if scores is None:
        return None
    if len(scores) != len(claims) or any(not 0.0 <= float(score) <= 1.0 for score in scores):
        raise ValueError("Claim verifier must return one [0,1] score per claim")
    return sum(float(score) for score in scores) / len(scores)
