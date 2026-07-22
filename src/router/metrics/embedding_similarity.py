"""Calibrated answer-embedding similarity for auxiliary reward signal.

Raw cosine similarity is deliberately retained separately: it is not safe to
interpret a high embedding score as correctness until it has been calibrated
against manually checked answer pairs for each answer type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from src.adapters.embedding.adapter import EmbeddingAdapter


@dataclass(frozen=True, slots=True)
class AnswerTypeCalibration:
    """Map raw cosine from a calibrated low/high range into [0, 1]."""

    low: float
    high: float

    def apply(self, raw_similarity: float) -> float:
        if self.high <= self.low:
            raise ValueError("Calibration high threshold must exceed low threshold")
        return max(0.0, min(1.0, (raw_similarity - self.low) / (self.high - self.low)))


def max_answer_embedding_similarity(
    prediction: str,
    accepted_answers: Sequence[str],
    *,
    embedder: EmbeddingAdapter,
) -> float | None:
    """Return cosine similarity to the closest accepted answer, or null."""
    candidates = [answer.strip() for answer in accepted_answers if answer and answer.strip()]
    if not prediction or not prediction.strip() or not candidates:
        return None
    vectors = embedder.embed_texts([prediction.strip(), *candidates]).vectors
    if len(vectors) != len(candidates) + 1:
        raise ValueError("Embedding adapter returned an unexpected vector count")
    return max(_cosine(vectors[0], answer_vector) for answer_vector in vectors[1:])


def calibrated_embedding_similarity(
    raw_similarity: float | None,
    *,
    answer_type: str,
    calibrations: Mapping[str, AnswerTypeCalibration],
    hard_conflict: bool,
) -> float | None:
    """Apply only an explicit answer-type calibration; conflicts override it."""
    if raw_similarity is None:
        return None
    if hard_conflict:
        return 0.0
    calibration = calibrations.get(answer_type) or calibrations.get("default")
    return calibration.apply(raw_similarity) if calibration else None


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must be non-empty and have equal dimensions")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
