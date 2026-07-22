"""Fit transparent answer-type calibration thresholds from labelled pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.router.metrics.embedding_similarity import AnswerTypeCalibration


@dataclass(frozen=True, slots=True)
class EmbeddingCalibrationExample:
    answer_type: str
    raw_similarity: float
    equivalent: bool

    def __post_init__(self) -> None:
        if not -1.0 <= self.raw_similarity <= 1.0:
            raise ValueError("raw_similarity must be a cosine value in [-1, 1]")


def fit_answer_type_calibrations(
    examples: Iterable[EmbeddingCalibrationExample],
    *,
    min_per_class: int = 20,
    negative_quantile: float = 0.90,
    positive_quantile: float = 0.10,
) -> dict[str, AnswerTypeCalibration]:
    """Fit plan.md low/high thresholds without hiding overlap in a heuristic.

    ``low`` is the high tail of non-equivalent pairs; ``high`` is the low tail
    of equivalent pairs. If they overlap, the embedding model is not separable
    enough for that answer type and fitting fails instead of inventing reward.
    """
    if not 0.0 <= negative_quantile <= 1.0 or not 0.0 <= positive_quantile <= 1.0:
        raise ValueError("Calibration quantiles must be in [0, 1]")
    grouped: dict[str, dict[bool, list[float]]] = {}
    for item in examples:
        bucket = grouped.setdefault(item.answer_type, {True: [], False: []})
        bucket[item.equivalent].append(item.raw_similarity)
    result: dict[str, AnswerTypeCalibration] = {}
    for answer_type, labels in grouped.items():
        positives, negatives = labels[True], labels[False]
        if len(positives) < min_per_class or len(negatives) < min_per_class:
            raise ValueError(
                f"{answer_type!r} requires at least {min_per_class} equivalent and "
                f"{min_per_class} non-equivalent labelled pairs; got {len(positives)} and {len(negatives)}"
            )
        low = quantile(negatives, negative_quantile)
        high = quantile(positives, positive_quantile)
        if high <= low:
            raise ValueError(
                f"Embedding similarity is not separable for {answer_type!r}: "
                f"low={low:.4f}, high={high:.4f}. Choose/validate another model."
            )
        result[answer_type] = AnswerTypeCalibration(low=low, high=high)
    return result


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]
