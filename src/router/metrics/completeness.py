"""Answer completeness metrics separated from evidence availability."""

from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.core import normalize_text


def accepted_answer_coverage(prediction: str, accepted_answers: Sequence[str]) -> float | None:
    """Fraction of accepted endpoint answers explicitly present in the answer."""
    expected = [normalize_text(answer) for answer in accepted_answers if normalize_text(answer)]
    if not expected:
        return None
    response = normalize_text(prediction)
    if not response:
        return 0.0
    return sum(answer in response for answer in expected) / len(expected)
