"""Plan-required human-label calibration contracts; no labels are fabricated."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.router.types import CritiqueRecord


@dataclass(frozen=True, slots=True)
class HumanCritiqueLabel:
    question_id: str
    candidate_id: str
    repeat_id: int
    run_id: str
    correctness: float
    groundedness: float
    completeness: float

    def validate(self) -> None:
        for field in ("correctness", "groundedness", "completeness"):
            if not 0.0 <= float(getattr(self, field)) <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    labels: int
    matched_critiques: int
    mean_absolute_error: dict[str, float | None]


def summarize_human_calibration(critiques: Sequence[CritiqueRecord], labels: Sequence[HumanCritiqueLabel]) -> CalibrationSummary:
    index = {(item.question_id, item.candidate_id, item.repeat_id, item.run_id): item for item in critiques}
    errors: dict[str, list[float]] = {"correctness": [], "groundedness": [], "completeness": []}
    matched = 0
    for label in labels:
        label.validate()
        critique = index.get((label.question_id, label.candidate_id, label.repeat_id, label.run_id))
        if critique is None:
            continue
        matched += 1
        errors["correctness"].append(abs(critique.correctness - label.correctness))
        errors["groundedness"].append(abs(critique.groundedness - label.groundedness))
        errors["completeness"].append(abs(critique.evidence_completeness - label.completeness))
    return CalibrationSummary(labels=len(labels), matched_critiques=matched, mean_absolute_error={name: sum(values) / len(values) if values else None for name, values in errors.items()})
