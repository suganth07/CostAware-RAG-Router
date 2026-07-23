"""Disagreement and uncertainty logic; adjudicator model is injectable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.router.types import CritiqueRecord

_DIMENSIONS = ("correctness", "groundedness", "evidence_completeness", "reasoning_completeness", "answer_relevance")


@dataclass(frozen=True, slots=True)
class DisagreementConfig:
    score_tolerance: float = 0.25
    confidence_tolerance: float = 0.30

    def validate(self) -> None:
        if not 0.0 <= self.score_tolerance <= 1.0 or not 0.0 <= self.confidence_tolerance <= 1.0:
            raise ValueError("Disagreement tolerances must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CriticDisagreement:
    dimension_gaps: dict[str, float]
    categorical_conflict: bool
    confidence_gap: float
    requires_adjudication: bool
    uncertainty_weight: float


class CriticAdjudicator(Protocol):
    def adjudicate(self, primary: CritiqueRecord, secondary: CritiqueRecord) -> CritiqueRecord: ...


@dataclass(frozen=True, slots=True)
class AdjudicationResult:
    selected: CritiqueRecord | None
    disagreement: CriticDisagreement
    adjudicated: bool


def compare_critiques(primary: CritiqueRecord, secondary: CritiqueRecord, *, config: DisagreementConfig) -> CriticDisagreement:
    config.validate()
    gaps = {dimension: abs(float(getattr(primary, dimension)) - float(getattr(secondary, dimension))) for dimension in _DIMENSIONS}
    confidence_gap = abs(primary.confidence - secondary.confidence)
    categorical = primary.failure_type != secondary.failure_type or primary.hallucination_detected != secondary.hallucination_detected
    requires = categorical or confidence_gap > config.confidence_tolerance or any(gap > config.score_tolerance for gap in gaps.values())
    severity = max([*gaps.values(), confidence_gap, float(categorical)])
    return CriticDisagreement(
        dimension_gaps=gaps,
        categorical_conflict=categorical,
        confidence_gap=confidence_gap,
        requires_adjudication=requires,
        uncertainty_weight=max(0.0, 1.0 - severity) if requires else 1.0,
    )


def adjudicate_or_mark_uncertain(primary: CritiqueRecord, secondary: CritiqueRecord | None, *, config: DisagreementConfig, adjudicator: CriticAdjudicator | None = None) -> AdjudicationResult:
    if secondary is None:
        return AdjudicationResult(selected=primary, disagreement=CriticDisagreement({}, False, 0.0, False, 1.0), adjudicated=False)
    disagreement = compare_critiques(primary, secondary, config=config)
    if not disagreement.requires_adjudication:
        return AdjudicationResult(selected=primary, disagreement=disagreement, adjudicated=False)
    if adjudicator is None:
        return AdjudicationResult(selected=None, disagreement=disagreement, adjudicated=False)
    return AdjudicationResult(selected=adjudicator.adjudicate(primary, secondary), disagreement=disagreement, adjudicated=True)
