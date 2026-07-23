"""Model-neutral primary/secondary critic orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord


class CriticRunner(Protocol):
    def judge(self, question: QuestionRecord, outcome: CandidateOutcome) -> CritiqueRecord: ...


@dataclass(frozen=True, slots=True)
class CriticSlotConfig:
    enabled: bool = False
    adapter: str | None = None
    model_id: str | None = None
    revision: str | None = None
    options: dict[str, object] | None = None

    def validate(self, *, role: str) -> None:
        if self.enabled and (not self.adapter or not self.model_id):
            raise ValueError(f"Enabled {role} critic requires configurable adapter and model_id")


@dataclass(frozen=True, slots=True)
class CriticEnsembleConfig:
    primary: CriticSlotConfig
    secondary: CriticSlotConfig
    secondary_mode: str = "uncertain_or_audit"  # always | uncertain_or_audit | disabled
    uncertainty_threshold: float = 0.40

    def validate(self) -> None:
        self.primary.validate(role="primary")
        self.secondary.validate(role="secondary")
        if self.secondary_mode not in {"always", "uncertain_or_audit", "disabled"}:
            raise ValueError("secondary_mode must be always, uncertain_or_audit, or disabled")
        if not 0.0 <= self.uncertainty_threshold <= 1.0:
            raise ValueError("uncertainty_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EnsembleCritique:
    primary: CritiqueRecord
    secondary: CritiqueRecord | None


class CriticEnsemble:
    def __init__(self, primary: CriticRunner, *, secondary: CriticRunner | None = None, config: CriticEnsembleConfig) -> None:
        config.validate()
        if not config.primary.enabled:
            raise ValueError("Primary critic must be enabled")
        if config.secondary.enabled and secondary is None:
            raise ValueError("A configured secondary critic requires a runner")
        self.primary = primary
        self.secondary = secondary
        self.config = config

    def judge(self, question: QuestionRecord, outcome: CandidateOutcome, *, force_secondary: bool = False) -> EnsembleCritique:
        primary = self.primary.judge(question, outcome)
        run_secondary = self.secondary is not None and (
            force_secondary
            or self.config.secondary_mode == "always"
            or (self.config.secondary_mode == "uncertain_or_audit" and primary.confidence < self.config.uncertainty_threshold)
        )
        return EnsembleCritique(primary=primary, secondary=self.secondary.judge(question, outcome) if run_secondary and self.secondary else None)
