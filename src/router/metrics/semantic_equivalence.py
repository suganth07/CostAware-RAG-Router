"""Plan-aligned interface for a question-conditioned semantic answer model.

The model is intentionally injectable: plan.md leaves the final cross-encoder
or answer-similarity model as an experimental choice. Returning ``None`` for
an unavailable model preserves nullable metric semantics and prevents an
unvalidated heuristic from entering reward.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


class SemanticEquivalenceScorer(ABC):
    """Scores (question, generated answer, accepted answer) in [0, 1]."""

    @abstractmethod
    def score(self, *, question: str, prediction: str, accepted_answers: Sequence[str]) -> float | None:
        raise NotImplementedError


SemanticAdapterFactory = Callable[[dict[str, object]], SemanticEquivalenceScorer]


@dataclass(frozen=True, slots=True)
class SemanticScorerConfig:
    """Model-neutral configuration required before enabling semantic reward."""

    enabled: bool = False
    adapter: str | None = None
    model_id: str | None = None
    revision: str | None = None
    options: dict[str, object] | None = None

    def validate(self) -> None:
        if self.enabled and (not self.adapter or not self.model_id):
            raise ValueError("An enabled semantic scorer requires configurable adapter and model_id")


def build_semantic_scorer(config: SemanticScorerConfig, factories: Mapping[str, SemanticAdapterFactory]) -> SemanticEquivalenceScorer:
    config.validate()
    if not config.enabled:
        return UnavailableSemanticEquivalenceScorer()
    assert config.adapter is not None
    if config.adapter not in factories:
        raise ValueError(f"No semantic-equivalence adapter registered for {config.adapter!r}")
    scorer = factories[config.adapter]({"model_id": config.model_id, "revision": config.revision, **(config.options or {})})
    return scorer


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
