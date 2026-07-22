"""Safe, explicit joins between critic artifacts and metric vectors."""

from __future__ import annotations

from dataclasses import replace

from src.router.types import CritiqueRecord, MetricVector


def with_critique(metrics: MetricVector, critique: CritiqueRecord) -> MetricVector:
    """Attach a validated critic record without overwriting deterministic metrics."""
    return replace(
        metrics,
        critic_correctness=critique.correctness,
        critic_groundedness=critique.groundedness,
        critic_evidence_completeness=critique.evidence_completeness,
        critic_reasoning_completeness=critique.reasoning_completeness,
        critic_answer_relevance=critique.answer_relevance,
        critic_confidence=critique.confidence,
        critic_hallucination_detected=critique.hallucination_detected,
    )
