"""Inspectable deterministic quality and compute composition with null masking."""

from __future__ import annotations

from dataclasses import dataclass

from src.router.config import RewardConfig
from src.router.types import CandidateOutcome, MetricVector


@dataclass(frozen=True, slots=True)
class QualityComposition:
    quality: float | None
    base_quality: float | None
    penalty: float
    availability_mask: dict[str, bool]


def compose_quality(metrics: MetricVector, *, config: RewardConfig) -> float | None:
    """Compatibility wrapper for callers that need only the final scalar."""
    return compose_quality_details(metrics, config=config).quality


def compose_quality_details(metrics: MetricVector, *, config: RewardConfig) -> QualityComposition:
    """Compose nullable blocks and expose availability for calibration audits."""
    config.validate()
    # Raw embedding cosine is intentionally excluded.  Only an answer-type
    # calibrated value can contribute to quality, and it is overridden to zero
    # by hard conflicts in the calibration helper.
    answer = _mean_available(
        metrics.exact_or_alias_match,
        metrics.token_or_set_f1,
        metrics.boolean_accuracy,
        metrics.numeric_date_accuracy,
        metrics.semantic_equivalence,
        metrics.embedding_similarity_calibrated,
        metrics.critic_correctness,
    )
    faithfulness = _mean_available(metrics.claim_faithfulness, metrics.critic_groundedness)
    evidence = _mean_available(metrics.supporting_evidence_recall, metrics.context_precision, metrics.evidence_ndcg_at_k, metrics.evidence_mrr)
    completeness = _mean_available(
        metrics.answer_completeness,
        metrics.critic_evidence_completeness,
        metrics.critic_reasoning_completeness,
        metrics.critic_answer_relevance,
    )
    weighted = (
        (answer, config.answer_weight),
        (faithfulness, config.faithfulness_weight),
        (evidence, config.evidence_weight),
        (completeness, config.completeness_weight),
    )
    availability = {"answer": answer is not None, "faithfulness": faithfulness is not None, "evidence": evidence is not None, "completeness": completeness is not None}
    active_weight = sum(weight for value, weight in weighted if value is not None)
    if active_weight == 0:
        return QualityComposition(quality=None, base_quality=None, penalty=0.0, availability_mask=availability)
    base = sum(float(value) * weight for value, weight in weighted if value is not None) / active_weight
    penalty = config.contradiction_penalty * float(metrics.contradiction_probability or 0.0)
    if metrics.claim_faithfulness is not None:
        penalty += config.unsupported_penalty * (1.0 - float(metrics.claim_faithfulness))
    penalty += config.execution_failure_penalty * metrics.failure_penalty
    if metrics.critic_hallucination_detected is True:
        penalty += config.unsupported_penalty
    if metrics.correct_refusal == 0.0:
        penalty += config.incorrect_refusal_penalty
    if metrics.numeric_conflict or metrics.polarity_conflict:
        return QualityComposition(quality=-1.0, base_quality=base, penalty=penalty, availability_mask=availability)
    return QualityComposition(quality=max(-1.0, min(1.0, base - penalty)), base_quality=base, penalty=penalty, availability_mask=availability)


def compute_units(outcome: CandidateOutcome) -> float:
    """Reduced, documented compute unit when GPU/CPU meters are unavailable."""
    details = outcome.retrieval_record.details
    graph_nodes = float(details.get("graph_nodes") or 0.0)
    graph_edges = float(details.get("graph_edges") or 0.0)
    context_chars = float(len(outcome.retrieval_record.context_text))
    input_tokens = float(outcome.input_tokens or 0)
    output_tokens = float(outcome.output_tokens or 0)
    return 0.0005 * graph_nodes + 0.0001 * graph_edges + 0.00001 * context_chars + 0.001 * input_tokens + 0.002 * output_tokens


def _mean_available(*values: float | None) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None
