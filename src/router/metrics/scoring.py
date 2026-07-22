"""Bridge existing evaluator metrics into the router's nullable MetricVector."""

from __future__ import annotations

from src.adapters.datasets.base import EvaluationSample
from src.evaluation.core import evaluate_record
from src.router.metrics.answer_types import deterministic_correctness
from src.router.types import CandidateOutcome, MetricVector


def score_outcome(sample: EvaluationSample, outcome: CandidateOutcome) -> MetricVector:
    row = evaluate_record(sample, outcome.retrieval_record)
    deterministic = deterministic_correctness(outcome.generated_answer, sample.answers)
    # Retrieval-only outcomes deliberately retain unavailable answer metrics as
    # null. Reward composition masks missing values instead of converting them.
    evidence_recall = _number(row.get("supporting_evidence_recall"))
    answer_present = _number(row.get("context_answer_recall"))
    precision = _context_precision(outcome.retrieval_record.context_text, sample.answers)
    return MetricVector(
        exact_or_alias_match=_number(deterministic["exact"]),
        token_or_set_f1=_number(deterministic["f1"]),
        boolean_accuracy=_number(deterministic["boolean"]),
        numeric_date_accuracy=_number(deterministic["numeric_date_accuracy"]),
        entity_conflict=bool(deterministic["entity_conflict"]),
        numeric_conflict=bool(deterministic["numeric_conflict"]),
        polarity_conflict=bool(deterministic["polarity_conflict"]),
        supporting_evidence_recall=evidence_recall,
        context_precision=precision,
        evidence_mrr=_number(row.get("evidence_mrr")),
        claim_faithfulness=evidence_recall,
        answer_completeness=answer_present,
        answer_relevance=answer_present,
        failure_penalty=1.0 if outcome.error_type or outcome.timeout else 0.0,
    )


def _context_precision(context: str, answers: list[str]) -> float | None:
    if not context.strip() or not answers:
        return None
    # Conservative binary signal until a calibrated claim evaluator is added.
    from src.evaluation.core import normalize_text
    normalized = normalize_text(context)
    return float(any(normalize_text(answer) in normalized for answer in answers if answer))


def _number(value) -> float | None:
    return float(value) if value is not None else None
