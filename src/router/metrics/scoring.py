"""Bridge existing evaluator metrics into the router's nullable MetricVector."""

from __future__ import annotations

from src.adapters.datasets.base import EvaluationSample
from src.evaluation.core import evaluate_record
from src.router.metrics.answer_types import deterministic_correctness
from src.router.metrics.completeness import accepted_answer_coverage
from src.router.metrics.contradiction import ContradictionScorer, HardConflictContradictionScorer, validate_contradiction
from src.router.metrics.evidence import evidence_ndcg_at_k
from src.router.metrics.faithfulness import ClaimExtractor, ClaimVerifier, claim_faithfulness
from src.router.metrics.semantic_equivalence import SemanticEquivalenceScorer, UnavailableSemanticEquivalenceScorer, validate_score
from src.router.types import CandidateOutcome, MetricVector


def score_outcome(
    sample: EvaluationSample,
    outcome: CandidateOutcome,
    *,
    semantic_scorer: SemanticEquivalenceScorer | None = None,
    contradiction_scorer: ContradictionScorer | None = None,
    claim_extractor: ClaimExtractor | None = None,
    claim_verifier: ClaimVerifier | None = None,
    gold_evidence: list[dict] | None = None,
) -> MetricVector:
    row = evaluate_record(sample, outcome.retrieval_record)
    deterministic = deterministic_correctness(outcome.generated_answer, sample.answers)
    # Retrieval-only outcomes deliberately retain unavailable answer metrics as
    # null. Reward composition masks missing values instead of converting them.
    evidence_recall = _number(row.get("supporting_evidence_recall"))
    answer_present = _number(row.get("context_answer_recall"))
    precision = _context_precision(outcome.retrieval_record.context_text, sample.answers)
    semantic = (semantic_scorer or UnavailableSemanticEquivalenceScorer()).score(question=sample.question, prediction=outcome.generated_answer, accepted_answers=sample.answers)
    contradiction = (contradiction_scorer or HardConflictContradictionScorer()).score(
        question=sample.question,
        prediction=outcome.generated_answer,
        accepted_answers=sample.answers,
        numeric_conflict=bool(deterministic["numeric_conflict"]),
        polarity_conflict=bool(deterministic["polarity_conflict"]),
        entity_conflict=bool(deterministic["entity_conflict"]),
    )
    faithfulness = claim_faithfulness(answer=outcome.generated_answer, context=outcome.retrieval_record.context_text, extractor=claim_extractor, verifier=claim_verifier) if claim_extractor and claim_verifier else None
    return MetricVector(
        exact_or_alias_match=_number(deterministic["exact"]),
        token_or_set_f1=_number(deterministic["f1"]),
        boolean_accuracy=_number(deterministic["boolean"]),
        numeric_date_accuracy=_number(deterministic["numeric_date_accuracy"]),
        semantic_equivalence=validate_score(semantic),
        contradiction_probability=validate_contradiction(contradiction),
        entity_conflict=bool(deterministic["entity_conflict"]),
        numeric_conflict=bool(deterministic["numeric_conflict"]),
        polarity_conflict=bool(deterministic["polarity_conflict"]),
        supporting_evidence_recall=evidence_recall,
        context_precision=precision,
        evidence_ndcg_at_k=evidence_ndcg_at_k(gold_evidence or sample.metadata.get("gold_evidence") or [], outcome.retrieval_record.evidence_units),
        evidence_mrr=_number(row.get("evidence_mrr")),
        claim_faithfulness=faithfulness,
        answer_completeness=accepted_answer_coverage(outcome.generated_answer, sample.answers),
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
