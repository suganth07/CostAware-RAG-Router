from __future__ import annotations

import pytest

from scripts.router.analyze_latency_subset import _logical_repeat, _verify_compatible, _within_question_variation
from src.evaluation.core import RetrievalRecord
from src.router.constants import ACTION_IDS
from src.router.types import CandidateOutcome, MetricVector, ScoredOutcome


def _row(action: str, *, repeat_id: int, latency: float, hardware: str = "hardware") -> ScoredOutcome:
    outcome = CandidateOutcome(
        question_id="question",
        candidate_id=action,
        repeat_id=repeat_id,
        run_id=f"run-{repeat_id}",
        pipeline_fingerprint=f"pipeline-{action}",
        corpus_fingerprint="corpus",
        model_fingerprint="model",
        hardware_fingerprint=hardware,
        retrieval_record=RetrievalRecord(system=action, sample_id="question", question="Question", context_text=""),
        end_to_end_latency_ms=latency,
    )
    return ScoredOutcome(outcome=outcome, metrics=MetricVector(), quality=0.5, compute_units=0.1)


def test_latency_cohort_requires_matching_runtime_fingerprints() -> None:
    source = [_row(action, repeat_id=0, latency=10.0) for action in ACTION_IDS]
    supplementary = [_row(action, repeat_id=0, latency=12.0) for action in ACTION_IDS]
    _verify_compatible(source, supplementary)

    incompatible = [_row(action, repeat_id=0, latency=12.0, hardware="different") for action in ACTION_IDS]
    with pytest.raises(ValueError, match="hardware_fingerprint"):
        _verify_compatible(source, incompatible)


def test_latency_cohort_relabels_only_derived_in_memory_rows() -> None:
    original = _row("hybrid_rag", repeat_id=0, latency=10.0)
    derived = _logical_repeat(original, repeat_id=2, cohort_id="cohort")

    assert original.outcome.repeat_id == 0
    assert original.outcome.run_id == "run-0"
    assert derived.outcome.repeat_id == 2
    assert derived.outcome.run_id == "cohort"


def test_within_question_latency_variation_uses_logical_repeats() -> None:
    rows = []
    for action in ACTION_IDS:
        rows.extend(
            _logical_repeat(_row(action, repeat_id=repeat_id, latency=10.0 + repeat_id), repeat_id=repeat_id, cohort_id="cohort")
            for repeat_id in range(3)
        )

    variation = _within_question_variation(rows)
    assert variation["hybrid_rag"]["questions_with_all_latency_values"] == 1
    assert variation["hybrid_rag"]["mean_within_question_range_ms"] == 2.0
