from src.evaluation.core import RetrievalRecord
from src.router.config import BudgetConfig, RewardConfig
from src.router.reward.composer import compose_quality
from src.router.reward.oracle import select_budget_aware_oracle
from src.router.types import CandidateOutcome, MetricVector, ScoredOutcome


def _scored(action: str, quality: float, latency: float, compute: float) -> ScoredOutcome:
    outcome = CandidateOutcome(
        question_id="q1", candidate_id=action, repeat_id=0, run_id="run",
        pipeline_fingerprint="p", corpus_fingerprint="c", model_fingerprint="m", hardware_fingerprint="h",
        retrieval_record=RetrievalRecord(system=action, sample_id="q1", question="question", context_text=""),
        end_to_end_latency_ms=latency,
    )
    return ScoredOutcome(outcome=outcome, metrics=MetricVector(), quality=quality, compute_units=compute)


def test_hard_numeric_conflict_overrides_other_quality_credit():
    quality = compose_quality(
        MetricVector(exact_or_alias_match=0.0, token_or_set_f1=0.9, supporting_evidence_recall=1.0, numeric_conflict=True),
        config=RewardConfig(),
    )
    assert quality == -1.0


def test_oracle_chooses_highest_quality_feasible_action():
    selected = select_budget_aware_oracle([
        _scored("hybrid_rag", 0.70, 10.0, 0.2),
        _scored("graph_rag", 0.95, 200.0, 0.7),
        _scored("optimized_graph_rag", 0.85, 40.0, 0.3),
    ], budgets=BudgetConfig(max_request_latency_ms=50.0, compute_units=0.5))
    assert selected.outcome.candidate_id == "optimized_graph_rag"
