from src.evaluation.core import EvidenceUnit
from src.router.critics.adjudication import DisagreementConfig, compare_critiques
from src.router.metrics.completeness import accepted_answer_coverage
from src.router.metrics.embedding_calibration import EmbeddingCalibrationExample, fit_answer_type_calibrations
from src.router.metrics.evidence import evidence_ndcg_at_k
from src.router.reward.composer import compose_quality_details
from src.router.config import RewardConfig
from src.router.types import CritiqueRecord, MetricVector


def test_evidence_ndcg_is_one_for_ranked_gold_units():
    value = evidence_ndcg_at_k(
        [{"identifiers": ["a"]}, {"identifiers": ["b"]}],
        [EvidenceUnit(id="a", text="one"), EvidenceUnit(id="b", text="two")],
        k=2,
    )
    assert value == 1.0


def test_completeness_covers_all_multi_answer_endpoints():
    assert accepted_answer_coverage("The actors are Ada and Bea.", ["Ada", "Bea"]) == 1.0


def test_embedding_calibration_requires_separable_human_labels():
    examples = [
        *[EmbeddingCalibrationExample("span", 0.9, True) for _ in range(20)],
        *[EmbeddingCalibrationExample("span", 0.1, False) for _ in range(20)],
    ]
    assert fit_answer_type_calibrations(examples)["span"].high > fit_answer_type_calibrations(examples)["span"].low


def test_quality_composition_reports_availability_mask():
    result = compose_quality_details(MetricVector(exact_or_alias_match=1.0), config=RewardConfig())
    assert result.quality == 1.0
    assert result.availability_mask["answer"] is True
    assert result.availability_mask["faithfulness"] is False


def test_critic_disagreement_reduces_uncertainty_weight():
    first = _critique("first", correctness=1.0)
    second = _critique("second", correctness=0.0)
    result = compare_critiques(first, second, config=DisagreementConfig())
    assert result.requires_adjudication is True
    assert result.uncertainty_weight == 0.0


def _critique(model: str, *, correctness: float) -> CritiqueRecord:
    return CritiqueRecord(
        question_id="q", candidate_id="hybrid_rag", repeat_id=0, run_id="r",
        judge_model=model, prompt_version="p", judge_run_id="j",
        correctness=correctness, groundedness=1.0, evidence_completeness=1.0,
        reasoning_completeness=1.0, answer_relevance=1.0,
        answerable_from_context=True, hallucination_detected=False,
        failure_type="none", confidence=1.0,
    )
