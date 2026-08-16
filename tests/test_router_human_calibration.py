from __future__ import annotations

from src.evaluation.core import RetrievalRecord
from src.router.constants import ACTION_IDS
from src.router.critics.human_review import build_blind_calibration_queue, labels_from_reviewed_queue
from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord


def test_blind_queue_hides_candidate_and_restores_validated_labels() -> None:
    critiques = []
    outcomes = {}
    question = QuestionRecord(question_id="q", dataset="test", source_split="train", question="Question?", accepted_answers=["Answer"])
    for action in ACTION_IDS:
        outcome = CandidateOutcome(
            question_id="q", candidate_id=action, repeat_id=0, run_id="run",
            pipeline_fingerprint=f"pipeline-{action}", corpus_fingerprint="corpus", model_fingerprint="model", hardware_fingerprint="hardware",
            retrieval_record=RetrievalRecord(system=action, sample_id="q", question="Question?", context_text="Evidence"),
            generated_answer="Answer",
        )
        outcomes[outcome.key] = outcome
        critiques.append(CritiqueRecord(
            question_id="q", candidate_id=action, repeat_id=0, run_id="run", judge_model="judge", prompt_version="v1", judge_run_id="judge-run",
            correctness=1.0, groundedness=1.0, evidence_completeness=1.0, reasoning_completeness=1.0, answer_relevance=1.0,
            answerable_from_context=True, hallucination_detected=False, failure_type="none", confidence=1.0,
        ))

    queue, join_map = build_blind_calibration_queue(critiques, outcomes, {"q": question}, count=3, seed=7, context_chars=100)
    assert all("candidate_id" not in row for row in queue)
    assert {row["candidate_id"] for row in join_map} == set(ACTION_IDS)
    for row in queue:
        row["human_review"].update(answer_correct=True, answer_grounded=True, answer_complete=True)

    labels = labels_from_reviewed_queue(queue, join_map)
    assert len(labels) == 3
    assert all(label.correctness == label.groundedness == label.completeness == 1.0 for label in labels)
