"""Blind human-review queue construction and strict label conversion."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable

from src.router.constants import ACTION_IDS
from src.router.critics.calibration import HumanCritiqueLabel
from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord


def build_blind_calibration_queue(
    critiques: Iterable[CritiqueRecord],
    outcomes: dict[tuple[str, str, int, str], CandidateOutcome],
    questions: dict[str, QuestionRecord],
    *,
    count: int,
    seed: int,
    context_chars: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a shareable blind queue plus a private join map.

    The queue never includes a candidate ID or raw outcome key. The mapping
    must stay with the experiment owner until annotated rows are joined back.
    """
    if count < 1:
        raise ValueError("count must be at least one")
    groups: dict[str, list[CritiqueRecord]] = defaultdict(list)
    for critique in critiques:
        key = _key(critique)
        if key in outcomes and critique.question_id in questions:
            groups[critique.candidate_id].append(critique)
    if any(not groups[action] for action in ACTION_IDS):
        raise ValueError("Every action needs at least one joinable critic record for a balanced blind queue")
    if count > sum(len(rows) for rows in groups.values()):
        raise ValueError("Requested more blind reviews than joinable critic records")
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    selected: list[CritiqueRecord] = []
    actions = list(ACTION_IDS)
    while actions and len(selected) < count:
        next_actions = []
        for action in actions:
            if len(selected) >= count:
                break
            if groups[action]:
                selected.append(groups[action].pop())
            if groups[action]:
                next_actions.append(action)
        actions = next_actions

    queue: list[dict[str, Any]] = []
    join_map: list[dict[str, Any]] = []
    for index, critique in enumerate(selected, start=1):
        review_id = f"human-calibration-{index:04d}"
        outcome = outcomes[_key(critique)]
        question = questions[critique.question_id]
        queue.append(
            {
                "review_id": review_id,
                "question": question.question,
                "accepted_answers": question.accepted_answers,
                "gold_evidence": question.gold_evidence,
                "retrieved_context": outcome.retrieval_record.context_text[:context_chars],
                "generated_answer": outcome.generated_answer,
                "critic": {
                    "correctness": critique.correctness,
                    "groundedness": critique.groundedness,
                    "evidence_completeness": critique.evidence_completeness,
                    "reasoning_completeness": critique.reasoning_completeness,
                    "answer_relevance": critique.answer_relevance,
                    "confidence": critique.confidence,
                    "hallucination_detected": critique.hallucination_detected,
                    "failure_type": critique.failure_type,
                },
                "human_review": {
                    "answer_correct": None,
                    "answer_grounded": None,
                    "answer_complete": None,
                    "notes": "",
                },
            }
        )
        join_map.append(
            {
                "review_id": review_id,
                "question_id": critique.question_id,
                "candidate_id": critique.candidate_id,
                "repeat_id": critique.repeat_id,
                "run_id": critique.run_id,
            }
        )
    return queue, join_map


def labels_from_reviewed_queue(
    queue_rows: Iterable[dict[str, Any]], join_rows: Iterable[dict[str, Any]]
) -> list[HumanCritiqueLabel]:
    join_map = {str(row["review_id"]): row for row in join_rows}
    rows = list(queue_rows)
    if len(join_map) != len(rows):
        raise ValueError("Reviewed queue and private join map have different row counts")
    labels: list[HumanCritiqueLabel] = []
    seen: set[str] = set()
    for row in rows:
        review_id = str(row.get("review_id") or "")
        if not review_id or review_id in seen or review_id not in join_map:
            raise ValueError("Every reviewed row needs one unique review_id present in the private join map")
        seen.add(review_id)
        review = row.get("human_review") or {}
        join = join_map[review_id]
        label = HumanCritiqueLabel(
            question_id=str(join["question_id"]),
            candidate_id=str(join["candidate_id"]),
            repeat_id=int(join["repeat_id"]),
            run_id=str(join["run_id"]),
            correctness=_binary_score(review.get("answer_correct"), review_id, "answer_correct"),
            groundedness=_binary_score(review.get("answer_grounded"), review_id, "answer_grounded"),
            completeness=_binary_score(review.get("answer_complete"), review_id, "answer_complete"),
        )
        label.validate()
        labels.append(label)
    return labels


def _binary_score(value: Any, review_id: str, field: str) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) in {0.0, 1.0}:
        return float(value)
    raise ValueError(f"{review_id}: human_review.{field} must be true/false or 1/0")


def _key(item: CritiqueRecord) -> tuple[str, str, int, str]:
    return (item.question_id, item.candidate_id, item.repeat_id, item.run_id)
