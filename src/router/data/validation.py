"""Contract checks that stop partial or incompatible matrices entering training."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.router.constants import ACTION_IDS
from src.router.exceptions import OutcomeContractError
from src.router.types import CandidateOutcome


def validate_complete_matrix(outcomes: Iterable[CandidateOutcome], *, repeats: int) -> None:
    grouped: dict[tuple[str, int], set[str]] = defaultdict(set)
    run_ids: set[str] = set()
    for outcome in outcomes:
        grouped[(outcome.question_id, outcome.repeat_id)].add(outcome.candidate_id)
        run_ids.add(outcome.run_id)
    if len(run_ids) > 1:
        raise OutcomeContractError("One training matrix must have one run_id")
    expected = set(ACTION_IDS)
    missing = [key for key, actions in grouped.items() if actions != expected]
    if missing:
        raise OutcomeContractError(f"Incomplete candidate matrix for {len(missing)} question/repeat records; first={missing[0]}")
    expected_repeats = set(range(repeats))
    by_question: dict[str, set[int]] = defaultdict(set)
    for question_id, repeat_id in grouped:
        by_question[question_id].add(repeat_id)
    invalid = [question_id for question_id, seen in by_question.items() if seen != expected_repeats]
    if invalid:
        raise OutcomeContractError(f"Missing repeats for {len(invalid)} questions; first={invalid[0]}")
