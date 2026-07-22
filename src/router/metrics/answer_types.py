"""Answer-type-aware deterministic scoring and hard conflict guards."""

from __future__ import annotations

import re
from typing import Sequence

from src.evaluation.core import answer_kind, boolean_accuracy, exact_match, normalize_text, token_f1


_NUMBER_RE = re.compile(r"(?<!\w)-?\d+(?:\.\d+)?")
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def infer_answer_type(answers: Sequence[str]) -> str:
    kind = answer_kind(answers)
    if kind in {"boolean", "null"}:
        return kind
    normalized = [normalize_text(answer) for answer in answers]
    if normalized and all(_NUMBER_RE.fullmatch(answer.replace(" ", "")) for answer in normalized):
        return "number"
    if normalized and all(_DATE_RE.search(answer) for answer in normalized):
        return "date"
    return "multi_entity_set" if len(answers) > 1 else "short_span"


def deterministic_correctness(prediction: str, answers: Sequence[str]) -> dict[str, float | bool | None]:
    kind = infer_answer_type(answers)
    exact = exact_match(prediction, answers)
    boolean = boolean_accuracy(prediction, answers)
    numeric_conflict = kind == "number" and _number_conflict(prediction, answers)
    date_conflict = kind == "date" and _date_conflict(prediction, answers)
    polarity_conflict = kind == "boolean" and boolean == 0.0
    entity_conflict = bool(prediction.strip()) and exact == 0.0 and token_f1(prediction, answers) == 0.0 and kind in {"short_span", "multi_entity_set"}
    # Hard types must never receive credit through a future semantic metric.
    if numeric_conflict or date_conflict or polarity_conflict:
        f1 = 0.0
    else:
        f1 = token_f1(prediction, answers)
    return {
        "answer_type": kind,
        "exact": exact,
        "f1": f1,
        "boolean": boolean,
        "numeric_date_accuracy": exact if kind in {"number", "date"} else None,
        "numeric_conflict": numeric_conflict or date_conflict,
        "polarity_conflict": polarity_conflict,
        "entity_conflict": entity_conflict,
    }


def _number_conflict(prediction: str, answers: Sequence[str]) -> bool:
    predicted = set(_NUMBER_RE.findall(prediction.replace(",", "")))
    expected = {number for answer in answers for number in _NUMBER_RE.findall(str(answer).replace(",", ""))}
    return bool(predicted and expected and predicted.isdisjoint(expected))


def _date_conflict(prediction: str, answers: Sequence[str]) -> bool:
    predicted = set(_DATE_RE.findall(prediction))
    expected = {date for answer in answers for date in _DATE_RE.findall(str(answer))}
    return bool(predicted and expected and predicted.isdisjoint(expected))
