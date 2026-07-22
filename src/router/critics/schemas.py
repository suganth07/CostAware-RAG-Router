"""Strict parsing for critic output; malformed ratings never silently enter rewards."""

from __future__ import annotations

import json
from typing import Any

FAILURE_TYPES = frozenset({
    "none",
    "incorrect_answer",
    "unsupported_claim",
    "incomplete_answer",
    "irrelevant_answer",
    "unanswerable_from_context",
    "insufficient_evidence",
    "missing_release_year",
    "execution_failure",
    "other",
})
SCORE_FIELDS = (
    "correctness",
    "groundedness",
    "evidence_completeness",
    "reasoning_completeness",
    "answer_relevance",
    "confidence",
)
LIST_FIELDS = ("supporting_claims", "unsupported_claims", "missing_information")
ANSWER_FIELDS = ("correctness", "answer_relevance", "confidence")


class CritiqueParseError(ValueError):
    pass


def parse_critique(text: str) -> dict[str, Any]:
    """Validate the entire response against the critic's narrow JSON contract."""
    cleaned = _strip_fence(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise CritiqueParseError(f"Critic did not return valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise CritiqueParseError("Critic response must be a JSON object")
    required = set(SCORE_FIELDS) | set(LIST_FIELDS) | {
        "answerable_from_context", "hallucination_detected", "failure_type", "critique"
    }
    missing = sorted(required.difference(value))
    unknown = sorted(set(value).difference(required))
    if missing or unknown:
        raise CritiqueParseError(f"Critic schema keys mismatch; missing={missing}, unknown={unknown}")
    for field in SCORE_FIELDS:
        score = value[field]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            raise CritiqueParseError(f"{field} must be a number in [0, 1]")
        value[field] = float(score)
    for field in ("answerable_from_context", "hallucination_detected"):
        if not isinstance(value[field], bool):
            raise CritiqueParseError(f"{field} must be boolean")
    # qwen3 occasionally returns an empty enum while providing a complete,
    # valid set of scalar scores. Preserve those scores but make the recovery
    # visible in the artifact; never infer a more specific failure category.
    value["failure_type_normalized"] = False
    if value["failure_type"] == "":
        value["failure_type"] = "other"
        value["failure_type_normalized"] = True
    elif value["failure_type"] not in FAILURE_TYPES:
        raise CritiqueParseError(f"Unsupported failure_type: {value['failure_type']!r}")
    for field in LIST_FIELDS:
        if not isinstance(value[field], list) or not all(isinstance(item, str) for item in value[field]):
            raise CritiqueParseError(f"{field} must be an array of strings")
    if not isinstance(value["critique"], str):
        raise CritiqueParseError("critique must be a string")
    return value


def parse_answer_critique(text: str) -> dict[str, Any]:
    """Validate the small, context-free global-correctness judgement."""
    try:
        value = json.loads(_strip_fence(text))
    except json.JSONDecodeError as exc:
        raise CritiqueParseError(f"Answer judge did not return valid JSON: {exc.msg}") from exc
    required = set(ANSWER_FIELDS) | {"failure_type", "critique"}
    if not isinstance(value, dict) or set(value) != required:
        raise CritiqueParseError("Answer judge schema keys do not match the required contract")
    for field in ANSWER_FIELDS:
        score = value[field]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            raise CritiqueParseError(f"{field} must be a number in [0, 1]")
        value[field] = float(score)
    value["failure_type_normalized"] = False
    if value["failure_type"] == "":
        value["failure_type"] = "other"
        value["failure_type_normalized"] = True
    elif value["failure_type"] not in FAILURE_TYPES:
        raise CritiqueParseError(f"Unsupported failure_type: {value['failure_type']!r}")
    if not isinstance(value["critique"], str):
        raise CritiqueParseError("critique must be a string")
    return value


def _strip_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
    return value
