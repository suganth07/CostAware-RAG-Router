"""Audit structured-critic coverage, bias indicators, and proxy agreement."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.constants import ACTION_IDS
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord
from src.evaluation.core import normalize_text

SCORE_FIELDS = (
    "correctness",
    "groundedness",
    "evidence_completeness",
    "reasoning_completeness",
    "answer_relevance",
    "confidence",
)


def main() -> None:
    args = parse_args()
    critiques = [CritiqueRecord(**_with_defaults(row)) for row in read_jsonl(args.critiques)]
    if not critiques:
        raise RuntimeError("No critic records found")
    outcomes = {outcome_key(item): item for item in (CandidateOutcome.from_dict(row) for row in read_jsonl(args.outcomes))}
    questions = {row["question_id"]: QuestionRecord(**row) for row in read_jsonl(args.questions)}
    deterministic = {
        outcome_key_from_row(row["outcome"]): row["metrics"]
        for row in read_jsonl(args.scored_outcomes)
    }
    summary = build_summary(critiques, outcomes, questions, deterministic, threshold=args.correctness_threshold)
    write_json(Path(args.output), summary)
    review_rows = build_blind_review_sample(
        critiques, outcomes, questions,
        count=args.manual_review_records,
        seed=args.seed,
        context_chars=args.review_context_chars,
    )
    write_jsonl_atomic(args.review_output, review_rows)
    print(json.dumps({
        "report": str(args.output),
        "blind_review_queue": str(args.review_output),
        "records": len(critiques),
        "warnings": summary["warnings"],
    }, indent=2))


def build_summary(
    critiques: list[CritiqueRecord],
    outcomes: dict[tuple[str, str, int, str], CandidateOutcome],
    questions: dict[str, QuestionRecord],
    deterministic: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    per_action: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    join_missing = 0
    coverage_labels = {
        critique_key(item): accepted_answer_coverage(outcomes[critique_key(item)].generated_answer, questions[item.question_id].accepted_answers)
        for item in critiques
        if critique_key(item) in outcomes and item.question_id in questions
    }
    for action in ACTION_IDS:
        rows = [item for item in critiques if item.candidate_id == action]
        joined = [(item, deterministic.get(critique_key(item))) for item in rows]
        join_missing += sum(metrics is None or critique_key(item) not in outcomes for item, metrics in joined)
        per_action[action] = action_summary(rows, joined, threshold, coverage_labels)
    coverage = Counter(item.candidate_id for item in critiques)
    repair_count = sum(item.failure_type_normalized for item in critiques)
    if any(coverage[action] == 0 for action in ACTION_IDS):
        warnings.append("At least one route has no critic coverage; do not compare critic scores across actions.")
    if max(coverage.values()) - min(coverage.values()) > 1:
        warnings.append("Critic action counts are not balanced; candidate-level comparisons may be confounded.")
    if repair_count:
        warnings.append(f"{repair_count}/{len(critiques)} records normalized an empty failure_type to 'other'; audit this model/prompt behaviour before using the taxonomy.")
    if join_missing:
        warnings.append(f"{join_missing} critic records could not be joined to a raw/scored outcome.")
    exact_proxy = aggregate_proxy(per_action, "deterministic_exact_match_proxy")
    coverage_proxy = aggregate_proxy(per_action, "accepted_answer_coverage_proxy")
    if coverage_proxy["count"] and coverage_proxy["precision"] == 0.0 and coverage_proxy["positive_labels"]:
        warnings.append("Critic correctness has zero precision against accepted-answer coverage proxy; reject this critic version for reward promotion.")
    if not coverage_proxy["positive_labels"]:
        warnings.append("The audited slice has no accepted-answer-coverage positives, so critic recall cannot be estimated from this proxy.")
    warnings.append("Deterministic exact-match agreement is only a proxy, not a human-validity measurement. Do not promote critic scores into RL reward before reviewing the blind review queue.")
    return {
        "overall": {
            "critic_records": len(critiques),
            "actions": list(ACTION_IDS),
            "action_coverage": dict(coverage),
            "failure_type_normalized_count": repair_count,
            "failure_type_distribution": dict(Counter(item.failure_type for item in critiques)),
            "judge_models": sorted({item.judge_model for item in critiques}),
            "prompt_versions": sorted({item.prompt_version for item in critiques}),
            "judge_run_ids": sorted({item.judge_run_id for item in critiques}),
            "correctness_threshold": threshold,
            "join_missing": join_missing,
        },
        "per_action": per_action,
        "aggregate_deterministic_exact_match_proxy": exact_proxy,
        "aggregate_accepted_answer_coverage_proxy": coverage_proxy,
        "promotion_decision": "reject" if coverage_proxy["count"] and coverage_proxy["precision"] == 0.0 and coverage_proxy["positive_labels"] else "human_review_required",
        "warnings": warnings,
    }


def action_summary(
    rows: list[CritiqueRecord],
    joined: list[tuple[CritiqueRecord, dict[str, Any] | None]],
    threshold: float,
    coverage_labels: dict[tuple[str, str, int, str], bool],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "records": len(rows),
        "failure_type_distribution": dict(Counter(item.failure_type for item in rows)),
        "failure_type_normalized_count": sum(item.failure_type_normalized for item in rows),
        "score_distribution": {field: distribution([float(getattr(item, field)) for item in rows]) for field in SCORE_FIELDS},
        "hallucination_rate": mean([float(item.hallucination_detected) for item in rows]),
        "answerable_from_context_rate": mean([float(item.answerable_from_context) for item in rows]),
    }
    proxy_rows = [(item, metrics) for item, metrics in joined if metrics is not None and metrics.get("exact_or_alias_match") is not None]
    output["deterministic_exact_match_proxy"] = agreement(proxy_rows, threshold)
    coverage_rows = [(item, coverage_labels[critique_key(item)]) for item in rows if critique_key(item) in coverage_labels]
    output["accepted_answer_coverage_proxy"] = agreement_labels(coverage_rows, threshold)
    return output


def aggregate_proxy(per_action: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    totals = Counter()
    for summary in per_action.values():
        confusion = summary[key].get("confusion", {})
        totals.update(confusion)
    tp = totals["critic_positive_exact_positive"]
    fp = totals["critic_positive_exact_negative"]
    fn = totals["critic_negative_exact_positive"]
    count = sum(totals.values())
    return {
        "count": count,
        "accuracy": safe_divide(tp + totals["critic_negative_exact_negative"], count),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "positive_labels": tp + fn,
        "confusion": dict(totals),
    }


def agreement(rows: list[tuple[CritiqueRecord, dict[str, Any]]], threshold: float) -> dict[str, Any]:
    confusion = Counter()
    for critique, metrics in rows:
        predicted = critique.correctness >= threshold
        observed = float(metrics["exact_or_alias_match"]) >= 0.5
        confusion[(predicted, observed)] += 1
    return agreement_from_confusion(confusion)


def agreement_labels(rows: list[tuple[CritiqueRecord, bool]], threshold: float) -> dict[str, Any]:
    confusion = Counter()
    for critique, observed in rows:
        confusion[(critique.correctness >= threshold, observed)] += 1
    return agreement_from_confusion(confusion)


def agreement_from_confusion(confusion: Counter) -> dict[str, Any]:
    count = sum(confusion.values())
    if not count:
        return {"count": 0, "accuracy": None, "precision": None, "recall": None, "positive_labels": 0, "confusion": {}}
    true_positive = confusion[(True, True)]
    false_positive = confusion[(True, False)]
    false_negative = confusion[(False, True)]
    correct = true_positive + confusion[(False, False)]
    return {
        "count": count,
        "accuracy": correct / count,
        "precision": safe_divide(true_positive, true_positive + false_positive),
        "recall": safe_divide(true_positive, true_positive + false_negative),
        "positive_labels": true_positive + false_negative,
        "confusion": {
            "critic_positive_exact_positive": true_positive,
            "critic_positive_exact_negative": false_positive,
            "critic_negative_exact_positive": false_negative,
            "critic_negative_exact_negative": confusion[(False, False)],
        },
    }


def accepted_answer_coverage(prediction: str, answers: list[str]) -> bool:
    """Transparent multi-answer proxy where exact full-string match is too strict."""
    normalized_prediction = normalize_text(prediction)
    normalized_answers = [normalize_text(answer) for answer in answers if normalize_text(answer)]
    return bool(normalized_prediction and normalized_answers and all(answer in normalized_prediction for answer in normalized_answers))


def build_blind_review_sample(
    critiques: list[CritiqueRecord],
    outcomes: dict[tuple[str, str, int, str], CandidateOutcome],
    questions: dict[str, QuestionRecord],
    *, count: int,
    seed: int,
    context_chars: int,
) -> list[dict[str, Any]]:
    """Create a balanced review queue without candidate IDs to avoid human route bias."""
    groups: dict[str, list[CritiqueRecord]] = defaultdict(list)
    for critique in critiques:
        groups[critique.candidate_id].append(critique)
    rng = random.Random(seed)
    for records in groups.values():
        rng.shuffle(records)
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
    queue = []
    for index, critique in enumerate(selected, start=1):
        outcome = outcomes[critique_key(critique)]
        question = questions[critique.question_id]
        queue.append({
            "review_id": f"critic-review-{index:03d}",
            "question": question.question,
            "accepted_answers": question.accepted_answers,
            "gold_evidence": question.gold_evidence,
            "retrieved_context": outcome.retrieval_record.context_text[:context_chars],
            "generated_answer": outcome.generated_answer,
            # Keep route identity and join metadata out of the human-review
            # artifact. Aggregate reports retain those fields for bias checks.
            "critic": {
                "correctness": critique.correctness,
                "groundedness": critique.groundedness,
                "evidence_completeness": critique.evidence_completeness,
                "reasoning_completeness": critique.reasoning_completeness,
                "answer_relevance": critique.answer_relevance,
                "answerable_from_context": critique.answerable_from_context,
                "hallucination_detected": critique.hallucination_detected,
                "failure_type": critique.failure_type,
                "failure_type_normalized": critique.failure_type_normalized,
                "confidence": critique.confidence,
                "supporting_claims": critique.supporting_claims,
                "unsupported_claims": critique.unsupported_claims,
                "missing_information": critique.missing_information,
                "critique": critique.critique,
            },
            "human_review_template": {
                "answer_correct": None,
                "answer_grounded": None,
                "critic_correctness_appropriate": None,
                "critic_groundedness_appropriate": None,
                "notes": "",
            },
        })
    return queue


def outcome_key(item: CandidateOutcome) -> tuple[str, str, int, str]:
    return (item.question_id, item.candidate_id, item.repeat_id, item.run_id)


def outcome_key_from_row(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (str(row["question_id"]), str(row["candidate_id"]), int(row["repeat_id"]), str(row["run_id"]))


def critique_key(item: CritiqueRecord) -> tuple[str, str, int, str]:
    return (item.question_id, item.candidate_id, item.repeat_id, item.run_id)


def _with_defaults(row: dict[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.setdefault("failure_type_normalized", False)
    return value


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "minimum": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": quantile(ordered, 0.5),
        "p95": quantile(ordered, 0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def quantile(values: list[float], fraction: float) -> float:
    return values[round((len(values) - 1) * fraction)]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate-blind router critic output before reward promotion.")
    parser.add_argument("--critiques", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--scored-outcomes", required=True)
    parser.add_argument("--output", default="reports/router/critic_audit/pilot.json")
    parser.add_argument("--review-output", default="artifacts/router/review_queue/critic_blind_review.jsonl")
    parser.add_argument("--manual-review-records", type=int, default=20)
    parser.add_argument("--review-context-chars", type=int, default=4000)
    parser.add_argument("--correctness-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main()
