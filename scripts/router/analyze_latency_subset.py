"""Audit three logical latency repeats while reusing a compatible main-pilot repeat.

The first logical repeat is selected from the completed M4 main matrix. Two
new repeats are read from a smaller, separate collection. Raw artifacts are
never merged or rewritten: their individual run fingerprints remain intact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.router.analyze_pilot import summarize_pilot
from src.router.config import BudgetConfig
from src.router.constants import ACTION_IDS
from src.router.data import fingerprint_payload, validate_complete_matrix
from src.router.data.serialization import read_jsonl
from src.router.types import CandidateOutcome, MetricVector, ScoredOutcome


def main() -> None:
    args = parse_args()
    question_ids = _question_ids(args.questions)
    source = _load_scored(args.source_scored_outcomes)
    supplementary = _load_scored(args.supplementary_scored_outcomes)
    source_rows = _select_rows(source, question_ids, repeat_ids={args.source_repeat_id})
    supplementary_rows = _select_rows(supplementary, question_ids, repeat_ids=set(range(args.supplementary_repeats)))
    _validate_source_matrix(source_rows, questions=len(question_ids), repeats=1, label="source")
    _validate_source_matrix(supplementary_rows, questions=len(question_ids), repeats=args.supplementary_repeats, label="supplementary")
    _verify_compatible(source_rows, supplementary_rows)

    cohort_id = fingerprint_payload(
        {
            "kind": "derived_latency_repeat_cohort",
            "questions": sorted(question_ids),
            "source_runs": sorted({item.outcome.run_id for item in source_rows}),
            "supplementary_runs": sorted({item.outcome.run_id for item in supplementary_rows}),
            "logical_repeats": 1 + args.supplementary_repeats,
        }
    )
    logical = [
        _logical_repeat(item, repeat_id=0, cohort_id=cohort_id)
        for item in source_rows
    ]
    logical.extend(
        _logical_repeat(item, repeat_id=item.outcome.repeat_id + 1, cohort_id=cohort_id)
        for item in supplementary_rows
    )
    total_repeats = 1 + args.supplementary_repeats
    validate_complete_matrix([item.outcome for item in logical], repeats=total_repeats)
    budget = BudgetConfig(
        mean_latency_ms=args.mean_latency_ms,
        tail_latency_ms=args.tail_latency_ms,
        compute_units=args.compute_budget,
        max_request_latency_ms=args.max_request_latency_ms,
    )
    budget.validate()
    report = summarize_pilot(logical, budget=budget, repeats=total_repeats)
    report["latency_repeat_provenance"] = {
        "derived_cohort_id": cohort_id,
        "questions": len(question_ids),
        "logical_repeats": total_repeats,
        "source": {
            "path": str(args.source_scored_outcomes),
            "source_repeat_id": args.source_repeat_id,
            "run_ids": sorted({item.outcome.run_id for item in source_rows}),
            "rows": len(source_rows),
        },
        "supplementary": {
            "path": str(args.supplementary_scored_outcomes),
            "source_repeat_ids": list(range(args.supplementary_repeats)),
            "run_ids": sorted({item.outcome.run_id for item in supplementary_rows}),
            "rows": len(supplementary_rows),
        },
        "raw_artifacts_preserved": True,
    }
    report["within_question_latency_variation"] = _within_question_variation(logical)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(output),
                "logical_outcomes": len(logical),
                "logical_repeats": total_repeats,
                "within_question_latency_variation": report["within_question_latency_variation"],
                "warnings": report["warnings"],
            },
            indent=2,
        )
    )


def _question_ids(path: str) -> set[str]:
    rows = read_jsonl(path)
    values = {str(row["question_id"]) for row in rows}
    if not values or len(values) != len(rows):
        raise ValueError("--questions must contain non-empty, unique QuestionRecord rows")
    return values


def _load_scored(path: str) -> list[ScoredOutcome]:
    rows = []
    for value in read_jsonl(path):
        rows.append(
            ScoredOutcome(
                outcome=CandidateOutcome.from_dict(value["outcome"]),
                metrics=MetricVector(**value["metrics"]),
                quality=value.get("quality"),
                compute_units=value.get("compute_units"),
            )
        )
    if not rows:
        raise ValueError(f"No scored outcomes found in {path}")
    return rows


def _select_rows(rows: list[ScoredOutcome], question_ids: set[str], *, repeat_ids: set[int]) -> list[ScoredOutcome]:
    return [
        item
        for item in rows
        if item.outcome.question_id in question_ids and item.outcome.repeat_id in repeat_ids
    ]


def _validate_source_matrix(rows: list[ScoredOutcome], *, questions: int, repeats: int, label: str) -> None:
    expected_rows = questions * repeats * len(ACTION_IDS)
    if len(rows) != expected_rows:
        raise ValueError(f"{label} rows={len(rows)}, expected {expected_rows}; do not use an incomplete latency subset")
    validate_complete_matrix([item.outcome for item in rows], repeats=repeats)


def _verify_compatible(source: list[ScoredOutcome], supplementary: list[ScoredOutcome]) -> None:
    def values(rows: list[ScoredOutcome], attribute: str) -> set[str]:
        return {str(getattr(item.outcome, attribute)) for item in rows}

    for attribute in ("corpus_fingerprint", "model_fingerprint", "hardware_fingerprint"):
        source_values = values(source, attribute)
        supplementary_values = values(supplementary, attribute)
        if len(source_values) != 1 or len(supplementary_values) != 1 or source_values != supplementary_values:
            raise ValueError(
                f"Latency cohorts have incompatible {attribute}; collect all three repeats on the same corpus, model, and hardware."
            )
    for action in ACTION_IDS:
        source_values = {item.outcome.pipeline_fingerprint for item in source if item.outcome.candidate_id == action}
        supplementary_values = {item.outcome.pipeline_fingerprint for item in supplementary if item.outcome.candidate_id == action}
        if len(source_values) != 1 or source_values != supplementary_values:
            raise ValueError(f"Latency cohorts have incompatible pipeline fingerprint for {action}")


def _logical_repeat(item: ScoredOutcome, *, repeat_id: int, cohort_id: str) -> ScoredOutcome:
    return replace(item, outcome=replace(item.outcome, repeat_id=repeat_id, run_id=cohort_id))


def _within_question_variation(rows: list[ScoredOutcome]) -> dict[str, dict[str, float | int | None]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for item in rows:
        latency = item.outcome.end_to_end_latency_ms
        if latency is not None:
            values[item.outcome.candidate_id][item.outcome.question_id].append(float(latency))
    report: dict[str, dict[str, float | int | None]] = {}
    for action in ACTION_IDS:
        groups = list(values[action].values())
        ranges = [max(group) - min(group) for group in groups if len(group) >= 2]
        deviations = [_population_std(group) for group in groups if len(group) >= 2]
        report[action] = {
            "questions_with_all_latency_values": sum(len(group) >= 3 for group in groups),
            "mean_within_question_range_ms": _mean(ranges),
            "mean_within_question_std_ms": _mean(deviations),
            "max_within_question_range_ms": max(ranges) if ranges else None,
        }
    return report


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine one compatible main-pilot latency repeat with new repeats without rewriting raw artifacts.")
    parser.add_argument("--source-scored-outcomes", required=True)
    parser.add_argument("--supplementary-scored-outcomes", required=True)
    parser.add_argument("--questions", required=True, help="Frozen latency-subset QuestionRecord JSONL")
    parser.add_argument("--source-repeat-id", type=int, default=0)
    parser.add_argument("--supplementary-repeats", type=int, default=2)
    parser.add_argument("--mean-latency-ms", type=float)
    parser.add_argument("--tail-latency-ms", type=float)
    parser.add_argument("--compute-budget", type=float)
    parser.add_argument("--max-request-latency-ms", type=float)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
