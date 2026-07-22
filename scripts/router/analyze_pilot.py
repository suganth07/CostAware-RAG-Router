"""Audit a scored router pilot before semantic metrics or policy training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.config import BudgetConfig
from src.router.constants import ACTION_IDS
from src.router.data.serialization import read_jsonl
from src.router.data.validation import validate_complete_matrix
from src.router.reward.oracle import latency_statistics, select_budget_aware_oracle
from src.router.types import CandidateOutcome, MetricVector, ScoredOutcome


def main() -> None:
    args = parse_args()
    scored = _load_scored(args.scored_outcomes)
    validate_complete_matrix([item.outcome for item in scored], repeats=args.repeats)
    budget = BudgetConfig(
        mean_latency_ms=args.mean_latency_ms,
        tail_latency_ms=args.tail_latency_ms,
        compute_units=args.compute_budget,
        max_request_latency_ms=args.max_request_latency_ms,
    )
    budget.validate()
    summary = _summarize(scored, budget=budget, repeats=args.repeats)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": summary["overall"], "warnings": summary["warnings"]}, indent=2))


def _load_scored(path: str) -> list[ScoredOutcome]:
    result = []
    for row in read_jsonl(path):
        result.append(
            ScoredOutcome(
                outcome=CandidateOutcome.from_dict(row["outcome"]),
                metrics=MetricVector(**row["metrics"]),
                quality=row.get("quality"),
                compute_units=row.get("compute_units"),
            )
        )
    if not result:
        raise RuntimeError("No scored outcomes found")
    return result


def _summarize(scored: list[ScoredOutcome], *, budget: BudgetConfig, repeats: int) -> dict:
    latency = latency_statistics(scored)
    per_action: dict[str, dict] = {}
    for action in ACTION_IDS:
        rows = [item for item in scored if item.outcome.candidate_id == action]
        qualities = [float(item.quality) for item in rows if item.quality is not None]
        computes = [float(item.compute_units) for item in rows if item.compute_units is not None]
        latencies = [float(item.outcome.end_to_end_latency_ms) for item in rows if item.outcome.end_to_end_latency_ms is not None]
        per_action[action] = {
            "outcomes": len(rows),
            "errors": sum(bool(item.outcome.error_type or item.outcome.timeout) for item in rows),
            "quality": _distribution(qualities),
            "compute_units": _distribution(computes),
            "latency_ms": _distribution(latencies),
            "latency_reference": latency.get(action, {}),
        }
    grouped: dict[tuple[str, int], list[ScoredOutcome]] = defaultdict(list)
    for item in scored:
        grouped[(item.outcome.question_id, item.outcome.repeat_id)].append(item)
    oracle_actions = Counter()
    feasible_by_action = Counter()
    least_violation = 0
    for outcomes in grouped.values():
        selected = select_budget_aware_oracle(outcomes, budgets=budget, latency_reference=latency)
        oracle_actions[selected.outcome.candidate_id] += 1
        if not selected.feasible:
            least_violation += 1
        for outcome in outcomes:
            # `select_budget_aware_oracle` applies feasibility in place.
            feasible_by_action[outcome.outcome.candidate_id] += int(bool(outcome.feasible))
    expected_groups = len(grouped)
    warnings = []
    if least_violation == expected_groups:
        warnings.append("Every question has zero feasible actions under the selected budget; do not train a constrained policy from this configuration.")
    if any(per_action[action]["errors"] for action in ACTION_IDS):
        warnings.append("At least one candidate produced an error or timeout; inspect failures before increasing collection size.")
    if len({action for action, count in oracle_actions.items() if count}) == 1:
        warnings.append("The oracle selects one action for every question; inspect reward/budget settings before training because the policy may collapse.")
    return {
        "overall": {
            "raw_outcomes": len(scored),
            "question_repeat_groups": expected_groups,
            "questions": len({item.outcome.question_id for item in scored}),
            "repeats": repeats,
            "actions": list(ACTION_IDS),
            "budget": {
                "mean_latency_ms": budget.mean_latency_ms,
                "tail_latency_ms": budget.tail_latency_ms,
                "compute_units": budget.compute_units,
                "max_request_latency_ms": budget.max_request_latency_ms,
            },
        },
        "per_action": per_action,
        "oracle_action_distribution": dict(oracle_actions),
        "feasible_action_counts": dict(feasible_by_action),
        "least_violation_groups": least_violation,
        "warnings": warnings,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "minimum": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "p50": _quantile(ordered, 0.50),
        "p95": _quantile(ordered, 0.95),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _quantile(values: list[float], fraction: float) -> float:
    return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and summarize a scored full-information router pilot.")
    parser.add_argument("--scored-outcomes", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mean-latency-ms", type=float)
    parser.add_argument("--tail-latency-ms", type=float)
    parser.add_argument("--compute-budget", type=float)
    parser.add_argument("--max-request-latency-ms", type=float)
    parser.add_argument("--output", default="reports/router/data_audit/pilot_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
