"""Build complete budget-aware router training examples from scored JSONL outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.config import BudgetConfig
from src.router.constants import ACTION_IDS
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.features import query_features
from src.router.reward.oracle import latency_statistics, select_budget_aware_oracle
from src.router.types import CandidateOutcome, MetricVector, ScoredOutcome


def main() -> None:
    args = parse_args()
    groups: dict[tuple[str, int], list[ScoredOutcome]] = defaultdict(list)
    all_scored: list[ScoredOutcome] = []
    questions: dict[str, str] = {}
    for row in read_jsonl(args.scored_outcomes):
        outcome = CandidateOutcome.from_dict(row["outcome"])
        metric = MetricVector(**row["metrics"])
        scored = ScoredOutcome(outcome=outcome, metrics=metric, quality=row.get("quality"), compute_units=row.get("compute_units"))
        groups[(outcome.question_id, outcome.repeat_id)].append(scored)
        all_scored.append(scored)
        questions[outcome.question_id] = outcome.retrieval_record.question
    budgets = BudgetConfig(mean_latency_ms=args.mean_latency_ms, tail_latency_ms=args.tail_latency_ms, compute_units=args.compute_budget, max_request_latency_ms=args.max_request_latency_ms)
    budgets.validate()
    latency_reference = latency_statistics(all_scored)
    examples = []
    for (question_id, repeat_id), outcomes in sorted(groups.items()):
        if {outcome.outcome.candidate_id for outcome in outcomes} != set(ACTION_IDS):
            continue
        oracle = select_budget_aware_oracle(outcomes, budgets=budgets, latency_reference=latency_reference)
        by_action = {outcome.outcome.candidate_id: outcome for outcome in outcomes}
        examples.append({
            "question_id": question_id,
            "repeat_id": repeat_id,
            "question": questions[question_id],
            "runtime_features": query_features(questions[question_id]),
            "quality_by_action": {action: by_action[action].quality for action in ACTION_IDS},
            "latency_ms_by_action": {action: by_action[action].outcome.end_to_end_latency_ms for action in ACTION_IDS},
            "compute_by_action": {action: by_action[action].compute_units for action in ACTION_IDS},
            "failure_by_action": {action: bool(by_action[action].outcome.error_type or by_action[action].outcome.timeout) for action in ACTION_IDS},
            "feasible_action_mask": {action: bool(by_action[action].feasible) for action in ACTION_IDS},
            "oracle_action": oracle.outcome.candidate_id,
            "oracle_infeasibility": oracle.infeasibility,
            "budgets": {"mean_latency_ms": args.mean_latency_ms, "tail_latency_ms": args.tail_latency_ms, "compute_units": args.compute_budget, "max_request_latency_ms": args.max_request_latency_ms},
        })
    if not examples:
        raise RuntimeError("No complete three-action scored outcome groups were found")
    write_jsonl_atomic(args.output, examples)
    print(json.dumps({"training_examples": str(args.output), "records": len(examples)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build budget-aware, full-information router training examples.")
    parser.add_argument("--scored-outcomes", required=True)
    parser.add_argument("--output", default="artifacts/router/datasets/pilot_training.jsonl")
    parser.add_argument("--mean-latency-ms", type=float)
    parser.add_argument("--tail-latency-ms", type=float)
    parser.add_argument("--compute-budget", type=float)
    parser.add_argument("--max-request-latency-ms", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    main()
