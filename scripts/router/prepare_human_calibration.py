"""Prepare a candidate-blind human calibration queue from critic records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.critics.human_review import build_blind_calibration_queue
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord


def main() -> None:
    args = parse_args()
    critiques = [CritiqueRecord(**_with_defaults(row)) for row in read_jsonl(args.critiques)]
    outcomes = {
        (item.question_id, item.candidate_id, item.repeat_id, item.run_id): item
        for item in (CandidateOutcome.from_dict(row) for row in read_jsonl(args.outcomes))
    }
    questions = {row["question_id"]: QuestionRecord(**row) for row in read_jsonl(args.questions)}
    queue, join_map = build_blind_calibration_queue(
        critiques,
        outcomes,
        questions,
        count=args.count,
        seed=args.seed,
        context_chars=args.context_chars,
    )
    write_jsonl_atomic(args.queue_output, queue)
    write_jsonl_atomic(args.private_map_output, join_map)
    print(json.dumps({"review_queue": str(args.queue_output), "private_join_map": str(args.private_map_output), "records": len(queue), "warning": "Do not share the private join map with human reviewers."}, indent=2))


def _with_defaults(row: dict) -> dict:
    value = dict(row)
    value.setdefault("failure_type_normalized", False)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a candidate-blind human calibration queue and private join map.")
    parser.add_argument("--critiques", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-chars", type=int, default=4000)
    parser.add_argument("--queue-output", required=True, type=Path)
    parser.add_argument("--private-map-output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
