"""Create a deterministic stratified latency subset from a frozen pilot manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.data.serialization import read_jsonl, write_jsonl_atomic


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.questions)
    if args.max_questions < 1 or args.max_questions > len(rows):
        raise ValueError("--max-questions must be between 1 and the source question count")
    selected = stratified_rows(rows, limit=args.max_questions, seed=args.seed)
    write_jsonl_atomic(args.output, selected)
    print(json.dumps({"latency_subset": str(args.output), "questions": len(selected), "source_questions": len(rows), "seed": args.seed}, indent=2))


def stratified_rows(rows: list[dict], *, limit: int, seed: int) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        strata = row.get("sampling_strata") or {}
        buckets[(str(strata.get("question_type") or "unknown"), str(strata.get("answer_type") or "unknown"))].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict] = []
    keys = sorted(buckets)
    while keys and len(selected) < limit:
        next_keys = []
        for key in keys:
            if len(selected) >= limit:
                break
            if buckets[key]:
                selected.append(buckets[key].pop())
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a reproducible stratified latency subset from pilot questions.")
    parser.add_argument("--questions", required=True, help="Pilot .questions.jsonl")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
