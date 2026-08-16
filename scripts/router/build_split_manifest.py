"""Build a leakage-audited grouped router split manifest from frozen questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.data import SplitFractions, build_grouped_split_manifest
from src.router.data.serialization import read_jsonl, write_json_atomic, write_jsonl_atomic
from src.router.types import QuestionRecord


def main() -> None:
    args = parse_args()
    questions = [QuestionRecord(**row) for row in read_jsonl(args.questions)]
    manifest = build_grouped_split_manifest(
        questions,
        group_by=args.group_by,
        cross_domain_datasets=set(args.cross_domain_dataset),
        fractions=SplitFractions(args.train_fraction, args.validation_fraction, args.locked_test_fraction),
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments = output_dir / f"{args.name}.assignments.jsonl"
    summary = output_dir / f"{args.name}.manifest.json"
    write_jsonl_atomic(assignments, manifest["assignments"])
    write_json_atomic(summary, {key: value for key, value in manifest.items() if key != "assignments"})
    print(json.dumps({"assignments": str(assignments), "manifest": str(summary), "summary": manifest["summary"]}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-aware train/validation/locked-test router manifests.")
    parser.add_argument("--questions", required=True, help="Frozen QuestionRecord JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group-by", choices=("template", "entity", "template_entity"), default="template")
    parser.add_argument("--cross-domain-dataset", action="append", default=[], help="Dataset name held out entirely for cross-domain evaluation; repeat flag for multiple datasets.")
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--locked-test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main()
