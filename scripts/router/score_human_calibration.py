"""Validate completed blind reviews and report critic-versus-human agreement."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.critics.calibration import summarize_human_calibration
from src.router.critics.human_review import labels_from_reviewed_queue
from src.router.data.serialization import read_jsonl, write_json_atomic, write_jsonl_atomic
from src.router.types import CritiqueRecord


def main() -> None:
    args = parse_args()
    labels = labels_from_reviewed_queue(read_jsonl(args.reviewed_queue), read_jsonl(args.private_map))
    critiques = [CritiqueRecord(**_with_defaults(row)) for row in read_jsonl(args.critiques)]
    summary = summarize_human_calibration(critiques, labels)
    report = {
        "labels": summary.labels,
        "matched_critiques": summary.matched_critiques,
        "mean_absolute_error": summary.mean_absolute_error,
        "required_label_range": {"minimum": args.minimum_labels, "maximum": args.maximum_labels},
        "promotion_ready": args.minimum_labels <= summary.labels <= args.maximum_labels and summary.matched_critiques == summary.labels,
        "warning": "Promotion still requires a documented human decision and reward validation; this report does not change reward weights.",
    }
    write_jsonl_atomic(args.labels_output, [asdict(label) for label in labels])
    write_json_atomic(args.report_output, report)
    print(json.dumps({"labels": str(args.labels_output), "report": str(args.report_output), **report}, indent=2))


def _with_defaults(row: dict) -> dict:
    value = dict(row)
    value.setdefault("failure_type_normalized", False)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score completed human calibration labels against a critic artifact.")
    parser.add_argument("--reviewed-queue", required=True)
    parser.add_argument("--private-map", required=True)
    parser.add_argument("--critiques", required=True)
    parser.add_argument("--labels-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--minimum-labels", type=int, default=300)
    parser.add_argument("--maximum-labels", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    main()
