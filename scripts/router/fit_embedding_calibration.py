"""Fit plan.md embedding-calibration artifacts from human-labelled answer pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.data.serialization import read_jsonl
from src.router.metrics.embedding_calibration import EmbeddingCalibrationExample, fit_answer_type_calibrations


def main() -> None:
    args = parse_args()
    examples = []
    for row in read_jsonl(args.labels):
        if not isinstance(row.get("equivalent"), bool):
            raise ValueError("Each label row must contain boolean equivalent: true or false")
        examples.append(EmbeddingCalibrationExample(
            answer_type=str(row["answer_type"]),
            raw_similarity=float(row["raw_similarity"]),
            equivalent=row["equivalent"],
        ))
    calibrations = fit_answer_type_calibrations(examples, min_per_class=args.min_per_class)
    payload = {
        "schema_version": 1,
        "source_labels": str(args.labels),
        "min_per_class": args.min_per_class,
        "calibrations": {answer_type: {"low": value.low, "high": value.high} for answer_type, value in calibrations.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"calibration": str(output), "answer_types": sorted(calibrations), "examples": len(examples)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit answer-type embedding calibration from labelled equivalence pairs.")
    parser.add_argument("--labels", required=True, help="JSONL rows: answer_type, raw_similarity, equivalent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-per-class", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    main()
