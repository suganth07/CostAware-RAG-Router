"""Join raw outcomes to questions and write inspectable deterministic metric vectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.datasets.base import EvaluationSample
from src.router.config import RewardConfig
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.metrics import score_outcome
from src.router.reward import compose_quality_details, compute_units
from src.router.types import CandidateOutcome, QuestionRecord


def main() -> None:
    args = parse_args()
    questions = {row["question_id"]: QuestionRecord(**row) for row in read_jsonl(args.questions)}
    rows = []
    for raw in read_jsonl(args.outcomes):
        outcome = CandidateOutcome.from_dict(raw)
        question = questions[outcome.question_id]
        sample = EvaluationSample(id=question.question_id, question=question.question, answers=question.accepted_answers, metadata=question.dataset_metadata)
        metrics = score_outcome(sample, outcome, gold_evidence=question.gold_evidence)
        composition = compose_quality_details(metrics, config=RewardConfig())
        rows.append({
            "outcome": outcome.to_dict(),
            "metrics": metrics.to_dict(),
            "quality": composition.quality,
            "quality_composition": {
                "base_quality": composition.base_quality,
                "penalty": composition.penalty,
                "availability_mask": composition.availability_mask,
            },
            "compute_units": compute_units(outcome),
        })
    write_jsonl_atomic(args.output, rows)
    print(json.dumps({"scored_outcomes": str(args.output), "records": len(rows)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score raw router outcomes with deterministic, nullable metrics.")
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="artifacts/router/datasets/pilot_scored.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    main()
