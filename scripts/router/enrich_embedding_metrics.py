"""Add raw and (only when configured) calibrated embedding metrics to scored JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.embedding.ollama import OllamaEmbeddingAdapter, OllamaEmbeddingConfig
from src.router.config import RewardConfig
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.metrics import AnswerTypeCalibration, calibrated_embedding_similarity, max_answer_embedding_similarity
from src.router.reward import compose_quality
from src.router.types import CandidateOutcome, MetricVector, QuestionRecord


def main() -> None:
    args = parse_args()
    questions = {row["question_id"]: QuestionRecord(**row) for row in read_jsonl(args.questions)}
    calibrations = load_calibrations(args.calibration) if args.calibration else {}
    cache_rows = read_jsonl(args.cache)
    cache = {str(row["key"]): float(row["raw_similarity"]) for row in cache_rows}
    embedder = OllamaEmbeddingAdapter(OllamaEmbeddingConfig(model=args.embedding_model, timeout_seconds=args.timeout_seconds))
    enriched: list[dict] = []
    for row in read_jsonl(args.scored_outcomes):
        outcome = CandidateOutcome.from_dict(row["outcome"])
        metrics = MetricVector(**row["metrics"])
        question = questions[outcome.question_id]
        cache_key = f"{args.embedding_model}|{outcome.question_id}|{outcome.candidate_id}|{outcome.repeat_id}|{outcome.run_id}"
        raw_similarity = cache.get(cache_key)
        if raw_similarity is None:
            raw_similarity = max_answer_embedding_similarity(outcome.generated_answer, question.accepted_answers, embedder=embedder)
            if raw_similarity is not None:
                cache[cache_key] = raw_similarity
                write_jsonl_atomic(args.cache, [{"key": key, "raw_similarity": value} for key, value in sorted(cache.items())])
        metrics.embedding_similarity_raw = raw_similarity
        metrics.embedding_similarity_calibrated = calibrated_embedding_similarity(
            raw_similarity,
            answer_type=question.answer_type,
            calibrations=calibrations,
            hard_conflict=metrics.entity_conflict or metrics.numeric_conflict or metrics.polarity_conflict,
        )
        row["metrics"] = metrics.to_dict()
        row["quality"] = compose_quality(metrics, config=RewardConfig())
        enriched.append(row)
    write_jsonl_atomic(args.output, enriched)
    print(json.dumps({
        "enriched_outcomes": str(args.output),
        "records": len(enriched),
        "embedding_model": args.embedding_model,
        "calibration_applied": bool(calibrations),
        "cache": str(args.cache),
    }, indent=2))


def load_calibrations(path: str | Path) -> dict[str, AnswerTypeCalibration]:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Calibration JSON must be an object keyed by answer type")
    result: dict[str, AnswerTypeCalibration] = {}
    for answer_type, bounds in raw.items():
        if not isinstance(bounds, dict) or set(bounds) != {"low", "high"}:
            raise ValueError(f"Calibration for {answer_type!r} must contain only low and high")
        result[str(answer_type)] = AnswerTypeCalibration(low=float(bounds["low"]), high=float(bounds["high"]))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add raw/correctness-calibrated answer embedding metrics to scored outcomes.")
    parser.add_argument("--scored-outcomes", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-model", default="embeddinggemma")
    parser.add_argument("--calibration", help="Validated JSON mapping of answer type to {low, high}. Omit to store raw scores only.")
    parser.add_argument("--cache", default="artifacts/router/cache/embedding_similarity.jsonl")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
