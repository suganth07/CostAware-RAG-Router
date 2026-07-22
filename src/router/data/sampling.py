"""Dataset-to-router conversion and deterministic, sampling-only stratification."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.adapters.datasets.base import EvaluationSample
from src.evaluation.core import answer_kind, gold_evidence_items
from src.router.types import QuestionRecord


def question_record_from_sample(sample: EvaluationSample) -> QuestionRecord:
    metadata = dict(sample.metadata)
    dataset = str(metadata.get("dataset") or "unknown")
    source_split = str(metadata.get("split") or _split_from_id(sample.id) or "unknown")
    return QuestionRecord(
        question_id=sample.id,
        dataset=dataset,
        source_split=source_split,
        question=sample.question,
        accepted_answers=list(sample.answers),
        gold_evidence=[{"identifiers": sorted(item.identifiers), "texts": item.texts} for item in gold_evidence_items(sample)],
        answer_type=answer_kind(sample.answers),
        sampling_strata={
            "hop": metadata.get("hop"),
            "question_type": metadata.get("question_type") or metadata.get("type"),
            "answer_type": answer_kind(sample.answers),
        },
        group_keys={"template": _template_key(sample.question), "entity": _first_bracketed_entity(sample.question)},
        dataset_metadata=metadata,
    )


def stratified_sample(samples: Iterable[EvaluationSample], *, limit: int, seed: int = 7) -> list[EvaluationSample]:
    """Round-robin sample over visible training strata; never used as runtime input."""
    if limit <= 0:
        return []
    buckets: dict[tuple[str, str, str], list[EvaluationSample]] = defaultdict(list)
    for sample in samples:
        metadata = sample.metadata
        key = (
            str(metadata.get("dataset") or "unknown"),
            str(metadata.get("hop") or "none"),
            answer_kind(sample.answers),
        )
        buckets[key].append(sample)
    # Stable pseudo-random order without changing global random state.
    import random
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[EvaluationSample] = []
    keys = sorted(buckets)
    while keys and len(selected) < limit:
        next_keys = []
        for key in keys:
            if len(selected) >= limit:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _split_from_id(value: str) -> str | None:
    parts = str(value).split(":")
    return parts[-2] if len(parts) >= 2 and parts[-2] in {"train", "test", "validation", "dev"} else None


def _template_key(question: str) -> str:
    import re
    normalized = re.sub(r"\[[^\]]+\]", "[ENTITY]", question.lower())
    normalized = re.sub(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*\b", "[ENTITY]", normalized)
    return " ".join(normalized.split())


def _first_bracketed_entity(question: str) -> str:
    import re
    match = re.search(r"\[([^\]]+)\]", question)
    return match.group(1).strip().lower() if match else ""
