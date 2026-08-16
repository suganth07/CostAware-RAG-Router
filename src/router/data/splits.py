"""Leakage-aware grouped split construction for offline router datasets."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from src.router.data.manifests import fingerprint_payload
from src.router.types import QuestionRecord


SPLIT_IDS: tuple[str, ...] = ("train", "validation", "locked_test")


@dataclass(frozen=True, slots=True)
class SplitFractions:
    train: float = 0.80
    validation: float = 0.10
    locked_test: float = 0.10

    def validate(self) -> None:
        values = (self.train, self.validation, self.locked_test)
        if any(value <= 0.0 or value >= 1.0 for value in values):
            raise ValueError("All grouped split fractions must be between zero and one")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Grouped split fractions must sum to one")

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "validation": self.validation, "locked_test": self.locked_test}


def build_grouped_split_manifest(
    questions: Iterable[QuestionRecord],
    *,
    group_by: str = "template",
    cross_domain_datasets: set[str] | None = None,
    fractions: SplitFractions = SplitFractions(),
    seed: int = 7,
) -> dict:
    """Assign a whole leakage group to exactly one offline split.

    Questions designated as cross-domain are held out before grouped allocation.
    They are intentionally absent from train/validation/locked-test targets.
    """
    fractions.validate()
    values = list(questions)
    if not values:
        raise ValueError("Cannot construct a split manifest from zero questions")
    if len({item.question_id for item in values}) != len(values):
        raise ValueError("Question IDs must be unique before grouped splitting")
    if group_by not in {"template", "entity", "template_entity"}:
        raise ValueError("group_by must be template, entity, or template_entity")

    cross_domain_datasets = {str(value) for value in (cross_domain_datasets or set())}
    held_out = [item for item in values if item.dataset in cross_domain_datasets]
    allocatable = [item for item in values if item.dataset not in cross_domain_datasets]
    if not allocatable:
        raise ValueError("All questions were designated cross-domain; no train/validation/locked-test questions remain")

    grouped: dict[str, list[QuestionRecord]] = defaultdict(list)
    for question in allocatable:
        grouped[_group_id(question, group_by=group_by)].append(question)
    if len(grouped) < len(SPLIT_IDS):
        raise ValueError(
            f"Grouped split needs at least {len(SPLIT_IDS)} distinct {group_by} groups; found {len(grouped)}"
        )

    assignments = _allocate_groups(grouped, fractions=fractions, seed=seed)
    rows = []
    for question in values:
        group_id = _group_id(question, group_by=group_by)
        split = "cross_domain" if question.dataset in cross_domain_datasets else assignments[group_id]
        rows.append(
            {
                "question_id": question.question_id,
                "dataset": question.dataset,
                "source_split": question.source_split,
                "group_id": group_id,
                "split": split,
            }
        )
    rows.sort(key=lambda row: row["question_id"])
    _validate_no_group_leakage(rows)
    summary = _summary(rows, group_by=group_by, fractions=fractions, cross_domain_datasets=cross_domain_datasets, seed=seed)
    return {
        "schema_version": 1,
        "kind": "router_grouped_split_manifest",
        "fingerprint": fingerprint_payload({"rows": rows, "summary": summary}),
        "assignments": rows,
        "summary": summary,
    }


def _allocate_groups(
    groups: dict[str, list[QuestionRecord]], *, fractions: SplitFractions, seed: int
) -> dict[str, str]:
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    # Larger groups first lets the greedy deficit rule approximate desired row
    # fractions while preserving strict whole-group allocation.
    ordered.sort(key=lambda group_id: (-len(groups[group_id]), group_id))
    total = sum(len(values) for values in groups.values())
    targets = {split: total * fraction for split, fraction in fractions.as_dict().items()}
    allocated = Counter({split: 0 for split in SPLIT_IDS})
    result: dict[str, str] = {}
    for index, group_id in enumerate(ordered):
        remaining_groups = len(ordered) - index
        empty = [split for split in SPLIT_IDS if allocated[split] == 0]
        candidates = empty if len(empty) >= remaining_groups else list(SPLIT_IDS)
        selected = max(
            candidates,
            key=lambda split: (targets[split] - allocated[split], -allocated[split], split),
        )
        result[group_id] = selected
        allocated[selected] += len(groups[group_id])
    return result


def _group_id(question: QuestionRecord, *, group_by: str) -> str:
    template = str(question.group_keys.get("template") or question.question_id)
    entity = str(question.group_keys.get("entity") or question.question_id)
    if group_by == "template":
        value = template
    elif group_by == "entity":
        value = entity
    else:
        value = f"{template}||{entity}"
    return f"{question.dataset}:{value}"


def _validate_no_group_leakage(rows: list[dict]) -> None:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[str(row["group_id"])].add(str(row["split"]))
    leaked = [group_id for group_id, splits in group_splits.items() if len(splits) != 1]
    if leaked:
        raise ValueError(f"Grouped split leakage found for {len(leaked)} groups; first={leaked[0]}")


def _summary(
    rows: list[dict],
    *,
    group_by: str,
    fractions: SplitFractions,
    cross_domain_datasets: set[str],
    seed: int,
) -> dict:
    counts = Counter(str(row["split"]) for row in rows)
    groups = defaultdict(set)
    datasets = defaultdict(Counter)
    for row in rows:
        split = str(row["split"])
        groups[split].add(str(row["group_id"]))
        datasets[split][str(row["dataset"])] += 1
    return {
        "questions": len(rows),
        "split_counts": dict(counts),
        "group_counts": {split: len(values) for split, values in groups.items()},
        "dataset_counts_by_split": {split: dict(values) for split, values in datasets.items()},
        "group_by": group_by,
        "fractions": fractions.as_dict(),
        "cross_domain_datasets": sorted(cross_domain_datasets),
        "seed": seed,
        "cross_domain_available": bool(cross_domain_datasets),
    }
