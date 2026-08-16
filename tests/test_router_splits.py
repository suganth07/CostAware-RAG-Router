from __future__ import annotations

from src.router.data.splits import SplitFractions, build_grouped_split_manifest
from src.router.types import QuestionRecord


def _question(index: int, template: str, *, dataset: str = "metaqa") -> QuestionRecord:
    return QuestionRecord(
        question_id=f"q{index}",
        dataset=dataset,
        source_split="train",
        question=f"question {index}",
        accepted_answers=["answer"],
        group_keys={"template": template, "entity": f"entity-{index}"},
    )


def test_grouped_split_keeps_templates_in_one_split() -> None:
    questions = [
        _question(1, "a"),
        _question(2, "a"),
        _question(3, "b"),
        _question(4, "c"),
        _question(5, "d"),
    ]
    manifest = build_grouped_split_manifest(questions, fractions=SplitFractions(0.6, 0.2, 0.2), seed=7)
    by_group: dict[str, set[str]] = {}
    for row in manifest["assignments"]:
        by_group.setdefault(row["group_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in by_group.values())
    assert manifest["summary"]["questions"] == len(questions)


def test_grouped_split_holds_selected_dataset_out_for_cross_domain() -> None:
    questions = [
        _question(1, "a"),
        _question(2, "b"),
        _question(3, "c"),
        _question(4, "foreign-template", dataset="2wiki"),
    ]
    manifest = build_grouped_split_manifest(questions, cross_domain_datasets={"2wiki"})
    held_out = [row for row in manifest["assignments"] if row["dataset"] == "2wiki"]

    assert held_out[0]["split"] == "cross_domain"
    assert manifest["summary"]["cross_domain_available"] is True
