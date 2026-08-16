"""Create a compact provenance and coverage card for a router outcome dataset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.router.constants import ACTION_IDS
from src.router.data.serialization import read_jsonl, write_text_atomic


def main() -> None:
    args = parse_args()
    questions = read_jsonl(args.questions)
    outcomes = read_jsonl(args.outcomes)
    scored = read_jsonl(args.scored_outcomes) if args.scored_outcomes else []
    assignments = read_jsonl(args.split_assignments) if args.split_assignments else []
    text = _render_card(
        title=args.title,
        questions=questions,
        outcomes=outcomes,
        scored=scored,
        assignments=assignments,
        scope=args.scope,
    )
    write_text_atomic(args.output, text)
    print(json.dumps({"data_card": str(args.output), "questions": len(questions), "outcomes": len(outcomes)}, indent=2))


def _render_card(*, title: str, questions: list[dict], outcomes: list[dict], scored: list[dict], assignments: list[dict], scope: str) -> str:
    question_datasets = Counter(str(row.get("dataset") or "unknown") for row in questions)
    answer_types = Counter(str(row.get("answer_type") or "unknown") for row in questions)
    source_splits = Counter(str(row.get("source_split") or "unknown") for row in questions)
    actions = Counter(str(row.get("candidate_id") or "unknown") for row in outcomes)
    failures = Counter(str(row.get("candidate_id") or "unknown") for row in outcomes if row.get("error_type") or row.get("timeout"))
    latency = defaultdict(list)
    for row in outcomes:
        value = row.get("end_to_end_latency_ms")
        if isinstance(value, (int, float)):
            latency[str(row.get("candidate_id"))].append(float(value))
    metric_availability = _metric_availability(scored)
    split_counts = Counter(str(row.get("split") or "unknown") for row in assignments)
    lines = [
        f"# {title}",
        "",
        "## Scope",
        "",
        scope,
        "",
        "This card records observed dataset provenance and coverage. It does not claim reward calibration, policy training, or production readiness.",
        "",
        "## Questions",
        "",
        f"- Total frozen questions: `{len(questions)}`",
        f"- Dataset counts: {_inline_counts(question_datasets)}",
        f"- Source split counts: {_inline_counts(source_splits)}",
        f"- Answer-type counts: {_inline_counts(answer_types)}",
        "",
        "## Candidate outcomes",
        "",
        f"- Total raw outcomes: `{len(outcomes)}`",
        f"- Candidate counts: {_inline_counts(actions)}",
        f"- Candidate failures/timeouts: {_inline_counts(failures, include_actions=True)}",
        "",
        "| Candidate | Measured rows | Mean latency (ms) | p95 latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for action in ACTION_IDS:
        values = sorted(latency[action])
        lines.append(f"| `{action}` | {len(values)} | {_mean(values)} | {_p95(values)} |")
    lines.extend([
        "",
        "## Metric availability",
        "",
    ])
    if metric_availability:
        lines.extend(["| Metric | Present rows | Availability |", "|---|---:|---:|"])
        total = len(scored)
        for metric, present in sorted(metric_availability.items()):
            lines.append(f"| `{metric}` | {present} | {present / total:.1%} |")
    else:
        lines.append("No scored-outcome artifact was supplied.")
    lines.extend(["", "## Grouped split manifest", ""])
    if assignments:
        lines.append(f"- Assignment rows: `{len(assignments)}`")
        lines.append(f"- Split counts: {_inline_counts(split_counts)}")
        lines.append("- Group leakage is checked by `build_split_manifest.py` before this card is generated.")
    else:
        lines.append("No grouped split manifest was supplied. This is expected until M5 split construction.")
    lines.extend([
        "",
        "## Limitations and next gate",
        "",
        "- Optional semantic and critic metrics may be null until their configured models and human-calibration requirements are satisfied.",
        "- A pilot card is not a substitute for the 15,000–30,000-question M5 full-information dataset.",
        "- Do not use this card to authorize M6/M7 training before reward calibration and the M5 data gates are complete.",
        "",
    ])
    return "\n".join(lines)


def _metric_availability(scored: list[dict]) -> Counter:
    result: Counter = Counter()
    for row in scored:
        for metric, value in (row.get("metrics") or {}).items():
            if value is not None:
                result[str(metric)] += 1
    return result


def _inline_counts(values: Counter, *, include_actions: bool = False) -> str:
    if include_actions:
        return ", ".join(f"`{action}`: {values.get(action, 0)}" for action in ACTION_IDS)
    return ", ".join(f"`{key}`: {value}" for key, value in sorted(values.items())) or "none"


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.2f}" if values else "null"


def _p95(values: list[float]) -> str:
    if not values:
        return "null"
    return f"{values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)]:.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown data card for router outcomes.")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--scored-outcomes")
    parser.add_argument("--split-assignments")
    parser.add_argument("--title", default="Router data card")
    parser.add_argument("--scope", default="Offline router dataset artifact.")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
