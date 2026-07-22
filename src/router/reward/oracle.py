"""Budget feasibility and the full-information oracle used for warm-start labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.router.config import BudgetConfig
from src.router.constants import ACTION_IDS
from src.router.types import ScoredOutcome


def latency_statistics(outcomes: Iterable[ScoredOutcome]) -> dict[str, dict[str, float]]:
    """Return per-action observed mean and p95 latency over a collection slice."""
    values: dict[str, list[float]] = defaultdict(list)
    for item in outcomes:
        if item.outcome.end_to_end_latency_ms is not None:
            values[item.outcome.candidate_id].append(item.outcome.end_to_end_latency_ms)
    return {
        action: {"mean": sum(action_values) / len(action_values), "p95": _percentile(action_values, 0.95) or 0.0}
        for action, action_values in values.items()
        if action_values
    }


def apply_feasibility(
    outcomes: Iterable[ScoredOutcome],
    *,
    budgets: BudgetConfig,
    latency_reference: dict[str, dict[str, float]] | None = None,
) -> list[ScoredOutcome]:
    budgets.validate()
    items = list(outcomes)
    observed = latency_reference or latency_statistics(items)
    for item in items:
        violations: dict[str, float] = {}
        latency = item.outcome.end_to_end_latency_ms
        if budgets.max_request_latency_ms and latency is not None and latency > budgets.max_request_latency_ms:
            violations["request_latency"] = latency - budgets.max_request_latency_ms
        action_latency = observed.get(item.outcome.candidate_id, {})
        if budgets.mean_latency_ms and action_latency.get("mean", 0.0) > budgets.mean_latency_ms:
            violations["mean_latency"] = action_latency["mean"] - budgets.mean_latency_ms
        if budgets.tail_latency_ms and action_latency.get("p95", 0.0) > budgets.tail_latency_ms:
            violations["tail_latency"] = action_latency["p95"] - budgets.tail_latency_ms
        if budgets.compute_units and item.compute_units is not None and item.compute_units > budgets.compute_units:
            violations["compute"] = item.compute_units - budgets.compute_units
        item.feasible = not violations
        item.infeasibility = violations
    return items


def select_budget_aware_oracle(
    outcomes: Iterable[ScoredOutcome],
    *,
    budgets: BudgetConfig,
    latency_reference: dict[str, dict[str, float]] | None = None,
) -> ScoredOutcome:
    items = apply_feasibility(outcomes, budgets=budgets, latency_reference=latency_reference)
    if {item.outcome.candidate_id for item in items} != set(ACTION_IDS):
        raise ValueError("Oracle selection requires one scored outcome for every action")
    feasible = [item for item in items if item.feasible and item.quality is not None]
    if feasible:
        return max(feasible, key=lambda item: (float(item.quality), -float(item.compute_units or 0.0), item.outcome.candidate_id))
    # Explicit least-violation fallback for an infeasible budget, never hidden.
    return min(items, key=lambda item: (sum(item.infeasibility.values()), -float(item.quality or -1.0), item.outcome.candidate_id))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]
