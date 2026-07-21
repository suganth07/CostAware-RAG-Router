"""Quality, cost, and reward calculations for logged route outcomes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Relative cost of work done by a completed retrieval route."""

    latency: float = 0.55
    graph_nodes: float = 0.15
    graph_edges: float = 0.15
    evidence_units: float = 0.10
    context_chars: float = 0.05

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Quality-first reward settings for the offline contextual bandit."""

    acceptable_quality: float = 0.65
    unacceptable_penalty: float = 2.0
    cost_penalty: float = 0.35

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def route_quality(metrics: dict[str, Any]) -> float:
    """Compute the documented quality proxy from an evaluated route result.

    When generation is enabled, exact match is the answer-correctness term.
    For retrieval-only runs it is intentionally replaced by Answer Recall@10;
    that makes the output useful for route collection without pretending it is
    final generated-answer accuracy.
    """
    answer = _metric(metrics, "exact_match", fallback="answer_recall_at_10")
    faithfulness = _metric(metrics, "supporting_evidence_recall", fallback="evidence_recall_at_10")
    evidence = _metric(metrics, "evidence_recall_at_10", fallback="supporting_evidence_recall")
    completeness = _metric(metrics, "token_f1", fallback="context_answer_recall")
    return _clamp(0.40 * answer + 0.25 * faithfulness + 0.20 * evidence + 0.15 * completeness)


def route_cost(
    outcome: dict[str, Any],
    *,
    maxima: dict[str, float],
    weights: CostWeights = CostWeights(),
) -> float:
    """Return a [0, 1] normalized measured route cost.

    Normalization occurs within one outcome collection run.  This is deliberate:
    milliseconds and graph-edge counts are incomparable raw units.
    """
    details = dict(outcome.get("details") or {})
    values = {
        "latency": float(outcome.get("latency_seconds") or 0.0) * 1000.0,
        "graph_nodes": float(details.get("graph_nodes") or 0.0),
        "graph_edges": float(details.get("graph_edges") or 0.0),
        "evidence_units": float(outcome.get("retrieved_units") or 0.0),
        "context_chars": float(outcome.get("context_chars") or 0.0),
    }
    total = 0.0
    for key, weight in asdict(weights).items():
        denominator = max(1.0, float(maxima.get(key) or 0.0))
        total += weight * _clamp(values[key] / denominator)
    return _clamp(total)


def route_reward(
    quality: float,
    cost: float,
    *,
    config: RewardConfig = RewardConfig(),
) -> float:
    """Strongly penalize cheap but unacceptable answers, as required by the proposal."""
    if quality < config.acceptable_quality:
        return -config.unacceptable_penalty * (config.acceptable_quality - quality) - config.cost_penalty * cost
    return 1.0 + quality - config.cost_penalty * cost


def cheapest_acceptable_route(
    outcomes: Iterable[dict[str, Any]],
    *,
    config: RewardConfig = RewardConfig(),
) -> dict[str, Any]:
    """Choose the cheapest outcome meeting the quality gate, then best fallback.

    This provides the transparent supervised/oracle baseline against which the
    learned contextual-bandit policy is compared.
    """
    items = list(outcomes)
    if not items:
        raise ValueError("At least one route outcome is required")
    acceptable = [item for item in items if float(item["quality_score"]) >= config.acceptable_quality]
    if acceptable:
        return min(acceptable, key=lambda item: (float(item["normalized_cost"]), -float(item["quality_score"]), str(item["route"])))
    return max(items, key=lambda item: (float(item["quality_score"]), -float(item["normalized_cost"]), str(item["route"])))


def run_cost_maxima(outcomes: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Collect the normalization denominators used for a saved outcome run."""
    maxima = {"latency": 0.0, "graph_nodes": 0.0, "graph_edges": 0.0, "evidence_units": 0.0, "context_chars": 0.0}
    for outcome in outcomes:
        details = dict(outcome.get("details") or {})
        maxima["latency"] = max(maxima["latency"], float(outcome.get("latency_seconds") or 0.0) * 1000.0)
        maxima["graph_nodes"] = max(maxima["graph_nodes"], float(details.get("graph_nodes") or 0.0))
        maxima["graph_edges"] = max(maxima["graph_edges"], float(details.get("graph_edges") or 0.0))
        maxima["evidence_units"] = max(maxima["evidence_units"], float(outcome.get("retrieved_units") or 0.0))
        maxima["context_chars"] = max(maxima["context_chars"], float(outcome.get("context_chars") or 0.0))
    return maxima


def _metric(metrics: dict[str, Any], name: str, *, fallback: str) -> float:
    value = metrics.get(name)
    if value is None:
        value = metrics.get(fallback)
    return _clamp(float(value or 0.0))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
