"""Explicit, serializable experiment choices for the router."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from src.router.constants import ACTION_IDS, ROUTER_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """Request-level limits.  `None` means the constraint is disabled."""

    mean_latency_ms: float | None = None
    tail_latency_ms: float | None = None
    compute_units: float | None = None
    max_request_latency_ms: float | None = None

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Initial deterministic reward composition; calibrate before promotion."""

    answer_weight: float = 0.50
    faithfulness_weight: float = 0.20
    evidence_weight: float = 0.20
    completeness_weight: float = 0.10
    contradiction_penalty: float = 0.50
    unsupported_penalty: float = 0.40
    execution_failure_penalty: float = 1.00
    incorrect_refusal_penalty: float = 0.50

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("Reward weights and penalties must be non-negative")


@dataclass(frozen=True, slots=True)
class RouterConfig:
    schema_version: int = ROUTER_SCHEMA_VERSION
    actions: tuple[str, ...] = ACTION_IDS
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    fallback_action: str | None = None
    random_seed: int = 7

    def validate(self) -> None:
        if self.schema_version != ROUTER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported router schema version {self.schema_version}")
        if self.actions != ACTION_IDS:
            raise ValueError("Router action IDs and ordering are immutable for schema version 1")
        if self.fallback_action is not None and self.fallback_action not in self.actions:
            raise ValueError("fallback_action must be one of the registered actions")
        self.budgets.validate()
        self.reward.validate()

    def to_dict(self) -> dict:
        return asdict(self)
