"""Load model-neutral router configuration without embedding model choices in code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.router.critics.ensemble import CriticEnsembleConfig, CriticSlotConfig
from src.router.metrics.semantic_equivalence import SemanticScorerConfig


def load_semantic_config(path: str | Path) -> SemanticScorerConfig:
    value = _load_yaml_mapping(path)
    return SemanticScorerConfig(
        enabled=bool(value.get("enabled", False)),
        adapter=_optional_text(value.get("adapter")),
        model_id=_optional_text(value.get("model_id")),
        revision=_optional_text(value.get("revision")),
        options=dict(value.get("options") or {}),
    )


def load_critic_config(path: str | Path) -> CriticEnsembleConfig:
    value = _load_yaml_mapping(path)
    config = CriticEnsembleConfig(
        primary=_critic_slot(value.get("primary") or {}),
        secondary=_critic_slot(value.get("secondary") or {}),
        secondary_mode=str(value.get("secondary_mode") or "uncertain_or_audit"),
        uncertainty_threshold=float(value.get("uncertainty_threshold", 0.40)),
    )
    config.validate()
    return config


def _critic_slot(value: dict[str, Any]) -> CriticSlotConfig:
    return CriticSlotConfig(
        enabled=bool(value.get("enabled", False)),
        adapter=_optional_text(value.get("adapter")),
        model_id=_optional_text(value.get("model_id")),
        revision=_optional_text(value.get("revision")),
        options=dict(value.get("options") or {}),
    )


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load router YAML configuration files") from exc
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Router configuration at {path} must be a mapping")
    return value


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None
