from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.optimized_grag.config import OptimizedGraphRAGConfig


@dataclass(slots=True)
class OptimizedIndexingReport:
    documents: int = 0
    chunks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    vectors: int = 0
    capsules: int = 0
    capsule_edges: int = 0
    details: dict[str, Any] = field(default_factory=dict)


__all__ = ["OptimizedGraphRAGConfig", "OptimizedIndexingReport"]
