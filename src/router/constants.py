"""Stable router schema constants.  Action ordering is part of every artifact."""

from __future__ import annotations

ACTION_IDS: tuple[str, ...] = (
    "hybrid_rag",
    "graph_rag",
    "optimized_graph_rag",
)
ROUTER_SCHEMA_VERSION = 1
METRIC_SCHEMA_VERSION = 1

