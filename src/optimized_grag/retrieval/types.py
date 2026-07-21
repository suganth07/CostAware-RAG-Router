from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.adapters.storage.graph_store import GraphEdge, GraphNode


@dataclass(slots=True)
class EvidencePath:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceSnippet:
    id: str
    text: str
    score: float
    capsule_id: str
    document_id: str | None = None
    chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BuiltContext:
    text: str
    source_snippets: list[SourceSnippet] = field(default_factory=list)
