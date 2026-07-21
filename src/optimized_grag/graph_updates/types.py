from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.optimized_grag.graph_construction import GraphConstructionConfig

GraphRebuildMode = Literal["full", "skip"]


@dataclass(slots=True)
class GraphUpdateConfig:
    construction: GraphConstructionConfig = field(default_factory=GraphConstructionConfig)
    rebuild_mode: GraphRebuildMode = "full"
    embed_chunks: bool = True


@dataclass(slots=True)
class GraphUpdateReport:
    documents: int = 0
    chunks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    vectors: int = 0
    capsules: int = 0
    capsule_edges: int = 0
    affected_node_ids: set[str] = field(default_factory=set)
    affected_edge_ids: set[str] = field(default_factory=set)
    rebuild_mode: str = "full"
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "vectors": self.vectors,
            "capsules": self.capsules,
            "capsule_edges": self.capsule_edges,
            "affected_node_ids": sorted(self.affected_node_ids),
            "affected_edge_ids": sorted(self.affected_edge_ids),
            "rebuild_mode": self.rebuild_mode,
            "details": dict(self.details),
        }
