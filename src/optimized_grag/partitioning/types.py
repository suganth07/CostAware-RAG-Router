from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.adapters.storage.graph_store import GraphEdge, GraphNode


@dataclass(slots=True)
class Region:
    id: str
    node_ids: set[str]
    level: int = 0
    parent_id: str | None = None


@dataclass(slots=True)
class CapsuleStats:
    node_count: int = 0
    edge_count: int = 0
    hub_count: int = 0
    density: float = 0.0
    connected_component_ratio: float = 1.0
    overlap_ratio: float = 0.0


@dataclass(slots=True)
class GraphCapsule:
    id: str
    anchor_id: str
    region_id: str
    node_ids: set[str]
    edge_ids: set[str]
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    level: int = 0
    boundary_node_ids: set[str] = field(default_factory=set)
    relation_types: set[str] = field(default_factory=set)
    source_document_ids: set[str] = field(default_factory=set)
    summary: str = ""
    stats: CapsuleStats = field(default_factory=CapsuleStats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.relation_types:
            self.relation_types = {edge.type for edge in self.edges}
        if not self.source_document_ids:
            source_ids = set()
            for edge in self.edges:
                source_id = edge.properties.get("source_document_id") or edge.properties.get("document_id")
                if source_id:
                    source_ids.add(str(source_id))
            for node in self.nodes:
                source_id = node.properties.get("source_document_id") or node.properties.get("document_id")
                if source_id:
                    source_ids.add(str(source_id))
            self.source_document_ids = source_ids
        if not self.summary:
            self.summary = capsule_summary(self)


@dataclass(slots=True)
class CapsuleCandidate:
    capsule: GraphCapsule
    score: float
    text_score: float = 0.0
    entity_score: float = 0.0
    structural_score: float = 0.0
    source_score: float = 0.0
    proximity_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def capsule_summary(capsule: GraphCapsule) -> str:
    names = [
        str(node.properties.get("name") or node.id)
        for node in capsule.nodes[:30]
        if "Chunk" not in node.labels
    ]
    relations = sorted(capsule.relation_types)[:12]
    routing_snippets = [
        " ".join(str(node.properties["text"]).split())[:400]
        for node in capsule.nodes
        if node.properties.get("text")
    ][:4]
    summary = (
        f"Capsule {capsule.id}. Anchor: {capsule.anchor_id}. "
        f"Entities: {', '.join(names)}. Relations: {', '.join(relations)}."
    )
    if routing_snippets:
        summary += " Representative source text: " + " | ".join(routing_snippets)
    return summary
