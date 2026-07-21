from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

Direction = Literal["out", "in", "both"]


@dataclass(slots=True)
class GraphNode:
    id: str
    labels: tuple[str, ...] = ("Entity",)
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.labels = tuple(self.labels or ("Entity",))
        self.properties = dict(self.properties)


@dataclass(slots=True)
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.source_id = str(self.source_id)
        self.target_id = str(self.target_id)
        self.type = str(self.type or "RELATED_TO").upper()
        self.properties = dict(self.properties)


@dataclass(slots=True)
class GraphNeighborhood:
    seed_ids: tuple[str, ...]
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


class GraphStore(ABC):
    """Minimal graph persistence API used by GraphRAG."""

    @abstractmethod
    def upsert_node(self, node: GraphNode) -> None:
        raise NotImplementedError

    def upsert_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self.upsert_node(node)

    @abstractmethod
    def upsert_edge(self, edge: GraphEdge) -> None:
        raise NotImplementedError

    def upsert_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            self.upsert_edge(edge)

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(
        self,
        node_ids: Sequence[str],
        *,
        depth: int = 1,
        edge_types: Optional[set[str]] = None,
        direction: Direction = "both",
        limit: int = 100,
    ) -> GraphNeighborhood:
        raise NotImplementedError

    @abstractmethod
    def search_nodes(
        self,
        text: str,
        *,
        labels: Optional[set[str]] = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        raise NotImplementedError

    @abstractmethod
    def run_query(
        self,
        query: str,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        return None
