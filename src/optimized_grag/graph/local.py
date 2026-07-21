from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from src.adapters.storage.graph_store import Direction, GraphEdge, GraphNode


class LocalGraph:
    """Small local graph representation used for capsule construction and search."""

    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.incoming: dict[str, set[str]] = defaultdict(set)
        self._neighbors_cache: dict[
            tuple[str, Direction], tuple[tuple[str, GraphEdge], ...]
        ] = {}

    @classmethod
    def from_nodes_edges(
        cls,
        nodes: Iterable[GraphNode],
        edges: Iterable[GraphEdge],
    ) -> "LocalGraph":
        graph = cls()
        graph.add_nodes(nodes)
        graph.add_edges(edges)
        return graph

    def add_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return
        self.nodes[node.id] = GraphNode(
            id=node.id,
            labels=tuple(dict.fromkeys((*existing.labels, *node.labels))),
            properties={**existing.properties, **node.properties},
        )

    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        for node in nodes:
            self.add_node(node)

    def add_edge(self, edge: GraphEdge) -> None:
        if edge.source_id not in self.nodes:
            self.add_node(GraphNode(id=edge.source_id))
        if edge.target_id not in self.nodes:
            self.add_node(GraphNode(id=edge.target_id))

        previous = self.edges.get(edge.id)
        if previous is not None:
            self.outgoing[previous.source_id].discard(edge.id)
            self.incoming[previous.target_id].discard(edge.id)
            self._invalidate_neighbors(previous.source_id, previous.target_id)

        self.edges[edge.id] = edge
        self.outgoing[edge.source_id].add(edge.id)
        self.incoming[edge.target_id].add(edge.id)
        self._invalidate_neighbors(edge.source_id, edge.target_id)

    def add_edges(self, edges: Iterable[GraphEdge]) -> None:
        for edge in edges:
            self.add_edge(edge)

    def edge_ids(self, node_id: str, direction: Direction = "both") -> set[str]:
        if direction == "out":
            return set(self.outgoing.get(node_id, set()))
        if direction == "in":
            return set(self.incoming.get(node_id, set()))
        return set(self.outgoing.get(node_id, set())) | set(self.incoming.get(node_id, set()))

    def neighbors(self, node_id: str, *, direction: Direction = "both") -> list[tuple[str, GraphEdge]]:
        cache_key = (node_id, direction)
        cached = self._neighbors_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        neighbors: list[tuple[str, GraphEdge]] = []
        for edge_id in sorted(self.edge_ids(node_id, direction)):
            edge = self.edges[edge_id]
            next_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbors.append((next_id, edge))
        self._neighbors_cache[cache_key] = tuple(neighbors)
        return neighbors

    def degree(self, node_id: str) -> int:
        return len(self.edge_ids(node_id))

    def subgraph(self, node_ids: set[str]) -> "LocalGraph":
        graph = LocalGraph()
        graph.add_nodes(self.nodes[node_id] for node_id in node_ids if node_id in self.nodes)
        graph.add_edges(
            edge
            for edge in self.edges.values()
            if edge.source_id in node_ids and edge.target_id in node_ids
        )
        return graph

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.outgoing.clear()
        self.incoming.clear()
        self._neighbors_cache.clear()

    def _invalidate_neighbors(self, *node_ids: str) -> None:
        for node_id in node_ids:
            for direction in ("in", "out", "both"):
                self._neighbors_cache.pop((node_id, direction), None)
