from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping, Optional, Sequence

from logger import get_logger
from src.adapters.storage.graph_store.base import (
    Direction,
    GraphEdge,
    GraphNeighborhood,
    GraphNode,
    GraphStore,
)


class InMemoryGraphStore(GraphStore):
    """Process-local graph store for tests and small experiments."""

    def __init__(self) -> None:
        self._nodes: dict[str, GraphNode] = {}
        self._edges: dict[str, GraphEdge] = {}
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._logger = get_logger(__name__)

    def upsert_node(self, node: GraphNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None:
            self._nodes[node.id] = node
            return

        labels = tuple(dict.fromkeys((*existing.labels, *node.labels)))
        properties = {**existing.properties, **node.properties}
        self._nodes[node.id] = GraphNode(id=node.id, labels=labels, properties=properties)

    def upsert_edge(self, edge: GraphEdge) -> None:
        if edge.source_id not in self._nodes:
            self.upsert_node(GraphNode(id=edge.source_id))
        if edge.target_id not in self._nodes:
            self.upsert_node(GraphNode(id=edge.target_id))

        previous = self._edges.get(edge.id)
        if previous is not None:
            self._outgoing[previous.source_id].discard(previous.id)
            self._incoming[previous.target_id].discard(previous.id)

        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].add(edge.id)
        self._incoming[edge.target_id].add(edge.id)

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(str(node_id))

    def get_neighbors(
        self,
        node_ids: Sequence[str],
        *,
        depth: int = 1,
        edge_types: Optional[set[str]] = None,
        direction: Direction = "both",
        limit: int = 100,
    ) -> GraphNeighborhood:
        if depth < 1 or not node_ids:
            return GraphNeighborhood(seed_ids=tuple(str(node_id) for node_id in node_ids))

        normalized_types = {edge_type.upper() for edge_type in edge_types} if edge_types else None
        seed_ids = tuple(str(node_id) for node_id in node_ids)
        visited_nodes = set(seed_ids)
        ordered_node_ids = list(dict.fromkeys(seed_ids))
        collected_edges: dict[str, GraphEdge] = {}
        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in seed_ids)

        while queue and len(visited_nodes) < limit:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            for edge_id in self._edge_ids_for_direction(current_id, direction):
                edge = self._edges[edge_id]
                if normalized_types and edge.type not in normalized_types:
                    continue

                collected_edges[edge.id] = edge
                next_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if next_id not in visited_nodes:
                    visited_nodes.add(next_id)
                    ordered_node_ids.append(next_id)
                    queue.append((next_id, current_depth + 1))
                    if len(visited_nodes) >= limit:
                        break

        # Preserve deterministic BFS discovery order. Context construction then
        # renders seeds/direct neighbors before deeper nodes, instead of relying
        # on hash-randomized set iteration that could truncate answer nodes.
        nodes = [
            self._nodes[node_id]
            for node_id in ordered_node_ids
            if node_id in self._nodes
        ]
        return GraphNeighborhood(
            seed_ids=seed_ids,
            nodes=nodes,
            edges=list(collected_edges.values()),
        )

    def search_nodes(
        self,
        text: str,
        *,
        labels: Optional[set[str]] = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        needle = text.lower().strip()
        normalized_labels = {label.lower() for label in labels} if labels else None
        matches: list[GraphNode] = []

        for node in self._nodes.values():
            if normalized_labels and not ({label.lower() for label in node.labels} & normalized_labels):
                continue

            haystack = " ".join(
                [
                    node.id,
                    *node.labels,
                    *[str(value) for value in node.properties.values()],
                ]
            ).lower()
            if not needle or needle in haystack:
                matches.append(node)
                if len(matches) >= limit:
                    break

        return matches

    def run_query(
        self,
        query: str,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> list[Mapping[str, Any]]:
        raise NotImplementedError(
            "InMemoryGraphStore does not support a graph query language. "
            "Use typed methods or Neo4jGraphStore.run_query for Cypher."
        )

    def clear(self) -> None:
        self._nodes.clear()
        self._edges.clear()
        self._outgoing.clear()
        self._incoming.clear()

    def _edge_ids_for_direction(self, node_id: str, direction: Direction) -> list[str]:
        if direction == "out":
            return sorted(self._outgoing.get(node_id, set()))
        if direction == "in":
            return sorted(self._incoming.get(node_id, set()))
        return sorted(
            set(self._outgoing.get(node_id, set()))
            | set(self._incoming.get(node_id, set()))
        )
