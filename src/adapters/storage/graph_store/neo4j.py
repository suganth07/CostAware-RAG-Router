from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from logger import get_logger
from src.adapters.storage.graph_store.base import (
    Direction,
    GraphEdge,
    GraphNeighborhood,
    GraphNode,
    GraphStore,
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(slots=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: Optional[str] = None


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph store using the official Python driver."""

    def __init__(self, config: Neo4jConfig) -> None:
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise ImportError(
                "Neo4jGraphStore requires the 'neo4j' package. Install it with: pip install neo4j"
            ) from exc

        self.config = config
        self._driver = GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        self._logger = get_logger(__name__)

    def upsert_node(self, node: GraphNode) -> None:
        labels = ":".join(_safe_identifier(label) for label in node.labels)
        query = f"MERGE (n:{labels} {{id: $id}}) SET n += $properties"
        self._execute(query, id=node.id, properties={**node.properties, "id": node.id})

    def upsert_edge(self, edge: GraphEdge) -> None:
        relationship_type = _safe_identifier(edge.type)
        query = (
            "MERGE (source:Entity {id: $source_id}) "
            "MERGE (target:Entity {id: $target_id}) "
            f"MERGE (source)-[rel:{relationship_type} {{id: $id}}]->(target) "
            "SET rel += $properties"
        )
        self._execute(
            query,
            source_id=edge.source_id,
            target_id=edge.target_id,
            id=edge.id,
            properties={**edge.properties, "id": edge.id},
        )

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        records = self._execute("MATCH (n {id: $id}) RETURN n LIMIT 1", id=str(node_id))
        if not records:
            return None
        return _node_from_neo4j(records[0]["n"])

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

        pattern = _relationship_pattern(direction, depth)
        edge_type_filter = ""
        parameters: dict[str, Any] = {
            "ids": [str(node_id) for node_id in node_ids],
            "limit": int(limit),
        }
        if edge_types:
            edge_type_filter = "AND all(rel IN relationships(path) WHERE type(rel) IN $edge_types)"
            parameters["edge_types"] = [edge_type.upper() for edge_type in edge_types]

        query = (
            f"MATCH path=(seed){pattern}(neighbor) "
            "WHERE seed.id IN $ids "
            f"{edge_type_filter} "
            "RETURN nodes(path) AS nodes, relationships(path) AS relationships "
            "LIMIT $limit"
        )
        records = self._execute(query, **parameters)

        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        for record in records:
            for raw_node in record["nodes"]:
                node = _node_from_neo4j(raw_node)
                nodes[node.id] = node
            for raw_edge in record["relationships"]:
                edge = _edge_from_neo4j(raw_edge)
                edges[edge.id] = edge

        return GraphNeighborhood(
            seed_ids=tuple(parameters["ids"]),
            nodes=list(nodes.values()),
            edges=list(edges.values()),
        )

    def search_nodes(
        self,
        text: str,
        *,
        labels: Optional[set[str]] = None,
        limit: int = 20,
    ) -> list[GraphNode]:
        parameters: dict[str, Any] = {"text": text.lower(), "limit": int(limit)}
        label_filter = ""
        if labels:
            label_filter = "AND any(label IN labels(n) WHERE label IN $labels)"
            parameters["labels"] = list(labels)

        query = (
            "MATCH (n) "
            "WHERE any(key IN keys(n) WHERE toLower(toString(n[key])) CONTAINS $text) "
            f"{label_filter} "
            "RETURN n LIMIT $limit"
        )
        return [_node_from_neo4j(record["n"]) for record in self._execute(query, **parameters)]

    def run_query(
        self,
        query: str,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> list[Mapping[str, Any]]:
        return self._execute(query, **dict(parameters or {}))

    def close(self) -> None:
        self._driver.close()

    def _execute(self, query: str, **parameters: Any) -> list[Mapping[str, Any]]:
        records, _, _ = self._driver.execute_query(
            query,
            database_=self.config.database,
            **parameters,
        )
        return [record.data() for record in records]


def _safe_identifier(value: str) -> str:
    normalized = str(value).replace("-", "_").replace(" ", "_")
    if not _IDENTIFIER_RE.match(normalized):
        raise ValueError(f"Unsafe Cypher identifier: {value}")
    return normalized


def _relationship_pattern(direction: Direction, depth: int) -> str:
    if direction == "out":
        return f"-[*1..{depth}]->"
    if direction == "in":
        return f"<-[*1..{depth}]-"
    return f"-[*1..{depth}]-"


def _node_from_neo4j(raw_node: Any) -> GraphNode:
    properties = dict(raw_node)
    node_id = str(properties.get("id") or raw_node.element_id)
    return GraphNode(
        id=node_id,
        labels=tuple(raw_node.labels) or ("Entity",),
        properties=properties,
    )


def _edge_from_neo4j(raw_edge: Any) -> GraphEdge:
    properties = dict(raw_edge)
    edge_id = str(properties.get("id") or raw_edge.element_id)
    return GraphEdge(
        id=edge_id,
        source_id=str(raw_edge.start_node.get("id") or raw_edge.start_node.element_id),
        target_id=str(raw_edge.end_node.get("id") or raw_edge.end_node.element_id),
        type=raw_edge.type,
        properties=properties,
    )
