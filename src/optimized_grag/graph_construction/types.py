from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.adapters.storage.graph_store import GraphEdge, GraphNode
from src.helpers import Chunk
from src.optimized_grag.graph import LocalGraph


@dataclass(slots=True)
class GraphConstructionConfig:
    """Controls how raw documents or dataset bundles become a local graph."""

    chunk_size: int = 900
    chunk_overlap: int = 120
    use_native_graph: bool = True
    min_extracted_edges: int = 2
    fallback_to_document_chain: bool = True
    include_chunk_nodes: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "GraphConstructionConfig":
        values = values or {}
        graph_from_documents = bool(values.get("graph_from_documents", False))
        return cls(
            chunk_size=int(values.get("chunk_size", 900)),
            chunk_overlap=int(values.get("chunk_overlap", 120)),
            use_native_graph=not graph_from_documents,
            min_extracted_edges=int(values.get("min_extracted_edges", 2)),
            fallback_to_document_chain=bool(
                values.get("fallback_to_document_chain", True)
            ),
            include_chunk_nodes=bool(values.get("include_chunk_nodes", True)),
        )


@dataclass(slots=True)
class ConstructedChunkGraph:
    chunk: Chunk
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


@dataclass(slots=True)
class GraphConstructionReport:
    source: str
    documents: int = 0
    chunks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    fallback_used: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "documents": self.documents,
            "chunks": self.chunks,
            "nodes": self.graph_nodes,
            "edges": self.graph_edges,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "fallback_used": self.fallback_used,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class GraphConstructionResult:
    graph: LocalGraph
    chunks: list[Chunk] = field(default_factory=list)
    chunk_graphs: list[ConstructedChunkGraph] = field(default_factory=list)
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    report: GraphConstructionReport | None = None
