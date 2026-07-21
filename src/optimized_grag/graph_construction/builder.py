from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from logger import get_logger
from src.adapters.storage.graph_store import GraphEdge, GraphNode
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.g_rag.extractors import EntityRelationExtractor, SimpleEntityRelationExtractor
from src.helpers import Document, TextChunker, normalize_documents
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.graph_construction.fallbacks import build_document_chain_graph, slug_id
from src.optimized_grag.graph_construction.types import (
    ConstructedChunkGraph,
    GraphConstructionConfig,
    GraphConstructionReport,
    GraphConstructionResult,
)


class DocumentGraphConstructor:
    """Reusable document/dataset graph construction for training and indexing."""

    def __init__(
        self,
        *,
        config: GraphConstructionConfig | None = None,
        extractor: EntityRelationExtractor | None = None,
        instrumentation: ComputeTracker | None = None,
    ) -> None:
        self.config = config or GraphConstructionConfig()
        self.extractor = extractor or SimpleEntityRelationExtractor()
        self.instrumentation = instrumentation or NoopComputeTracker()
        self._logger = get_logger(__name__)

    def from_documents(
        self,
        documents: Iterable[Document | dict[str, Any] | str],
        *,
        dataset_name: str = "documents",
    ) -> GraphConstructionResult:
        normalized_documents = normalize_documents(documents)
        chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

        with self.instrumentation.track(
            "graph_construction.chunk_documents",
            {"documents": len(normalized_documents)},
        ):
            chunks = chunker.chunk_documents(normalized_documents)

        graph = LocalGraph()
        chunk_graphs: list[ConstructedChunkGraph] = []
        previous_chunk_by_document: dict[str, str] = {}
        with self.instrumentation.track("graph_construction.extract_graph", {"chunks": len(chunks)}):
            for chunk in chunks:
                extraction = self.extractor.extract(chunk)
                nodes = list(extraction.nodes)
                edges = list(extraction.edges)
                if self.config.include_chunk_nodes:
                    chunk_node = _chunk_node(chunk, dataset_name)
                    chunk_edges = _chunk_edges(
                        chunk=chunk,
                        chunk_node_id=chunk_node.id,
                        entity_nodes=extraction.nodes,
                        previous_chunk_id=previous_chunk_by_document.get(chunk.document_id),
                    )
                    previous_chunk_by_document[chunk.document_id] = chunk_node.id
                    nodes = [chunk_node, *nodes]
                    edges = [*edges, *chunk_edges]

                chunk_graphs.append(
                    ConstructedChunkGraph(
                        chunk=chunk,
                        nodes=nodes,
                        edges=edges,
                    )
                )
                graph.add_nodes(nodes)
                graph.add_edges(edges)

        source = "document_entity_extraction"
        fallback_used = False
        details: dict[str, Any] = {
            "extracted_nodes": len(graph.nodes),
            "extracted_edges": len(graph.edges),
            "min_extracted_edges": self.config.min_extracted_edges,
        }
        if (
            self.config.fallback_to_document_chain
            and len(graph.edges) < self.config.min_extracted_edges
        ):
            graph = build_document_chain_graph(normalized_documents, dataset_name=dataset_name)
            source = "document_fallback_chain"
            fallback_used = True

        nodes = list(graph.nodes.values())
        edges = list(graph.edges.values())
        report = GraphConstructionReport(
            source=source,
            documents=len(normalized_documents),
            chunks=len(chunks),
            graph_nodes=len(nodes),
            graph_edges=len(edges),
            fallback_used=fallback_used,
            details=details,
        )
        self._logger.info(
            "constructed graph source=%s documents=%s chunks=%s nodes=%s edges=%s",
            source,
            report.documents,
            report.chunks,
            report.graph_nodes,
            report.graph_edges,
        )
        return GraphConstructionResult(
            graph=graph,
            chunks=chunks,
            chunk_graphs=chunk_graphs,
            nodes=nodes,
            edges=edges,
            report=report,
        )

    def from_dataset_bundle(
        self,
        bundle: Any,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> GraphConstructionResult:
        config = self.config
        if options is not None:
            config = GraphConstructionConfig.from_mapping(options)

        has_native_graph = bool(getattr(bundle, "graph_nodes", None)) and bool(
            getattr(bundle, "graph_edges", None)
        )
        if config.use_native_graph and has_native_graph:
            nodes = list(getattr(bundle, "graph_nodes", []) or [])
            edges = list(getattr(bundle, "graph_edges", []) or [])
            graph = LocalGraph.from_nodes_edges(nodes, edges)
            report = GraphConstructionReport(
                source="native_dataset_graph",
                documents=len(getattr(bundle, "documents", []) or []),
                chunks=0,
                graph_nodes=len(graph.nodes),
                graph_edges=len(graph.edges),
                fallback_used=False,
                details={"dataset": getattr(bundle, "name", "dataset")},
            )
            return GraphConstructionResult(
                graph=graph,
                nodes=list(graph.nodes.values()),
                edges=list(graph.edges.values()),
                report=report,
            )

        constructor = self
        if config != self.config:
            constructor = DocumentGraphConstructor(
                config=config,
                extractor=self.extractor,
                instrumentation=self.instrumentation,
            )
        return constructor.from_documents(
            getattr(bundle, "documents", []) or [],
            dataset_name=str(getattr(bundle, "name", "dataset")),
        )


def construct_training_graph(
    bundle: Any,
    options: Mapping[str, Any] | None = None,
    *,
    extractor: EntityRelationExtractor | None = None,
    instrumentation: ComputeTracker | None = None,
) -> GraphConstructionResult:
    config = GraphConstructionConfig.from_mapping(options)
    return DocumentGraphConstructor(
        config=config,
        extractor=extractor,
        instrumentation=instrumentation,
    ).from_dataset_bundle(bundle, options=options)


def build_training_graph(
    bundle: Any,
    options: Mapping[str, Any] | None = None,
    *,
    extractor: EntityRelationExtractor | None = None,
    instrumentation: ComputeTracker | None = None,
) -> tuple[LocalGraph, dict[str, Any]]:
    result = construct_training_graph(
        bundle,
        options,
        extractor=extractor,
        instrumentation=instrumentation,
    )
    report = result.report.to_dict() if result.report else {}
    return result.graph, report


def _chunk_node(chunk: Any, dataset_name: str) -> GraphNode:
    preview = " ".join(chunk.text.split())[:700]
    return GraphNode(
        id=f"chunk:{slug_id(chunk.id, 'chunk')}",
        labels=("Chunk",),
        properties={
            "dataset": dataset_name,
            "name": f"Chunk {chunk.document_id}:{chunk.index}",
            "source_document_id": chunk.document_id,
            "source_chunk_id": chunk.id,
            "chunk_index": chunk.index,
            "text": preview,
        },
    )


def _chunk_edges(
    *,
    chunk: Any,
    chunk_node_id: str,
    entity_nodes: list[GraphNode],
    previous_chunk_id: str | None,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for node in entity_nodes:
        edges.append(
            GraphEdge(
                id=f"{chunk.id}:mentions:{node.id}",
                source_id=chunk_node_id,
                target_id=node.id,
                type="MENTIONS",
                properties={
                    "source_document_id": chunk.document_id,
                    "source_chunk_id": chunk.id,
                    "chunk_index": chunk.index,
                },
            )
        )
    if previous_chunk_id:
        edges.append(
            GraphEdge(
                id=f"{chunk.id}:previous_chunk:{previous_chunk_id}",
                source_id=previous_chunk_id,
                target_id=chunk_node_id,
                type="NEXT_CHUNK",
                properties={
                    "source_document_id": chunk.document_id,
                    "source_chunk_id": chunk.id,
                    "chunk_index": chunk.index,
                },
            )
        )
    return edges
