from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from logger import get_logger
from src.adapters.embedding import EmbeddingAdapter
from src.adapters.storage.graph_store import GraphEdge, GraphNode, GraphStore
from src.adapters.storage.vector_store import VectorDocument, VectorStore
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.g_rag.extractors import EntityRelationExtractor
from src.helpers import Chunk, Document
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.graph_construction import (
    ConstructedChunkGraph,
    DocumentGraphConstructor,
)
from src.optimized_grag.graph_updates.types import GraphUpdateConfig, GraphUpdateReport
from src.optimized_grag.partitioning import CapsuleConstructionPipeline
from src.optimized_grag.retrieval import CapsuleIndex


class GraphUpdateManager:
    """Apply incremental graph changes and refresh optimized GraphRAG indexes."""

    def __init__(
        self,
        *,
        local_graph: LocalGraph,
        graph_store: GraphStore | None = None,
        vector_store: VectorStore | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
        capsule_pipeline: CapsuleConstructionPipeline | None = None,
        capsule_index: CapsuleIndex | None = None,
        extractor: EntityRelationExtractor | None = None,
        config: GraphUpdateConfig | None = None,
        instrumentation: ComputeTracker | None = None,
    ) -> None:
        self.local_graph = local_graph
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_adapter = embedding_adapter
        self.capsule_pipeline = capsule_pipeline
        self.capsule_index = capsule_index
        self.extractor = extractor
        self.config = config or GraphUpdateConfig()
        self.instrumentation = instrumentation or NoopComputeTracker()
        self._logger = get_logger(__name__)

    def apply_documents(
        self,
        documents: Iterable[Document | dict[str, Any] | str],
        *,
        dataset_name: str = "documents",
        config: GraphUpdateConfig | None = None,
    ) -> GraphUpdateReport:
        config = config or self.config
        constructor = DocumentGraphConstructor(
            config=config.construction,
            extractor=self.extractor,
            instrumentation=self.instrumentation,
        )
        construction = constructor.from_documents(documents, dataset_name=dataset_name)

        report = GraphUpdateReport(
            documents=construction.report.documents if construction.report else 0,
            chunks=construction.report.chunks if construction.report else len(construction.chunks),
            graph_nodes=len(construction.nodes),
            graph_edges=len(construction.edges),
            rebuild_mode=config.rebuild_mode,
            details={
                "graph_construction": construction.report.to_dict()
                if construction.report
                else {},
            },
        )
        self._upsert_graph(construction.nodes, construction.edges, report)

        if config.embed_chunks:
            report.vectors = self._upsert_vectors(construction.chunk_graphs)
        else:
            report.details["vectors"] = "skipped_by_config"

        self._maybe_rebuild_capsules(report, config)
        self._logger.info(
            "graph update documents=%s chunks=%s nodes=%s edges=%s vectors=%s capsules=%s",
            report.documents,
            report.chunks,
            report.graph_nodes,
            report.graph_edges,
            report.vectors,
            report.capsules,
        )
        return report

    def apply_graph(
        self,
        *,
        nodes: Iterable[GraphNode] = (),
        edges: Iterable[GraphEdge] = (),
        config: GraphUpdateConfig | None = None,
    ) -> GraphUpdateReport:
        config = config or self.config
        node_list = list(nodes)
        edge_list = list(edges)
        report = GraphUpdateReport(
            graph_nodes=len(node_list),
            graph_edges=len(edge_list),
            rebuild_mode=config.rebuild_mode,
            details={"source": "explicit_graph_update"},
        )
        self._upsert_graph(node_list, edge_list, report)
        self._maybe_rebuild_capsules(report, config)
        return report

    def _upsert_graph(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        report: GraphUpdateReport,
    ) -> None:
        with self.instrumentation.track(
            "graph_update.upsert_graph",
            {"nodes": len(nodes), "edges": len(edges)},
        ):
            self.local_graph.add_nodes(nodes)
            self.local_graph.add_edges(edges)
            if self.graph_store is not None:
                self.graph_store.upsert_nodes(nodes)
                self.graph_store.upsert_edges(edges)

        report.affected_node_ids.update(node.id for node in nodes)
        report.affected_edge_ids.update(edge.id for edge in edges)
        for edge in edges:
            report.affected_node_ids.add(edge.source_id)
            report.affected_node_ids.add(edge.target_id)

    def _upsert_vectors(self, chunk_graphs: list[ConstructedChunkGraph]) -> int:
        if self.vector_store is None or self.embedding_adapter is None:
            return 0

        vector_documents = [
            _chunk_to_vector_document(chunk_graph.chunk, chunk_graph.nodes)
            for chunk_graph in chunk_graphs
        ]
        if not vector_documents:
            return 0

        with self.instrumentation.track(
            "graph_update.embed_chunks",
            {"chunks": len(vector_documents)},
        ):
            vectors = self.embedding_adapter.embed_texts(
                [document.text for document in vector_documents]
            ).vectors
        for document, vector in zip(vector_documents, vectors):
            document.vector = vector

        with self.instrumentation.track(
            "graph_update.upsert_vectors",
            {"vectors": len(vector_documents)},
        ):
            self.vector_store.upsert_many(vector_documents)
        return len(vector_documents)

    def _maybe_rebuild_capsules(
        self,
        report: GraphUpdateReport,
        config: GraphUpdateConfig,
    ) -> None:
        if config.rebuild_mode == "skip":
            report.details["capsule_rebuild"] = "skipped_by_config"
            return
        if self.capsule_pipeline is None:
            report.details["capsule_rebuild"] = "skipped_no_capsule_pipeline"
            return

        with self.instrumentation.track(
            "graph_update.rebuild_capsules",
            {
                "mode": "full",
                "nodes": len(self.local_graph.nodes),
                "edges": len(self.local_graph.edges),
                "affected_nodes": len(report.affected_node_ids),
            },
        ):
            result = self.capsule_pipeline.build(self.local_graph)

        report.capsules = len(result.capsules)
        report.capsule_edges = sum(len(neighbors) for neighbors in result.supergraph.adjacency.values()) // 2
        report.details["capsule_build"] = result.metadata
        report.details["capsule_rebuild_scope"] = "full_graph_current_implementation"

        if self.capsule_index is None or self.embedding_adapter is None:
            report.details["capsule_index"] = "skipped_no_index_or_embeddings"
            return

        with self.instrumentation.track(
            "graph_update.index_capsules",
            {"capsules": len(result.capsules)},
        ):
            self.capsule_index.build(
                capsules=result.capsules,
                embedding_adapter=self.embedding_adapter,
                supergraph=result.supergraph,
            )


def _chunk_to_vector_document(chunk: Chunk, nodes: list[GraphNode]) -> VectorDocument:
    return VectorDocument(
        id=chunk.id,
        vector=[],
        text=chunk.text,
        metadata={
            **chunk.metadata,
            "document_id": chunk.document_id,
            "chunk_index": chunk.index,
            "entity_ids": [node.id for node in nodes],
        },
    )
