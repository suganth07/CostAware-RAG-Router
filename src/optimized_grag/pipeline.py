from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Optional

from logger import get_logger
from src.adapters.embedding import EmbeddingAdapter
from src.adapters.llm import LLM
from src.adapters.storage.graph_store import GraphEdge, GraphNode, GraphStore
from src.adapters.storage.vector_store import VectorDocument, VectorStore
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.g_rag.extractors import EntityRelationExtractor, SimpleEntityRelationExtractor
from src.helpers import Chunk, Document, normalize_documents
from src.optimized_grag.config import OptimizedGraphRAGConfig
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.graph_construction import DocumentGraphConstructor, GraphConstructionConfig
from src.optimized_grag.graph_updates import GraphUpdateConfig, GraphUpdateManager, GraphUpdateReport
from src.optimized_grag.models import FeatureStructuralEncoder, StructuralEncoder
from src.optimized_grag.partitioning import CapsuleCandidate, CapsuleConstructionPipeline
from src.optimized_grag.retrieval import (
    BoundedLocalPathSearcher,
    CapsuleIndex,
    CapsuleRouter,
    EvidencePath,
    OptimizedContextBuilder,
    SourceSnippet,
)
from src.optimized_grag.types import OptimizedIndexingReport


@dataclass(slots=True)
class OptimizedRetrievedContext:
    question: str
    selected_capsules: list[CapsuleCandidate]
    evidence_paths: list[EvidencePath]
    source_snippets: list[SourceSnippet]
    context_text: str
    seed_node_ids: list[str]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizedGraphRAGAnswer:
    question: str
    answer: str
    context: OptimizedRetrievedContext
    usage: dict[str, Any] = field(default_factory=dict)


class OptimizedGraphRAG:
    """Hierarchical capsule GraphRAG with bounded local path search."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedding_adapter: EmbeddingAdapter,
        llm: LLM,
        extractor: Optional[EntityRelationExtractor] = None,
        config: Optional[OptimizedGraphRAGConfig] = None,
        instrumentation: Optional[ComputeTracker] = None,
        structural_encoder: Optional[StructuralEncoder] = None,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_adapter = embedding_adapter
        self.llm = llm
        self.extractor = extractor or SimpleEntityRelationExtractor()
        self.config = config or OptimizedGraphRAGConfig()
        self.instrumentation = instrumentation or NoopComputeTracker()
        self.structural_encoder = structural_encoder or FeatureStructuralEncoder()
        construction_config = GraphConstructionConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            use_native_graph=False,
        )
        self.graph_constructor = DocumentGraphConstructor(
            config=construction_config,
            extractor=self.extractor,
            instrumentation=self.instrumentation,
        )
        self.local_graph = LocalGraph()
        self.capsule_index = CapsuleIndex(structural_encoder=self.structural_encoder)
        self.capsule_pipeline = CapsuleConstructionPipeline(
            config=self.config,
            instrumentation=self.instrumentation,
        )
        self.router = CapsuleRouter(
            capsule_index=self.capsule_index,
            structural_encoder=self.structural_encoder,
        )
        self.path_searcher = BoundedLocalPathSearcher(
            max_merged_nodes=self.config.max_merged_nodes,
            max_candidate_paths=self.config.max_candidate_paths,
            max_path_length=self.config.max_path_length,
            final_paths=self.config.final_paths_to_llm,
            max_seed_neighbors=self.config.max_seed_neighbors,
            seed_expansion_depth=self.config.seed_expansion_depth,
            max_seed_expansion_nodes=self.config.max_seed_expansion_nodes,
            direct_seed_path_search=self.config.direct_seed_path_search,
        )
        self.context_builder = OptimizedContextBuilder(
            max_chars=self.config.max_context_chars,
            snippets_per_capsule=self.config.snippets_per_capsule,
            max_source_snippets=self.config.max_source_snippets,
        )
        self.graph_updater = GraphUpdateManager(
            local_graph=self.local_graph,
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            embedding_adapter=self.embedding_adapter,
            capsule_pipeline=self.capsule_pipeline,
            capsule_index=self.capsule_index,
            extractor=self.extractor,
            config=GraphUpdateConfig(construction=construction_config),
            instrumentation=self.instrumentation,
        )
        self._logger = get_logger(__name__)

    def index_documents(
        self,
        documents: Iterable[Document | dict[str, Any] | str],
        *,
        extra_graph_nodes: Optional[list[GraphNode]] = None,
        extra_graph_edges: Optional[list[GraphEdge]] = None,
    ) -> OptimizedIndexingReport:
        normalized_documents = normalize_documents(documents)
        report = OptimizedIndexingReport(documents=len(normalized_documents))
        started = time.perf_counter()
        self._logger.info(
            "optimized_grag indexing started documents=%s native_nodes=%s native_edges=%s",
            len(normalized_documents),
            len(extra_graph_nodes or []),
            len(extra_graph_edges or []),
        )

        if extra_graph_nodes:
            self._add_graph_nodes(extra_graph_nodes)
            report.graph_nodes += len(extra_graph_nodes)
        if extra_graph_edges:
            self._add_graph_edges(extra_graph_edges)
            report.graph_edges += len(extra_graph_edges)
        self._logger.info(
            "optimized_grag native graph stage completed nodes=%s edges=%s elapsed_s=%.2f",
            report.graph_nodes,
            report.graph_edges,
            time.perf_counter() - started,
        )

        if normalized_documents:
            construction = self.graph_constructor.from_documents(normalized_documents)
            if construction.report is not None:
                report.chunks = construction.report.chunks
                report.details["graph_construction"] = construction.report.to_dict()

            vector_documents: list[VectorDocument] = []
            report.graph_nodes += len(construction.nodes)
            report.graph_edges += len(construction.edges)
            self._add_graph_nodes(construction.nodes)
            self._add_graph_edges(construction.edges)
            for chunk_graph in construction.chunk_graphs:
                vector_documents.append(
                    self._chunk_to_vector_document(chunk_graph.chunk, chunk_graph.nodes)
                )

            if vector_documents:
                with self.instrumentation.track("optimized.embed_chunks", {"chunks": len(vector_documents)}):
                    vectors = self.embedding_adapter.embed_texts([document.text for document in vector_documents]).vectors
                for document, vector in zip(vector_documents, vectors):
                    document.vector = vector
                with self.instrumentation.track("optimized.upsert_vectors", {"vectors": len(vector_documents)}):
                    self.vector_store.upsert_many(vector_documents)
                report.vectors = len(vector_documents)

        self.rebuild_capsules(report)
        self._logger.info(
            "optimized_grag indexed documents=%s chunks=%s nodes=%s edges=%s capsules=%s elapsed_s=%.2f",
            report.documents,
            report.chunks,
            report.graph_nodes,
            report.graph_edges,
            report.capsules,
            time.perf_counter() - started,
        )
        return report

    def index_dataset_bundle(
        self,
        bundle: Any,
        *,
        include_documents: bool = True,
    ) -> OptimizedIndexingReport:
        """Index a bundle, optionally building capsules from its native graph only."""
        return self.index_documents(
            bundle.documents if include_documents else [],
            extra_graph_nodes=list(getattr(bundle, "graph_nodes", []) or []),
            extra_graph_edges=list(getattr(bundle, "graph_edges", []) or []),
        )

    def update_documents(
        self,
        documents: Iterable[Document | dict[str, Any] | str],
        *,
        dataset_name: str = "documents",
        config: GraphUpdateConfig | None = None,
    ) -> GraphUpdateReport:
        return self.graph_updater.apply_documents(
            documents,
            dataset_name=dataset_name,
            config=config,
        )

    def update_graph(
        self,
        *,
        nodes: Iterable[GraphNode] = (),
        edges: Iterable[GraphEdge] = (),
        config: GraphUpdateConfig | None = None,
    ) -> GraphUpdateReport:
        return self.graph_updater.apply_graph(nodes=nodes, edges=edges, config=config)

    def rebuild_capsules(self, report: OptimizedIndexingReport | None = None) -> None:
        started = time.perf_counter()
        self._logger.info(
            "optimized_grag capsule rebuild started nodes=%s edges=%s",
            len(self.local_graph.nodes),
            len(self.local_graph.edges),
        )
        with self.instrumentation.track(
            "optimized.rebuild_capsules",
            {"nodes": len(self.local_graph.nodes), "edges": len(self.local_graph.edges)},
        ):
            result = self.capsule_pipeline.build(self.local_graph)
        artifact_match_rate = None
        bind_capsules = getattr(self.structural_encoder, "bind_capsules", None)
        if callable(bind_capsules):
            artifact_match_rate = bind_capsules(result.capsules)
        with self.instrumentation.track("optimized.index_capsules", {"capsules": len(result.capsules)}):
            self.capsule_index.build(
                capsules=result.capsules,
                embedding_adapter=self.embedding_adapter,
                supergraph=result.supergraph,
            )
        with self.instrumentation.track(
            "optimized.prepare_path_index",
            {"nodes": len(self.local_graph.nodes), "edges": len(self.local_graph.edges)},
        ):
            self.path_searcher.prepare_graph(self.local_graph)
        if report is not None:
            report.capsules = len(result.capsules)
            report.capsule_edges = sum(len(neighbors) for neighbors in result.supergraph.adjacency.values()) // 2
            report.details["capsule_build"] = result.metadata
            report.details["capsule_index"] = {
                "retrieval_mode": self.capsule_index.retrieval_mode,
                "summary_embeddings_skipped": (
                    self.capsule_index.retrieval_mode.startswith("trained_")
                ),
            }
            if artifact_match_rate is not None:
                report.details["trained_capsule_match_rate"] = artifact_match_rate
                report.details["trained_structural_space_enabled"] = bool(
                    getattr(self.structural_encoder, "trained_space_enabled", False)
                )
                report.details["runtime_capsule_fingerprint"] = getattr(
                    self.structural_encoder, "runtime_fingerprint", None
                )
                report.details["artifact_capsule_fingerprint"] = getattr(
                    self.structural_encoder, "artifact_fingerprint", None
                )
        self._logger.info(
            "optimized_grag capsule rebuild completed capsules=%s retrieval_mode=%s elapsed_s=%.2f",
            len(result.capsules),
            self.capsule_index.retrieval_mode,
            time.perf_counter() - started,
        )

    def retrieve(self, question: str) -> OptimizedRetrievedContext:
        query_chunk = Chunk(id="query", document_id="query", text=question, index=0)
        with self.instrumentation.track("optimized.embed_query", {"chars": len(question)}):
            query_vector = self.embedding_adapter.embed_text(question)

        with self.instrumentation.track("optimized.extract_query_entities", {"chars": len(question)}):
            extraction = self.extractor.extract(query_chunk)
        query_entities = [node.properties.get("name") or node.id for node in extraction.nodes]
        seed_node_ids = [node.id for node in extraction.nodes if node.id in self.local_graph.nodes]

        with self.instrumentation.track("optimized.route_capsules", {"candidates": self.config.first_stage_candidate_capsules}):
            if self.config.exact_seed_capsule_routing and seed_node_ids:
                selected_capsules = self.capsule_index.exact_seed_candidates(
                    seed_node_ids=seed_node_ids,
                    limit=self.config.top_k_capsules,
                )
                capsule_route_mode = "exact_seed_inverted_index"
            else:
                selected_capsules = self.router.route(
                    query=question,
                    query_vector=query_vector,
                    query_entities=[str(entity) for entity in query_entities],
                    seed_node_ids=seed_node_ids,
                    first_stage_limit=self.config.first_stage_candidate_capsules,
                    top_k=self.config.top_k_capsules,
                )
                capsule_route_mode = "learned_projection_fallback"

        with self.instrumentation.track("optimized.bounded_path_search", {"capsules": len(selected_capsules)}):
            paths = self.path_searcher.search(
                capsules=[candidate.capsule for candidate in selected_capsules],
                seed_node_ids=seed_node_ids,
                query_terms=_query_terms(question),
                seed_graph=self.local_graph,
            )

        with self.instrumentation.track("optimized.build_context", {"paths": len(paths)}):
            built_context = self.context_builder.build(
                query=question,
                candidates=selected_capsules,
                paths=paths,
            )

        return OptimizedRetrievedContext(
            question=question,
            selected_capsules=selected_capsules,
            evidence_paths=paths,
            source_snippets=built_context.source_snippets,
            context_text=built_context.text,
            seed_node_ids=seed_node_ids,
            details={
                "query_entities": query_entities,
                "candidate_capsules": self.config.first_stage_candidate_capsules,
                "top_k_capsules": self.config.top_k_capsules,
                "capsule_retrieval_mode": self.capsule_index.retrieval_mode,
                "capsule_route_mode": capsule_route_mode,
                "selected_capsule_ids": [
                    candidate.capsule.id for candidate in selected_capsules
                ],
                "matched_seed_ids": sorted(
                    {
                        seed_id
                        for candidate in selected_capsules
                        for seed_id in (candidate.metadata.get("matched_seed_ids") or [])
                    }
                ),
                "ranked_source_ids": [
                    snippet.document_id or snippet.chunk_id or snippet.id
                    for snippet in built_context.source_snippets
                ],
                "path_search_mode": self.path_searcher.last_search_mode,
            },
        )

    def answer(self, question: str) -> OptimizedGraphRAGAnswer:
        context = self.retrieve(question)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Optimized GraphRAG context:\n{context.context_text}\n\n"
            "Answer with a concise response grounded in the graph capsule evidence."
        )
        with self.instrumentation.track("optimized.generate_answer", {"context_chars": len(context.context_text)}):
            response = self.llm.generate(prompt, system_prompt=self.config.system_prompt)
        return OptimizedGraphRAGAnswer(
            question=question,
            answer=response.text,
            context=context,
            usage={
                "llm": response.usage,
                "instrumentation": self.instrumentation.snapshot(),
            },
        )

    def _add_graph_nodes(self, nodes: list[GraphNode]) -> None:
        self.local_graph.add_nodes(nodes)
        self.graph_store.upsert_nodes(nodes)

    def _add_graph_edges(self, edges: list[GraphEdge]) -> None:
        self.local_graph.add_edges(edges)
        self.graph_store.upsert_edges(edges)

    def _chunk_to_vector_document(self, chunk: Chunk, nodes: list[GraphNode]) -> VectorDocument:
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


def _query_terms(question: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9_]+", question)}
