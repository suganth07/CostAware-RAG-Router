from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional

from logger import get_logger
from src.adapters.embedding import EmbeddingAdapter
from src.adapters.llm import LLM
from src.adapters.storage.graph_store import GraphEdge, GraphNode, GraphStore
from src.adapters.storage.vector_store import VectorDocument, VectorStore
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.g_rag.context import ContextBuilder
from src.g_rag.extractors import EntityRelationExtractor, SimpleEntityRelationExtractor
from src.g_rag.ranking import ProductionEvidenceRanker
from src.g_rag.types import (
    Chunk,
    Document,
    GraphRAGAnswer,
    GraphRAGConfig,
    IndexingReport,
    RetrievedContext,
)
from src.helpers.chunking import TextChunker
from src.helpers.documents import normalize_documents


class GraphRAG:
    def __init__(
        self,
        *,
        graph_store: GraphStore,
        vector_store: VectorStore,
        embedding_adapter: EmbeddingAdapter,
        llm: LLM,
        extractor: Optional[EntityRelationExtractor] = None,
        config: Optional[GraphRAGConfig] = None,
        instrumentation: Optional[ComputeTracker] = None,
    ) -> None:
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_adapter = embedding_adapter
        self.llm = llm
        self.config = config or GraphRAGConfig()
        self.extractor = extractor or SimpleEntityRelationExtractor()
        self.instrumentation = instrumentation or NoopComputeTracker()
        self.chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        self.context_builder = ContextBuilder(max_chars=self.config.max_context_chars)
        self.evidence_ranker = ProductionEvidenceRanker()
        self._logger = get_logger(__name__)

    def index_documents(
        self,
        documents: Iterable[Document | dict[str, Any] | str],
        *,
        extra_graph_nodes: Optional[list[GraphNode]] = None,
        extra_graph_edges: Optional[list[GraphEdge]] = None,
        extract_document_graph: bool = True,
    ) -> IndexingReport:
        normalized_documents = normalize_documents(documents)
        report = IndexingReport(documents=len(normalized_documents))
        with self.instrumentation.track(
            "upsert_native_graph",
            {
                "nodes": len(extra_graph_nodes or []),
                "edges": len(extra_graph_edges or []),
            },
        ):
            if extra_graph_nodes:
                self.graph_store.upsert_nodes(extra_graph_nodes)
                report.graph_nodes += len(extra_graph_nodes)
            if extra_graph_edges:
                self.graph_store.upsert_edges(extra_graph_edges)
                report.graph_edges += len(extra_graph_edges)
        if not normalized_documents:
            return report

        with self.instrumentation.track("chunk_documents", {"documents": len(normalized_documents)}):
            chunks = self.chunker.chunk_documents(normalized_documents)
        report.chunks = len(chunks)
        self.instrumentation.add_count("chunks", len(chunks))

        index_batch_size = max(1, int(self.config.index_batch_size))
        for start in range(0, len(chunks), index_batch_size):
            batch = chunks[start : start + index_batch_size]
            vector_documents: list[VectorDocument] = []
            with self.instrumentation.track(
                "prepare_index_batch",
                {"chunks": len(batch), "extract_graph": extract_document_graph},
            ):
                for chunk in batch:
                    if extract_document_graph:
                        extraction = self.extractor.extract(chunk)
                        report.graph_nodes += len(extraction.nodes)
                        report.graph_edges += len(extraction.edges)
                        self.graph_store.upsert_nodes(extraction.nodes)
                        self.graph_store.upsert_edges(extraction.edges)
                        entity_ids = [node.id for node in extraction.nodes]
                    else:
                        entity_ids = _native_chunk_entity_ids(chunk)
                    chunk.metadata.update(
                        {
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.index,
                            "entity_ids": entity_ids,
                        }
                    )
                    vector_documents.append(
                        VectorDocument(
                            id=chunk.id,
                            vector=[],
                            text=chunk.text,
                            metadata=chunk.metadata,
                        )
                    )
            report.vectors += len(vector_documents)
            existing_ids = self.vector_store.existing_ids(
                document.id for document in vector_documents
            )
            missing = [
                document
                for document in vector_documents
                if document.id not in existing_ids
            ]
            report.cached_vectors += len(vector_documents) - len(missing)
            if not missing:
                continue
            with self.instrumentation.track(
                "embed_index_batch", {"chunks": len(missing)}
            ):
                embedding_result = self.embedding_adapter.embed_texts(
                    [document.text for document in missing]
                )
            if len(embedding_result.vectors) != len(missing):
                raise ValueError(
                    "Embedding adapter returned a different number of vectors "
                    f"({len(embedding_result.vectors)} != {len(missing)})"
                )
            for document, vector in zip(missing, embedding_result.vectors):
                document.vector = vector
            with self.instrumentation.track(
                "upsert_vector_batch", {"vectors": len(missing)}
            ):
                self.vector_store.upsert_many(missing)
            report.embedded_vectors += len(missing)

        with self.instrumentation.track(
            "persist_vector_index", {"vectors": report.vectors}
        ):
            self.vector_store.flush()
        self.instrumentation.add_count("vectors", report.vectors)

        self._logger.info(
            "indexed documents=%s chunks=%s graph_nodes=%s graph_edges=%s "
            "vectors=%s embedded=%s cached=%s",
            report.documents,
            report.chunks,
            report.graph_nodes,
            report.graph_edges,
            report.vectors,
            report.embedded_vectors,
            report.cached_vectors,
        )
        return report

    def index_dataset_bundle(
        self,
        bundle: Any,
        *,
        include_documents: bool = True,
    ) -> IndexingReport:
        """Index source evidence and use a supplied native graph as authoritative.

        When a dataset already provides nodes and typed edges, source documents
        are still embedded for hybrid retrieval but are not re-extracted into a
        second, noisier graph.  ``include_documents=False`` remains available
        for deliberately graph-only ablations.
        """
        graph_nodes = list(getattr(bundle, "graph_nodes", []) or [])
        graph_edges = list(getattr(bundle, "graph_edges", []) or [])
        return self.index_documents(
            bundle.documents if include_documents else [],
            extra_graph_nodes=graph_nodes,
            extra_graph_edges=graph_edges,
            extract_document_graph=not bool(graph_nodes and graph_edges),
        )

    def retrieve(self, question: str, *, top_k: Optional[int] = None) -> RetrievedContext:
        limit = self.config.top_k if top_k is None else max(0, int(top_k))
        question_chunk = Chunk(id="query", document_id="query", text=question, index=0)

        with self.instrumentation.track("embed_query", {"chars": len(question)}):
            query_vector = self.embedding_adapter.embed_text(question)

        vector_candidate_limit = max(limit, int(self.config.vector_candidate_k))
        with self.instrumentation.track(
            "dense_candidate_search", {"candidates": vector_candidate_limit}
        ):
            vector_candidates = self.vector_store.search(
                query_vector,
                limit=vector_candidate_limit,
            )
        with self.instrumentation.track(
            "rank_vector_evidence",
            {"candidates": len(vector_candidates), "selected": limit},
        ):
            vector_results = self.evidence_ranker.rank_vectors(
                query=question,
                results=vector_candidates,
                limit=limit,
            )

        with self.instrumentation.track("extract_query_entities", {"chars": len(question)}):
            query_extraction = self.extractor.extract(question_chunk)
        query_seed_node_ids = [
            node.id
            for node in query_extraction.nodes
            if self.graph_store.get_node(node.id) is not None
        ]

        vector_seed_node_ids: list[str] = []
        for result in vector_results[: max(0, int(self.config.vector_seed_k))]:
            vector_seed_node_ids.extend(
                str(value)
                for value in (result.document.metadata.get("entity_ids") or [])
            )
        vector_seed_node_ids = _ordered_unique(vector_seed_node_ids)[
            : max(0, int(self.config.max_vector_seed_entities))
        ]
        seed_node_ids = [*query_seed_node_ids, *vector_seed_node_ids]
        seed_node_ids = _ordered_unique(seed_node_ids)

        with self.instrumentation.track(
            "graph_expand",
            {"seeds": len(seed_node_ids), "depth": self.config.graph_depth},
        ):
            expanded_neighborhood = self.graph_store.get_neighbors(
                seed_node_ids,
                depth=self.config.graph_depth,
                limit=self.config.graph_limit,
            )

        with self.instrumentation.track(
            "rank_graph_evidence",
            {
                "edges": len(expanded_neighborhood.edges),
                "selected": self.config.ranked_graph_limit,
            },
        ):
            ranked_graph = self.evidence_ranker.rank_graph(
                query=question,
                neighborhood=expanded_neighborhood,
                seed_node_ids=seed_node_ids,
                vector_results=vector_results,
                max_depth=self.config.graph_depth,
                limit=self.config.ranked_graph_limit,
            )
        graph_neighborhood = ranked_graph.neighborhood

        with self.instrumentation.track("build_context"):
            context_text = self.context_builder.build(
                vector_results=vector_results,
                graph_neighborhood=graph_neighborhood,
                graph_edge_scores=ranked_graph.edge_scores,
                graph_edge_depths=ranked_graph.edge_depths,
            )

        return RetrievedContext(
            question=question,
            vector_results=vector_results,
            graph_neighborhood=graph_neighborhood,
            context_text=context_text,
            seed_node_ids=seed_node_ids,
            details={
                "retrieval_contract": "production_hybrid_graph_v1",
                "dense_candidates": len(vector_candidates),
                "ranked_vector_results": len(vector_results),
                "query_seed_nodes": len(query_seed_node_ids),
                "vector_seed_nodes": len(vector_seed_node_ids),
                "expanded_graph_nodes": len(expanded_neighborhood.nodes),
                "expanded_graph_edges": len(expanded_neighborhood.edges),
                "ranked_graph_edges": len(graph_neighborhood.edges),
                "graph_edge_scores": ranked_graph.edge_scores,
                "graph_edge_depths": ranked_graph.edge_depths,
                "vector_index_cache_loaded": bool(
                    getattr(self.vector_store, "cache_loaded", False)
                ),
            },
        )

    def answer(self, question: str, *, top_k: Optional[int] = None) -> GraphRAGAnswer:
        context = self.retrieve(question, top_k=top_k)
        prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{context.context_text}\n\n"
            "Answer with a concise response grounded in the context."
        )
        with self.instrumentation.track("generate_answer", {"context_chars": len(context.context_text)}):
            response = self.llm.generate(prompt, system_prompt=self.config.system_prompt)

        return GraphRAGAnswer(
            question=question,
            answer=response.text,
            context=context,
            usage={
                "llm": response.usage,
                "instrumentation": self.instrumentation.snapshot(),
            },
        )

def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _native_chunk_entity_ids(chunk: Chunk) -> list[str]:
    values: list[str] = []
    for value in chunk.metadata.get("entity_ids") or []:
        values.append(str(value))
    for key in ("subject", "object", "entity", "title"):
        value = chunk.metadata.get(key)
        if value:
            values.append(_entity_id(str(value)))
    return _ordered_unique(values)


def _entity_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
