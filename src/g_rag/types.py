from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.adapters.storage.graph_store import GraphNeighborhood
from src.adapters.storage.vector_store import VectorSearchResult
from src.helpers.documents import Chunk, Document


@dataclass(slots=True)
class GraphRAGConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 5
    vector_candidate_k: int = 40
    vector_seed_k: int = 12
    max_vector_seed_entities: int = 64
    graph_depth: int = 2
    graph_limit: int = 80
    ranked_graph_limit: int = 120
    index_batch_size: int = 512
    max_context_chars: int = 12_000
    system_prompt: str = (
        "You answer questions using only the supplied vector and graph context. "
        "If the context is insufficient, say what is missing."
    )


@dataclass(slots=True)
class IndexingReport:
    documents: int = 0
    chunks: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    vectors: int = 0
    embedded_vectors: int = 0
    cached_vectors: int = 0


@dataclass(slots=True)
class RetrievedContext:
    question: str
    vector_results: list[VectorSearchResult]
    graph_neighborhood: GraphNeighborhood
    context_text: str
    seed_node_ids: list[str]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphRAGAnswer:
    question: str
    answer: str
    context: RetrievedContext
    usage: dict[str, Any] = field(default_factory=dict)
