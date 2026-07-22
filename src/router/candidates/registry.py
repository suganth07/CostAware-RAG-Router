"""Candidate registry and one approved construction path for the three actions."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.adapters.embedding import EmbeddingAdapter
from src.adapters.llm import LLM
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.compute_eval import InMemoryComputeTracker
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig
from src.optimized_grag.models import FeatureStructuralEncoder
from src.rag.bm25 import BM25RAG
from src.rag.hybrid import HybridRAG
from src.rag.normal import NormalRAG
from src.rag.types import RAGConfig
from src.router.constants import ACTION_IDS
from src.router.candidates.base import CandidateAdapter
from src.router.candidates.graph import GraphRAGCandidate
from src.router.candidates.hybrid import HybridRAGCandidate
from src.router.candidates.optimized_graph import OptimizedGraphRAGCandidate


class CandidateRegistry:
    def __init__(self, candidates: list[CandidateAdapter]) -> None:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        if tuple(by_id) != ACTION_IDS:
            raise ValueError("Candidates must be registered once in the stable router action order")
        self._by_id = by_id

    @property
    def actions(self) -> tuple[str, ...]:
        return ACTION_IDS

    def get(self, candidate_id: str) -> CandidateAdapter:
        return self._by_id[candidate_id]

    def ordered(self) -> list[CandidateAdapter]:
        return [self._by_id[action] for action in ACTION_IDS]


def build_default_registry(*, bundle: Any, embedding_adapter: EmbeddingAdapter, llm: LLM) -> tuple[CandidateRegistry, dict[str, Any]]:
    """Build and index all router candidates without learned R-GCN artifacts."""
    rag_config = RAGConfig(chunk_size=900, chunk_overlap=120, top_k=12)
    normal = NormalRAG(
        vector_store=InMemoryVectorStore(), embedding_adapter=embedding_adapter, llm=llm,
        config=rag_config, instrumentation=InMemoryComputeTracker(log_stages=False, namespace="dense"),
    )
    bm25 = BM25RAG(llm=llm, config=rag_config, instrumentation=InMemoryComputeTracker(log_stages=False, namespace="bm25"))
    hybrid = HybridRAG(
        normal_rag=normal, bm25_rag=bm25, llm=llm, config=rag_config,
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="hybrid"),
    )
    graph_config = GraphRAGConfig(top_k=12, graph_depth=3, graph_limit=1200, ranked_graph_limit=240)
    graph = GraphRAG(
        graph_store=InMemoryGraphStore(), vector_store=InMemoryVectorStore(), embedding_adapter=embedding_adapter, llm=llm,
        config=graph_config, instrumentation=InMemoryComputeTracker(log_stages=False, namespace="graph"),
    )
    optimized_config = OptimizedGraphRAGConfig(
        max_region_nodes=1000, max_capsule_nodes=250, max_capsule_edges=900,
        first_stage_candidate_capsules=100, top_k_capsules=12,
        max_candidate_paths=11, max_path_length=3, final_paths_to_llm=11,
        max_seed_neighbors=200, seed_expansion_depth=3, max_seed_expansion_nodes=6000,
    )
    optimized = OptimizedGraphRAG(
        graph_store=InMemoryGraphStore(), vector_store=InMemoryVectorStore(), embedding_adapter=embedding_adapter, llm=llm,
        config=optimized_config, structural_encoder=FeatureStructuralEncoder(),
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="optimized"),
    )
    native_graph = bool(bundle.graph_nodes and bundle.graph_edges)
    reports = {
        "hybrid_rag": _jsonable(hybrid.index_documents(bundle.documents)),
        "graph_rag": _jsonable(graph.index_dataset_bundle(bundle, include_documents=True)),
        "optimized_graph_rag": _jsonable(optimized.index_dataset_bundle(bundle, include_documents=not native_graph)),
    }
    registry = CandidateRegistry([
        HybridRAGCandidate(hybrid, fingerprint={"config": asdict(rag_config), "implementation": "HybridRAG"}),
        GraphRAGCandidate(graph, fingerprint={"config": asdict(graph_config), "implementation": "GraphRAG"}),
        OptimizedGraphRAGCandidate(optimized, fingerprint={"config": asdict(optimized_config), "implementation": "OptimizedGraphRAG", "structural_encoder": "FeatureStructuralEncoder"}),
    ])
    return registry, reports


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
