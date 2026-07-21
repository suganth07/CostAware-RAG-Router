from __future__ import annotations

from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult
from src.adapters.datasets.base import DatasetBundle
from src.adapters.llm.adapter import LLM, LLMResponse
from src.adapters.storage.graph_store import GraphEdge, GraphNode, InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag.config import OptimizedGraphRAGConfig
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.partitioning import CapsuleConstructionPipeline
from src.optimized_grag.partitioning.types import GraphCapsule
from src.optimized_grag.retrieval import CapsuleIndex, CapsuleRouter
from src.helpers import Document


class KeywordEmbedding(EmbeddingAdapter):
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("ginger rogers" in lowered),
                    float("irrelevant" in lowered),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        return EmbeddingResult(vectors=vectors, model="keyword")


class NoopLLM(LLM):
    def generate(self, prompt, *, system_prompt=None, **kwargs):
        return LLMResponse(text="", model="noop")


class ZeroStructuralEncoder:
    def encode_capsule(self, capsule):
        return [0.0] * 8

    def project_query(self, query_vector):
        return [0.0] * 8


def test_native_graph_only_keeps_metaqa_fact_retrievable() -> None:
    ginger = GraphNode(id="ginger_rogers", properties={"name": "Ginger Rogers"})
    top_hat = GraphNode(id="top_hat", properties={"name": "Top Hat"})
    edge = GraphEdge(
        id="top_hat:starred_actors:ginger_rogers",
        source_id="top_hat",
        target_id="ginger_rogers",
        type="STARRED_ACTORS",
    )
    pipeline = GraphRAG(
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
        embedding_adapter=KeywordEmbedding(),
        llm=NoopLLM(),
        config=GraphRAGConfig(graph_depth=1, graph_limit=20),
    )

    report = pipeline.index_dataset_bundle(
        DatasetBundle(
            name="MetaQA fixture",
            documents=[Document(id="unused", text="This document must not be embedded.")],
            graph_nodes=[ginger, top_hat],
            graph_edges=[edge],
        ),
        include_documents=False,
    )
    context = pipeline.retrieve("what movies are about [ginger rogers]")

    assert report.documents == 0
    assert report.vectors == 0
    assert "Top Hat" in context.context_text


def test_capsule_router_preserves_exact_query_seed() -> None:
    exact = GraphCapsule(
        id="capsule:exact",
        anchor_id="ginger_rogers",
        region_id="region:exact",
        node_ids={"ginger_rogers", "top_hat"},
        edge_ids=set(),
        nodes=[
            GraphNode(id="ginger_rogers", properties={"name": "Ginger Rogers"}),
            GraphNode(id="top_hat", properties={"name": "Top Hat"}),
        ],
        edges=[],
    )
    irrelevant = GraphCapsule(
        id="capsule:irrelevant",
        anchor_id="irrelevant",
        region_id="region:irrelevant",
        node_ids={"irrelevant"},
        edge_ids=set(),
        nodes=[GraphNode(id="irrelevant", properties={"name": "Irrelevant"})],
        edges=[],
    )
    encoder = ZeroStructuralEncoder()
    index = CapsuleIndex(structural_encoder=encoder)
    index.build(
        capsules=[irrelevant, exact],
        embedding_adapter=KeywordEmbedding(),
        supergraph=type("EmptySupergraph", (), {"neighbors": lambda self, _: {}})(),
    )
    router = CapsuleRouter(capsule_index=index, structural_encoder=encoder)

    selected = router.route(
        query="what movies are about [ginger rogers]",
        query_vector=[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        query_entities=["ginger rogers"],
        seed_node_ids=["ginger_rogers"],
        first_stage_limit=2,
        top_k=1,
    )

    assert selected[0].capsule.id == "capsule:exact"
    assert selected[0].metadata["exact_seed_match"] is True

    direct = index.exact_seed_candidates(
        seed_node_ids=["ginger_rogers"],
        limit=1,
    )
    assert direct[0].capsule.id == "capsule:exact"
    assert direct[0].metadata["stage"] == "exact_seed_inverted_index"


def test_capsule_ids_are_stable_across_input_order() -> None:
    nodes = [GraphNode(id=node_id, properties={"name": node_id}) for node_id in "abcdef"]
    edges = [
        GraphEdge(id=f"e{i}", source_id=left, target_id=right, type="RELATED_TO")
        for i, (left, right) in enumerate(zip("abcde", "bcdef"))
    ]
    config = OptimizedGraphRAGConfig(
        max_region_nodes=3,
        max_capsule_nodes=4,
        anchors_per_region=2,
        max_capsules_per_region=2,
    )

    first = CapsuleConstructionPipeline(config=config).build(
        LocalGraph.from_nodes_edges(nodes, edges)
    )
    second = CapsuleConstructionPipeline(config=config).build(
        LocalGraph.from_nodes_edges(reversed(nodes), reversed(edges))
    )

    first_signature = [(capsule.id, sorted(capsule.node_ids)) for capsule in first.capsules]
    second_signature = [(capsule.id, sorted(capsule.node_ids)) for capsule in second.capsules]
    assert first_signature == second_signature
