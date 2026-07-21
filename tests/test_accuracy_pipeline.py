from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.adapters.datasets.base import EvaluationSample
from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    EmbeddingAdapter,
    EmbeddingResult,
    local_embedding_cache_namespace,
)
from src.adapters.storage.graph_store import GraphEdge, GraphNode, InMemoryGraphStore
from src.evaluation import (
    CachedEvaluationRunner,
    EvaluationCache,
    EvidenceUnit,
    RetrievalRecord,
    evaluate_record,
)
from src.optimized_grag import OptimizedGraphRAGConfig
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.partitioning import CapsuleConstructionPipeline
from src.optimized_grag.partitioning.capsule_builder import CapsuleBuilder
from src.optimized_grag.partitioning.hubs import HubProfile
from src.optimized_grag.partitioning.types import CapsuleCandidate, GraphCapsule
from src.optimized_grag.retrieval import BoundedLocalPathSearcher, OptimizedContextBuilder
from src.optimized_grag.models import (
    any_positive_ranking_metrics,
    mine_metaqa_path_projection_examples,
    mine_projection_examples,
)


class CountingEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return EmbeddingResult(
            vectors=[[float(len(text)), 1.0] for text in texts],
            model="counting",
        )


def test_capsule_construction_guarantees_every_node_is_covered() -> None:
    nodes = [GraphNode(id=f"n{i}", properties={"name": f"Node {i}"}) for i in range(8)]
    edges = [
        GraphEdge(id=f"e{i}", source_id=f"n{i}", target_id=f"n{i + 1}", type="LINK")
        for i in range(7)
    ]
    graph = LocalGraph.from_nodes_edges(nodes, edges)
    result = CapsuleConstructionPipeline(
        config=OptimizedGraphRAGConfig(
            max_region_nodes=20,
            max_hops=1,
            max_capsule_nodes=2,
            anchors_per_region=1,
            max_capsules_per_region=1,
            ensure_node_coverage=True,
        )
    ).build(graph)

    covered = set().union(*(capsule.node_ids for capsule in result.capsules))
    assert covered == set(graph.nodes)
    assert result.metadata["node_coverage_rate"] == 1.0
    assert result.metadata["coverage_capsule_count"] > 0


def test_boundary_overlap_uses_stable_id_tie_break() -> None:
    nodes = [GraphNode(id=value) for value in ("a", "b", "c", "d")]
    edges = [
        GraphEdge(id="a-c", source_id="a", target_id="c", type="LINK"),
        GraphEdge(id="b-d", source_id="b", target_id="d", type="LINK"),
    ]
    graph = LocalGraph.from_nodes_edges(nodes, edges)
    hubs = HubProfile(
        hub_node_ids=set(),
        threshold=99.0,
        degrees={node.id: graph.degree(node.id) for node in nodes},
    )

    selected = CapsuleBuilder()._boundary_overlap(
        graph,
        {"a", "b"},
        {"a", "b", "c", "d"},
        hubs,
    )

    assert selected == {"c"}


def test_path_search_reserves_final_evidence_for_each_seed() -> None:
    first = _two_node_capsule("first", "seed_a", "answer_a")
    second = _two_node_capsule("second", "seed_b", "answer_b")
    searcher = BoundedLocalPathSearcher(
        max_merged_nodes=20,
        max_candidate_paths=2,
        max_path_length=1,
        final_paths=2,
    )

    paths = searcher.search(
        capsules=[first, second],
        seed_node_ids=["seed_a", "seed_b"],
        query_terms={"answer"},
    )

    assert {path.metadata["seed_node_id"] for path in paths} == {"seed_a", "seed_b"}


def test_path_search_adds_exact_seed_neighbors_across_capsule_boundaries() -> None:
    seed = GraphNode(id="seed", properties={"name": "Seed"})
    answer = GraphNode(id="answer", properties={"name": "Answer"})
    edge = GraphEdge(
        id="seed:answer",
        source_id="seed",
        target_id="answer",
        type="DIRECT_FACT",
    )
    source_graph = LocalGraph.from_nodes_edges([seed, answer], [edge])
    incomplete_capsule = GraphCapsule(
        id="capsule:incomplete",
        anchor_id="seed",
        region_id="region:incomplete",
        node_ids={"seed"},
        edge_ids=set(),
        nodes=[seed],
        edges=[],
    )
    searcher = BoundedLocalPathSearcher(
        max_candidate_paths=10,
        max_path_length=1,
        final_paths=3,
        max_seed_neighbors=10,
    )
    paths = searcher.search(
        capsules=[incomplete_capsule],
        seed_node_ids=["seed"],
        query_terms={"answer"},
        seed_graph=source_graph,
    )

    assert any("Answer" in path.text for path in paths)
    assert searcher.last_search_mode == "direct_exact_seed"


def test_path_search_expands_exact_seed_to_three_hops_with_bounded_budget() -> None:
    nodes = [
        GraphNode(id=f"n{index}", properties={"name": f"Node {index}"})
        for index in range(4)
    ]
    edges = [
        GraphEdge(
            id=f"e{index}",
            source_id=f"n{index}",
            target_id=f"n{index + 1}",
            type=f"STEP_{index + 1}",
        )
        for index in range(3)
    ]
    source_graph = LocalGraph.from_nodes_edges(nodes, edges)
    seed_capsule = GraphCapsule(
        id="capsule:seed-only",
        anchor_id="n0",
        region_id="region:seed-only",
        node_ids={"n0"},
        edge_ids=set(),
        nodes=[nodes[0]],
        edges=[],
    )

    paths = BoundedLocalPathSearcher(
        max_candidate_paths=20,
        max_path_length=3,
        final_paths=20,
        max_seed_neighbors=5,
        seed_expansion_depth=3,
        max_seed_expansion_nodes=4,
    ).search(
        capsules=[seed_capsule],
        seed_node_ids=["n0"],
        query_terms={"step", "node"},
        seed_graph=source_graph,
    )

    assert any(len(path.edges) == 3 and path.nodes[-1].id == "n3" for path in paths)


def test_path_search_prefers_repeated_relation_for_co_director_question() -> None:
    nodes = [
        GraphNode(id="director", properties={"name": "Director"}),
        GraphNode(id="film", properties={"name": "Shared Film"}),
        GraphNode(id="answer", properties={"name": "Co Director"}),
        GraphNode(id="decoy", properties={"name": "Actor"}),
    ]
    edges = [
        GraphEdge(id="e1", source_id="director", target_id="film", type="DIRECTED_BY"),
        GraphEdge(id="e2", source_id="film", target_id="answer", type="DIRECTED_BY"),
        GraphEdge(id="e3", source_id="film", target_id="decoy", type="STARRED_ACTORS"),
    ]
    source_graph = LocalGraph.from_nodes_edges(nodes, edges)
    seed_capsule = GraphCapsule(
        id="capsule:director",
        anchor_id="director",
        region_id="region:director",
        node_ids={"director"},
        edge_ids=set(),
        nodes=[nodes[0]],
        edges=[],
    )

    paths = BoundedLocalPathSearcher(
        max_candidate_paths=8,
        max_path_length=2,
        final_paths=1,
        seed_expansion_depth=2,
        max_seed_expansion_nodes=4,
    ).search(
        capsules=[seed_capsule],
        seed_node_ids=["director"],
        query_terms={"who", "are", "the", "co", "directors"},
        seed_graph=source_graph,
    )

    assert paths[0].nodes[-1].id == "answer"


def test_context_ranks_relevant_source_text_instead_of_storage_order() -> None:
    capsule = GraphCapsule(
        id="capsule:sources",
        anchor_id="query",
        region_id="region:sources",
        node_ids={"old", "relevant"},
        edge_ids=set(),
        nodes=[
            GraphNode(id="old", labels=("Chunk",), properties={"text": "unrelated cooking notes"}),
            GraphNode(
                id="relevant",
                labels=("Chunk",),
                properties={
                    "text": "Sam Bankman-Fried faces a crypto fraud trial",
                    "source_document_id": "evidence-url",
                    "source_chunk_id": "chunk-relevant",
                },
            ),
        ],
        edges=[],
    )
    candidate = CapsuleCandidate(capsule=capsule, score=0.5)
    built = OptimizedContextBuilder(
        max_chars=5000,
        snippets_per_capsule=2,
        max_source_snippets=2,
    ).build(query="Who faces the crypto fraud trial?", candidates=[candidate], paths=[])

    assert built.source_snippets[0].id == "chunk-relevant"
    assert "Sam Bankman-Fried" in built.text


def test_evaluator_separates_evidence_answer_and_boolean_metrics() -> None:
    sample = EvaluationSample(
        id="q1",
        question="Who faced trial?",
        answers=["Sam Bankman-Fried"],
        metadata={
            "evidence_list": [
                {"url": "https://a", "title": "First", "fact": "crypto trial"},
                {"url": "https://b", "title": "Second", "fact": "fraud case"},
            ]
        },
    )
    record = RetrievalRecord(
        system="optimized_grag",
        sample_id="q1",
        question=sample.question,
        context_text="Sam Bankman-Fried faced a crypto trial.",
        generated_answer="Sam Bankman-Fried",
        evidence_units=[EvidenceUnit(id="https://a", text="First crypto trial")],
    )

    row = evaluate_record(sample, record)
    assert row["supporting_evidence_recall"] == 0.5
    assert row["evidence_recall_at_1"] == 0.5
    assert row["evidence_mrr"] == 1.0
    assert row["exact_match"] == 1.0
    assert row["token_f1"] == 1.0
    assert row["boolean_accuracy"] is None

    boolean_sample = EvaluationSample(id="q2", question="Both?", answers=["yes"])
    boolean_row = evaluate_record(
        boolean_sample,
        RetrievalRecord(
            system="optimized_grag",
            sample_id="q2",
            question="Both?",
            context_text="Evidence without a literal answer label.",
            generated_answer="Yes, both reports agree.",
        ),
    )
    assert boolean_row["boolean_accuracy"] == 1.0
    assert boolean_row["context_answer_recall"] is None


def test_embedding_and_evaluation_caches_avoid_repeat_work(tmp_path) -> None:
    assert (
        local_embedding_cache_namespace("nomic-embed-text")
        == "local:nomic-embed-text:mean_pooling_v1"
    )
    delegate = CountingEmbedding()
    embedding = CachingEmbeddingAdapter(
        delegate,
        cache_path=tmp_path / "embeddings.sqlite3",
        namespace="test-model-v1",
    )
    first = embedding.embed_texts(["alpha", "beta", "alpha"])
    second = embedding.embed_texts(["beta", "alpha"])
    assert delegate.calls == 1
    assert first.vectors[0] == first.vectors[2] == second.vectors[1]
    assert second.usage["embedding_cache_hits"] == 2

    sample = EvaluationSample(id="cached", question="Question", answers=["Answer"])
    calls = {"count": 0}

    def execute(_sample):
        calls["count"] += 1
        return SimpleNamespace(
            question="Question",
            context_text="Answer appears here.",
            results=[],
            details={},
        )

    runner = CachedEvaluationRunner(
        cache=EvaluationCache(root=tmp_path / "evaluations"),
        max_workers=1,
    )
    runner.run(
        namespace="dataset/system",
        fingerprint="fingerprint",
        system="normal_grag",
        samples=[sample],
        execute=execute,
    )
    _, rows = runner.run(
        namespace="dataset/system",
        fingerprint="fingerprint",
        system="normal_grag",
        samples=[sample],
        execute=execute,
    )
    assert calls["count"] == 1
    assert rows[0]["cache_hit"] is True


def test_evaluation_cache_checkpoints_completed_queries_before_interruption(
    tmp_path,
) -> None:
    samples = [
        EvaluationSample(id="first", question="First?", answers=["one"]),
        EvaluationSample(id="second", question="Second?", answers=["two"]),
    ]
    cache = EvaluationCache(root=tmp_path / "evaluations")
    runner = CachedEvaluationRunner(
        cache=cache,
        max_workers=1,
        checkpoint_interval=1,
    )

    def execute(sample):
        if sample.id == "second":
            raise RuntimeError("interrupted")
        return SimpleNamespace(
            question=sample.question,
            context_text="one",
            results=[],
            details={},
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(
            namespace="checkpoint/system",
            fingerprint="fingerprint",
            system="normal_grag",
            samples=samples,
            execute=execute,
        )

    persisted = cache.load("checkpoint/system", "fingerprint")
    assert "normal_grag:first" in persisted
    assert "normal_grag:second" not in persisted


def test_in_memory_graph_neighborhood_is_deterministic_bfs_order() -> None:
    nodes = [GraphNode(id=value) for value in ("seed", "a", "b", "deep")]
    edges = [
        GraphEdge(id="z-seed-b", source_id="seed", target_id="b", type="LINK"),
        GraphEdge(id="a-seed-a", source_id="seed", target_id="a", type="LINK"),
        GraphEdge(id="m-a-deep", source_id="a", target_id="deep", type="LINK"),
    ]

    observed = []
    for edge_order in (edges, list(reversed(edges))):
        store = InMemoryGraphStore()
        store.upsert_nodes(nodes)
        store.upsert_edges(edge_order)
        neighborhood = store.get_neighbors(["seed"], depth=2, limit=10)
        observed.append(
            (
                [node.id for node in neighborhood.nodes],
                [edge.id for edge in neighborhood.edges],
            )
        )

    assert observed[0] == observed[1]
    assert observed[0][0] == ["seed", "a", "b", "deep"]


def test_local_graph_neighbor_cache_is_invalidated_by_new_edges() -> None:
    graph = LocalGraph.from_nodes_edges(
        [GraphNode(id=value) for value in ("seed", "a", "b")],
        [GraphEdge(id="seed-a", source_id="seed", target_id="a", type="LINK")],
    )
    assert [node_id for node_id, _ in graph.neighbors("seed")] == ["a"]

    graph.add_edge(
        GraphEdge(id="seed-b", source_id="seed", target_id="b", type="LINK")
    )

    assert [node_id for node_id, _ in graph.neighbors("seed")] == ["a", "b"]


def test_projection_supervision_retains_all_exact_answer_capsules() -> None:
    sample = EvaluationSample(
        id="projection",
        question="Who starred with Amy Adams?",
        answers=["Philip Seymour Hoffman"],
        metadata={"hop": "2-hop"},
    )
    search_texts = {
        "positive-a": "Entities: Amy Adams, Philip Seymour Hoffman, The Master.",
        "positive-b": "Entities: Philip Seymour Hoffman, Capote.",
        "token-decoy": "Entities: Philip Glass, Seymour Cassel, Dustin Hoffman.",
        "query-negative": "Entities: Amy Adams, Arrival.",
    }

    examples, report = mine_projection_examples(
        [sample], search_texts, max_positives=None, max_negatives=2
    )

    assert len(examples) == 1
    assert set(examples[0].positive_capsule_ids) == {"positive-a", "positive-b"}
    assert "token-decoy" not in examples[0].positive_capsule_ids
    assert examples[0].slice_name == "2-hop"
    assert report["mean_all_positive_capsules"] == 2.0


def test_any_positive_metrics_use_best_gold_capsule_and_random_context() -> None:
    metrics = any_positive_ranking_metrics(
        ranks=[1, 7],
        positive_counts=[2, 1],
        total_capsules=100,
        ks=(1, 10),
    )

    assert metrics["any_positive_mrr"] == pytest.approx((1.0 + 1.0 / 7.0) / 2.0)
    assert metrics["any_positive_recall_at_1"] == 0.5
    assert metrics["any_positive_recall_at_10"] == 1.0
    assert 0.0 < metrics["random_any_positive_recall_at_1"] < 0.1
    assert metrics["random_any_positive_recall_at_10"] < 0.2


def test_metaqa_projection_labels_gold_path_launch_capsules() -> None:
    sample = EvaluationSample(
        id="path-label",
        question="who directed films acted by [Seed Movie]",
        answers=["Answer Director"],
        metadata={"hop": "3-hop"},
    )
    examples, report = mine_metaqa_path_projection_examples(
        [sample],
        {
            "right-launch": "Seed Movie starred actors Actor One",
            "wrong-seed": "Seed Movie release year 2001",
            "answer-only": "Answer Director directed films Other Movie",
        },
        {
            "right-launch": ["seed-actor"],
            "wrong-seed": ["seed-year"],
            "answer-only": ["movie-director"],
        },
        [
            ("seed-actor", "seed_movie", "actor_one"),
            ("actor-movie", "actor_one", "other_movie"),
            ("movie-director", "other_movie", "answer_director"),
            ("seed-year", "seed_movie", "2001"),
        ],
        max_positives=None,
        max_negatives=2,
    )

    assert len(examples) == 1
    assert examples[0].positive_capsule_ids == ("right-launch",)
    assert "answer-only" not in examples[0].positive_capsule_ids
    assert report["label_rule"] == "capsule_contains_gold_path_first_edge_or_endpoint_pair"


def _two_node_capsule(capsule_id: str, seed_id: str, answer_id: str) -> GraphCapsule:
    nodes = [
        GraphNode(id=seed_id, properties={"name": seed_id}),
        GraphNode(id=answer_id, properties={"name": answer_id}),
    ]
    edge = GraphEdge(
        id=f"{seed_id}:{answer_id}",
        source_id=seed_id,
        target_id=answer_id,
        type="LINK",
    )
    return GraphCapsule(
        id=f"capsule:{capsule_id}",
        anchor_id=seed_id,
        region_id=f"region:{capsule_id}",
        node_ids={seed_id, answer_id},
        edge_ids={edge.id},
        nodes=nodes,
        edges=[edge],
    )
