from __future__ import annotations

import pytest

from src.adapters.embedding import EmbeddingAdapter, EmbeddingResult
from src.adapters.storage.graph_store import GraphNode
from src.optimized_grag.partitioning.supergraph import CapsuleSupergraph
from src.optimized_grag.partitioning.types import GraphCapsule
from src.optimized_grag.retrieval.capsule_index import CapsuleIndex


class _NeverEmbedding(EmbeddingAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        raise AssertionError("trained capsule retrieval must not re-embed summaries")


class _TrainedEncoder:
    trained_space_enabled = True

    def encode_capsule(self, capsule):
        return [1.0, 0.0] if capsule.id == "capsule:semantic" else [0.0, 1.0]

    def project_query(self, query_vector):
        return list(query_vector)


def test_trained_capsule_index_skips_summary_embeddings_and_preserves_seed() -> None:
    semantic = _capsule("capsule:semantic", "semantic_node")
    seed = _capsule("capsule:seed", "linked_seed")
    embedding = _NeverEmbedding()
    index = CapsuleIndex(structural_encoder=_TrainedEncoder())

    index.build(
        capsules=[semantic, seed],
        embedding_adapter=embedding,
        supergraph=CapsuleSupergraph(),
    )
    selected = index.candidates(
        query="semantic question",
        query_vector=[1.0, 0.0],
        query_entities=[],
        seed_node_ids=["linked_seed"],
        limit=1,
    )

    assert embedding.calls == 0
    assert index.retrieval_mode == "trained_structural"
    assert selected[0].capsule.id == "capsule:seed"
    assert selected[0].metadata["matched_seed_ids"] == ["linked_seed"]


def test_relation_grouped_rgcn_matches_legacy_per_edge_math() -> None:
    torch = pytest.importorskip("torch")
    from src.optimized_grag.models.rgcn import RGCNLayer

    torch.manual_seed(7)
    layer = RGCNLayer(5, 4, dropout=0.0, message_batch_size=2)
    layer.eval()
    x = torch.randn(8, 5, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 0, 3, 2, 1, 3], dtype=torch.long)

    src, dst = edge_index
    legacy_messages = torch.bmm(
        x[src].unsqueeze(1), layer.rel_weights[edge_type]
    ).squeeze(1)
    expected = torch.zeros_like(x)
    expected.index_add_(0, dst, legacy_messages)
    degree = torch.bincount(dst, minlength=x.size(0)).to(dtype=x.dtype)
    expected = torch.relu(
        expected / degree.clamp_min(1).unsqueeze(1) + layer.self_loop(x)
    )

    actual = layer(x, edge_index, edge_type)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert layer.rel_weights.grad is not None


def test_metaqa_evaluation_registry_has_honest_test_targets_for_every_hop() -> None:
    from scripts.run_grag_evaluation import DATASETS, _retrieval_configs

    targets = [DATASETS[f"metaqa_{hop}hop"] for hop in (1, 2, 3)]
    assert [target["adapter_kwargs"]["hop"] for target in targets] == [
        "1-hop",
        "2-hop",
        "3-hop",
    ]
    assert all(target["adapter_kwargs"]["split"] == "test" for target in targets)
    assert len({target["report_name"] for target in targets}) == 3
    assert {
        target["artifact_name"] for target in targets
    } == {"metaqa_unified_vanilla_train"}
    assert {
        target["structural_artifact_name"] for target in targets
    } == {"metaqa_shared_graph_train"}
    assert len(
        {
            repr(target["optimized_config"])
            for target in targets
        }
    ) == 1

    configs = _retrieval_configs(targets[0], 2)
    assert configs["normal_grag"].graph_depth == 2
    assert configs["optimized_grag"].max_path_length == 2
    assert configs["optimized_grag"].seed_expansion_depth == 2
    # Offline capsule radius is an artifact property, not the query-time cap.
    assert configs["optimized_grag"].max_hops == targets[0]["optimized_config"].max_hops

    with pytest.raises(ValueError, match="at least 1"):
        _retrieval_configs(targets[0], 0)

    for dataset in DATASETS.values():
        default_cap = dataset["retrieval_max_hops"]
        resolved = _retrieval_configs(dataset, default_cap)
        assert resolved["normal_grag"].graph_depth == default_cap
        assert resolved["optimized_grag"].max_path_length == default_cap
        assert resolved["optimized_grag"].seed_expansion_depth == default_cap


def test_structural_encoder_can_load_hop_projector_from_separate_directory(
    tmp_path,
) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from src.optimized_grag.models.structural import TrainedStructuralEncoder

    structural_dir = tmp_path / "shared_structural"
    projector_dir = tmp_path / "hop_projector"
    structural_dir.mkdir()
    projector_dir.mkdir()
    np.savez_compressed(
        structural_dir / "capsules_v3.npz",
        capsule_ids=np.asarray(["capsule:a"]),
        embeddings=np.asarray([[1.0, 0.0]], dtype="float32"),
    )
    (structural_dir / "capsule_rgcn_metadata.json").write_text(
        '{"projection_manifest":"capsules_v3.npz:search_texts",'
        '"capsule_fingerprint":"shared-fingerprint"}',
        encoding="utf-8",
    )
    (projector_dir / "query_projection_metadata.json").write_text(
        '{"input_dim":2,"output_dim":2,'
        '"capsule_fingerprint":"shared-fingerprint"}',
        encoding="utf-8",
    )

    class ProjectionMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(256, 2),
            )

    torch.save(
        ProjectionMLP().state_dict(),
        projector_dir / "query_projection_mlp.pt",
    )
    encoder = TrainedStructuralEncoder(
        structural_dir,
        projector_artifact_dir=projector_dir,
    )

    assert encoder.is_available
    assert encoder.projector is not None
    assert encoder.artifact_fingerprint == encoder.projector_fingerprint


def test_semantic_residual_projector_keeps_rgcn_as_separate_feature(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    nn = pytest.importorskip("torch.nn")
    from src.optimized_grag.models.structural import (
        TrainedStructuralEncoder,
        capsule_fingerprint,
    )

    capsule = _capsule("capsule:a", "node:a")
    fingerprint = capsule_fingerprint([capsule])
    structural_dir = tmp_path / "structural"
    projector_dir = tmp_path / "projector"
    structural_dir.mkdir()
    projector_dir.mkdir()
    np.savez_compressed(
        structural_dir / "capsules.npz",
        capsule_ids=np.asarray([capsule.id]),
        embeddings=np.asarray([[1.0, 0.0]], dtype="float32"),
    )
    np.savez_compressed(
        projector_dir / "semantic.npz",
        capsule_ids=np.asarray([capsule.id]),
        embeddings=np.asarray([[0.0, 1.0]], dtype="float32"),
    )
    (structural_dir / "capsule_rgcn_metadata.json").write_text(
        '{"projection_manifest":"capsules.npz:search_texts",'
        f'"capsule_fingerprint":"{fingerprint}"}}',
        encoding="utf-8",
    )
    (projector_dir / "query_projection_metadata.json").write_text(
        '{"input_dim":2,"output_dim":2,"architecture":"identity_residual_mlp",'
        '"target_space":"semantic_capsule_text","target_manifest":"semantic.npz",'
        f'"capsule_fingerprint":"{fingerprint}"}}',
        encoding="utf-8",
    )

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Identity()
            self.residual = nn.Sequential(
                nn.Linear(2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(256, 2),
            )

    torch.save(ResidualMLP().state_dict(), projector_dir / "query_projection_mlp.pt")
    encoder = TrainedStructuralEncoder(
        structural_dir, projector_artifact_dir=projector_dir
    )
    assert encoder.bind_capsules([capsule]) == 1.0
    assert encoder.trained_space_enabled
    assert encoder.encode_capsule(capsule) == [1.0, 0.0]
    assert encoder.encode_routing_capsule(capsule) == [0.0, 1.0]
    assert encoder.project_structural_query([0.0, 1.0]) == []

    index = CapsuleIndex(structural_encoder=encoder)
    index.build(
        capsules=[capsule],
        embedding_adapter=_NeverEmbedding(),
        supergraph=CapsuleSupergraph(),
    )
    assert index.retrieval_mode == "trained_semantic_projection"


def _capsule(capsule_id: str, node_id: str) -> GraphCapsule:
    node = GraphNode(id=node_id, properties={"name": node_id})
    return GraphCapsule(
        id=capsule_id,
        anchor_id=node_id,
        region_id=f"region:{node_id}",
        node_ids={node_id},
        edge_ids=set(),
        nodes=[node],
        edges=[],
    )
