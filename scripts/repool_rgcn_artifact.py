from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.datasets import get_dataset_adapter
from src.compute_eval import InMemoryComputeTracker
from src.optimized_grag.graph_construction import build_training_graph
from src.optimized_grag.models import capsule_fingerprint
from src.optimized_grag.models.rgcn import (
    CapsuleRGCN,
    build_message_tensors,
    split_edges,
)
from src.optimized_grag.partitioning import CapsuleConstructionPipeline
from src.optimized_grag.types import OptimizedGraphRAGConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repool trained R-GCN node embeddings onto deterministic capsules."
    )
    parser.add_argument(
        "--artifact-name",
        required=True,
        help="Folder name under artifacts/rgcn, e.g. metaqa_shared_graph_train.",
    )
    args = parser.parse_args()
    artifact_dir = ROOT / "artifacts" / "rgcn" / args.artifact_name
    metadata_path = artifact_dir / "capsule_rgcn_metadata.json"
    weights_path = artifact_dir / "capsule_rgcn.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing R-GCN artifact files under {artifact_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = dict(metadata.get("dataset") or {})
    seed = int(metadata.get("seed") or 42)
    adapter_kwargs = dict(dataset.get("adapter_kwargs") or {})
    if dataset.get("dataset_root"):
        adapter_kwargs["root"] = ROOT / str(dataset["dataset_root"])
    bundle = get_dataset_adapter(
        str(dataset["adapter_name"]),
        **adapter_kwargs,
    ).load(
        limit_documents=dataset.get("limit_documents"),
        limit_eval_samples=dataset.get("limit_eval_samples"),
    )
    graph, graph_report = build_training_graph(bundle, dataset)
    config = OptimizedGraphRAGConfig(**dict(dataset.get("capsule_config") or {}))
    capsule_result = CapsuleConstructionPipeline(
        config=config,
        instrumentation=InMemoryComputeTracker(log_stages=False),
    ).build(graph)

    node_ids = sorted(graph.nodes)
    node_to_idx = {node_id: index for index, node_id in enumerate(node_ids)}
    relation_types = sorted({edge.type for edge in graph.edges.values()})
    relation_to_idx = {relation: index for index, relation in enumerate(relation_types)}
    edges = [
        (
            node_to_idx[edge.source_id],
            node_to_idx[edge.target_id],
            relation_to_idx[edge.type],
        )
        for edge in graph.edges.values()
        if edge.source_id in node_to_idx and edge.target_id in node_to_idx
    ]
    train_edges, _ = split_edges(edges, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    message_index, message_type = build_message_tensors(
        train_edges,
        num_relations=len(relation_types),
        device=device,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    dimension = int(state["node_emb.weight"].shape[1])
    layer_count = len(
        {key.split(".")[1] for key in state if key.startswith("layers.")}
    )
    model = CapsuleRGCN(
        len(node_ids),
        len(relation_types),
        dim=dimension,
        layers=layer_count,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode(message_index, message_type).cpu().numpy()

    capsule_ids: list[str] = []
    capsule_vectors: list[np.ndarray] = []
    search_texts: list[str] = []
    capsule_node_ids: list[list[str]] = []
    capsule_edge_ids: list[list[str]] = []
    for capsule in capsule_result.capsules:
        indices = [node_to_idx[node_id] for node_id in capsule.node_ids]
        vector = node_embeddings[indices].mean(axis=0)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        capsule_ids.append(capsule.id)
        capsule_vectors.append(vector)
        search_texts.append(capsule.summary)
        capsule_node_ids.append(sorted(capsule.node_ids))
        capsule_edge_ids.append(sorted(capsule.edge_ids))

    ordered_graph_edges = sorted(graph.edges.values(), key=lambda edge: edge.id)

    output_path = artifact_dir / "capsule_rgcn_embeddings_v4.npz"
    temporary_path = artifact_dir / "capsule_rgcn_embeddings_v4.tmp.npz"
    np.savez_compressed(
        temporary_path,
        capsule_ids=np.asarray(capsule_ids),
        embeddings=np.vstack(capsule_vectors),
        search_texts=np.asarray(search_texts, dtype=object),
        capsule_node_ids=np.asarray(capsule_node_ids, dtype=object),
        capsule_edge_ids=np.asarray(capsule_edge_ids, dtype=object),
        graph_edge_ids=np.asarray([edge.id for edge in ordered_graph_edges], dtype=object),
        graph_edge_sources=np.asarray([edge.source_id for edge in ordered_graph_edges], dtype=object),
        graph_edge_targets=np.asarray([edge.target_id for edge in ordered_graph_edges], dtype=object),
        graph_edge_types=np.asarray([edge.type for edge in ordered_graph_edges], dtype=object),
    )
    temporary_path.replace(output_path)

    metadata.update(
        {
            "graph_report": graph_report,
            "capsule_metadata": capsule_result.metadata,
            "node_ids": node_ids,
            "relation_types": relation_types,
            "capsule_fingerprint": capsule_fingerprint(capsule_result.capsules),
            "training_schema_version": max(int(metadata.get("training_schema_version") or 0), 5),
            "projection_manifest": "capsule_rgcn_embeddings_v4.npz:search_texts+membership+graph_edges",
            "seed": seed,
        }
    )
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    print(
        json.dumps(
            {
                "artifact": str(artifact_dir),
                "capsules": len(capsule_ids),
                "capsule_fingerprint": metadata["capsule_fingerprint"],
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
