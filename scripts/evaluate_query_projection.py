from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_grag_evaluation import DATASETS
from src.adapters.datasets import get_dataset_adapter
from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
    local_embedding_cache_namespace,
)
from src.optimized_grag.models import (
    TrainedStructuralEncoder,
    any_positive_ranking_metrics,
    mine_metaqa_path_projection_examples,
    mine_projection_examples,
)


METAQA_DATASETS = tuple(sorted(name for name in DATASETS if name.startswith("metaqa_")))


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    structural_dir = ROOT / "artifacts" / "rgcn" / str(
        dataset.get("structural_artifact_name") or dataset["artifact_name"]
    )
    projector_dir = ROOT / "artifacts" / "rgcn" / str(dataset["artifact_name"])
    structural_metadata = _read_json(structural_dir / "capsule_rgcn_metadata.json")
    projector_metadata = _read_json(projector_dir / "query_projection_metadata.json")
    manifest_name = str(structural_metadata.get("projection_manifest") or "").split(":", 1)[0]
    manifest_path = structural_dir / (manifest_name or "capsule_rgcn_embeddings.npz")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing capsule projection manifest: {manifest_path}")
    expected = str(structural_metadata.get("capsule_fingerprint") or "")
    actual = str(projector_metadata.get("capsule_fingerprint") or "")
    if not expected or expected != actual:
        raise RuntimeError(
            "The query projector and capsule embeddings do not describe the same capsule build. "
            "Retrain the projector against the selected structural artifact."
        )

    payload = np.load(manifest_path, allow_pickle=True)
    capsule_ids = [str(value) for value in payload["capsule_ids"]]
    structural_embeddings = np.asarray(payload["embeddings"], dtype="float32")
    capsule_search_texts = [str(value) for value in payload["search_texts"]]
    capsule_search_text_by_id = dict(zip(capsule_ids, capsule_search_texts))
    capsule_to_index = {capsule_id: index for index, capsule_id in enumerate(capsule_ids)}

    target_manifest = str(projector_metadata.get("target_manifest") or "")
    default_semantic_path = projector_dir / "capsule_semantic_embeddings_nomic_embed_text.npz"
    target_path = projector_dir / target_manifest if target_manifest else default_semantic_path
    if args.scoring_mode == "raw_semantic" or target_manifest:
        if not target_path.exists():
            raise FileNotFoundError(
                f"Missing capsule semantic target artifact: {target_path}. "
                "Run scripts/build_capsule_semantic_artifact.py first."
            )
        target_payload = np.load(target_path, allow_pickle=True)
        target_ids = [str(value) for value in target_payload["capsule_ids"]]
        if target_ids != capsule_ids:
            raise RuntimeError("Semantic target capsule IDs do not align with structural capsules.")
        capsule_embeddings = np.asarray(target_payload["embeddings"], dtype="float32")
    else:
        capsule_embeddings = structural_embeddings

    bundle = get_dataset_adapter(
        str(dataset["adapter_name"]), **dict(dataset["adapter_kwargs"])
    ).load(limit_documents=1, limit_eval_samples=args.max_questions)
    samples = bundle.eval_samples[: args.max_questions]
    path_fields = {
        "capsule_edge_ids",
        "capsule_node_ids",
        "graph_edge_ids",
        "graph_edge_sources",
        "graph_edge_targets",
    }
    if path_fields.issubset(payload.files):
        capsule_edge_ids_by_id = {
            capsule_id: [str(value) for value in edge_ids]
            for capsule_id, edge_ids in zip(capsule_ids, payload["capsule_edge_ids"])
        }
        capsule_node_ids_by_id = {
            capsule_id: [str(value) for value in node_ids]
            for capsule_id, node_ids in zip(capsule_ids, payload["capsule_node_ids"])
        }
        graph_edges = [
            (str(edge_id), str(source), str(target))
            for edge_id, source, target in zip(
                payload["graph_edge_ids"],
                payload["graph_edge_sources"],
                payload["graph_edge_targets"],
            )
        ]
        examples, mining_report = mine_metaqa_path_projection_examples(
            samples,
            capsule_search_text_by_id,
            capsule_edge_ids_by_id,
            graph_edges,
            capsule_node_ids_by_id=capsule_node_ids_by_id,
            max_examples=args.max_questions,
            max_positives=None,
            max_negatives=20,
        )
    else:
        examples, mining_report = mine_projection_examples(
            samples,
            capsule_search_text_by_id,
            max_examples=args.max_questions,
            max_positives=None,
            max_negatives=20,
        )
    if not examples:
        raise RuntimeError("No any-positive projection labels could be mined.")

    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    output_path = output_dir / f"{args.dataset}{suffix}.json"
    fingerprint = _evaluation_fingerprint(
        args=args,
        samples=samples,
        manifest_path=manifest_path,
        projector_dir=projector_dir,
        target_path=target_path if target_path.exists() else None,
    )
    if output_path.exists() and not args.refresh:
        cached = _read_json(output_path)
        if cached.get("evaluation_fingerprint") == fingerprint:
            print(json.dumps({"cache_hit": True, "report": str(output_path), **cached["metrics"]}, indent=2))
            return

    options = {"num_gpu": 999}
    base_embedding = OllamaEmbeddingAdapter(
        OllamaEmbeddingConfig(
            model=args.embedding_model,
            base_url=args.ollama_url,
            keep_alive="30m",
            options=options,
            timeout_seconds=args.timeout,
            batch_size=args.embedding_batch_size,
        )
    )
    embedding_adapter = CachingEmbeddingAdapter(
        base_embedding,
        cache_path=args.embedding_cache,
        namespace=local_embedding_cache_namespace(args.embedding_model),
        model_name=args.embedding_model,
    )
    query_vectors = np.asarray(
        embedding_adapter.embed_texts([example.question for example in examples]).vectors,
        dtype="float32",
    )

    positive_indices = [
        [capsule_to_index[capsule_id] for capsule_id in example.positive_capsule_ids]
        for example in examples
    ]
    seed_candidate_indices = _seed_candidate_indices(examples, payload, capsule_ids)
    if args.scoring_mode == "raw_semantic":
        ranks, seed_conditioned_ranks = _raw_semantic_ranks(
            query_vectors=query_vectors,
            capsule_embeddings=capsule_embeddings,
            positive_indices=positive_indices,
            candidate_indices=seed_candidate_indices,
            batch_size=args.score_batch_size,
        )
    else:
        encoder = TrainedStructuralEncoder(
            structural_dir,
            projector_artifact_dir=projector_dir,
        )
        if encoder.projector is None:
            raise RuntimeError(f"Could not load query projector from {projector_dir}")
        ranks, seed_conditioned_ranks = _best_positive_ranks(
            encoder=encoder,
            query_vectors=query_vectors,
            capsule_embeddings=capsule_embeddings,
            positive_indices=positive_indices,
            candidate_indices=seed_candidate_indices,
            batch_size=args.score_batch_size,
            device_name=args.device,
        )
    positive_counts = [len(example.positive_capsule_ids) for example in examples]
    metrics = any_positive_ranking_metrics(
        ranks,
        positive_counts,
        total_capsules=len(capsule_ids),
    )
    seed_metrics = any_positive_ranking_metrics(
        seed_conditioned_ranks,
        positive_counts,
        total_capsules=[len(indices) for indices in seed_candidate_indices],
    )
    metrics.update(
        {
            f"seed_conditioned_{key}": value
            for key, value in seed_metrics.items()
            if key != "evaluated_examples"
        }
    )
    metrics["mean_seed_candidate_capsules"] = float(
        np.mean([len(indices) for indices in seed_candidate_indices])
    )
    metrics["label_coverage"] = len(examples) / max(len(samples), 1)
    metrics["lift_over_random_mrr"] = (
        float(metrics["any_positive_mrr"])
        / max(float(metrics["random_any_positive_mrr"]), 1e-12)
    )
    report = {
        "evaluation_fingerprint": fingerprint,
        "dataset": args.dataset,
        "scoring_mode": args.scoring_mode,
        "split": bundle.metadata.get("split"),
        "questions_loaded": len(samples),
        "capsules": len(capsule_ids),
        "artifact": {
            "structural_dir": str(structural_dir),
            "projector_dir": str(projector_dir),
            "capsule_fingerprint": expected,
            "training_schema_version": projector_metadata.get("training_schema_version"),
            "objective": (
                "raw_semantic_cosine"
                if args.scoring_mode == "raw_semantic"
                else projector_metadata.get("objective", "legacy_single_positive_triplet")
            ),
            "target_manifest": target_path.name if target_path.exists() else None,
            "text_encoder": projector_metadata.get("text_encoder"),
        },
        "label_mining": mining_report,
        "metrics": metrics,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cache_hit": False, "report": str(output_path), **metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate learned capsule projection using any-positive gold capsules."
    )
    parser.add_argument("--dataset", choices=METAQA_DATASETS, required=True)
    parser.add_argument("--max-questions", type=int, default=500)
    parser.add_argument("--tag", help="Optional output suffix, e.g. before_multi_positive.")
    parser.add_argument(
        "--scoring-mode",
        choices=("projector", "raw_semantic"),
        default="projector",
        help="Evaluate the learned projector or direct query/capsule semantic cosine.",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--output-root", default="reports/projection_evaluation")
    return parser.parse_args()


def _best_positive_ranks(
    *,
    encoder: TrainedStructuralEncoder,
    query_vectors: np.ndarray,
    capsule_embeddings: np.ndarray,
    positive_indices: list[list[int]],
    candidate_indices: list[list[int]],
    batch_size: int,
    device_name: str,
) -> tuple[list[int], list[int]]:
    import torch

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else device_name
    )
    if device_name == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model, input_dim = encoder.projector
    model = model.to(device)
    model.eval()
    capsules = torch.tensor(capsule_embeddings, dtype=torch.float32, device=device)
    capsules = torch.nn.functional.normalize(capsules, dim=1)
    ranks: list[int] = []
    conditioned_ranks: list[int] = []
    with torch.no_grad():
        for start in range(0, len(query_vectors), batch_size):
            stop = min(start + batch_size, len(query_vectors))
            values = query_vectors[start:stop]
            if values.shape[1] < input_dim:
                values = np.pad(values, ((0, 0), (0, input_dim - values.shape[1])))
            values = values[:, :input_dim]
            projected = model(torch.tensor(values, dtype=torch.float32, device=device))
            scores = projected @ capsules.T
            for row, indices in enumerate(positive_indices[start:stop]):
                positive = torch.tensor(indices, dtype=torch.long, device=device)
                best_positive_score = scores[row, positive].max()
                rank = 1 + int((scores[row] > best_positive_score).sum().item())
                ranks.append(rank)
                candidates = torch.tensor(
                    candidate_indices[start + row], dtype=torch.long, device=device
                )
                conditioned_ranks.append(
                    1 + int((scores[row, candidates] > best_positive_score).sum().item())
                )
    return ranks, conditioned_ranks


def _raw_semantic_ranks(
    *,
    query_vectors: np.ndarray,
    capsule_embeddings: np.ndarray,
    positive_indices: list[list[int]],
    candidate_indices: list[list[int]],
    batch_size: int,
) -> tuple[list[int], list[int]]:
    queries = np.asarray(query_vectors, dtype="float32")
    capsules = np.asarray(capsule_embeddings, dtype="float32")
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    query_norms[query_norms == 0.0] = 1.0
    capsule_norms = np.linalg.norm(capsules, axis=1, keepdims=True)
    capsule_norms[capsule_norms == 0.0] = 1.0
    queries = queries / query_norms
    capsules = capsules / capsule_norms
    ranks: list[int] = []
    conditioned_ranks: list[int] = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        scores = queries[start:stop] @ capsules.T
        for row, indices in enumerate(positive_indices[start:stop]):
            best_positive = float(scores[row, indices].max())
            ranks.append(1 + int((scores[row] > best_positive).sum()))
            candidates = candidate_indices[start + row]
            conditioned_ranks.append(
                1 + int((scores[row, candidates] > best_positive).sum())
            )
    return ranks, conditioned_ranks


def _seed_candidate_indices(examples, payload, capsule_ids: list[str]) -> list[list[int]]:
    if "capsule_node_ids" not in payload.files:
        return [list(range(len(capsule_ids))) for _ in examples]
    node_to_positions: dict[str, list[int]] = {}
    for position, node_ids in enumerate(payload["capsule_node_ids"]):
        for node_id in node_ids:
            node_to_positions.setdefault(str(node_id), []).append(position)
    result: list[list[int]] = []
    for example in examples:
        match = re.search(r"\[([^\]]+)\]", example.question)
        seed_id = (
            re.sub(r"[^a-zA-Z0-9]+", "_", match.group(1).strip().lower()).strip("_")
            if match
            else ""
        )
        positions = node_to_positions.get(seed_id) or list(range(len(capsule_ids)))
        result.append(positions)
    return result


def _evaluation_fingerprint(
    *,
    args,
    samples,
    manifest_path: Path,
    projector_dir: Path,
    target_path: Path | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update((ROOT / "src" / "optimized_grag" / "models" / "projection_supervision.py").read_bytes())
    digest.update(manifest_path.read_bytes())
    for filename in ("query_projection_mlp.pt", "query_projection_metadata.json"):
        digest.update((projector_dir / filename).read_bytes())
    if target_path is not None:
        digest.update(target_path.read_bytes())
    digest.update(args.embedding_model.encode())
    digest.update(args.scoring_mode.encode())
    for sample in samples:
        digest.update(str(sample.id).encode())
        digest.update(str(sample.question).encode())
        digest.update(json.dumps(list(sample.answers), sort_keys=True).encode())
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
