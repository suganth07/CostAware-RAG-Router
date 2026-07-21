from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
    local_embedding_cache_namespace,
)


def main() -> None:
    args = parse_args()
    structural_dir = ROOT / "artifacts" / "rgcn" / args.structural_artifact
    projector_dir = ROOT / "artifacts" / "rgcn" / args.projector_artifact
    metadata_path = structural_dir / "capsule_rgcn_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_name = str(metadata["projection_manifest"]).split(":", 1)[0]
    structural_path = structural_dir / manifest_name
    structural = np.load(structural_path, allow_pickle=True)
    capsule_ids = [str(value) for value in structural["capsule_ids"]]
    search_texts = [str(value) for value in structural["search_texts"]]
    capsule_fingerprint = str(metadata["capsule_fingerprint"])

    safe_model = "".join(character if character.isalnum() else "_" for character in args.embedding_model)
    output_path = projector_dir / f"capsule_semantic_embeddings_{safe_model}.npz"
    build_fingerprint = _fingerprint(
        capsule_fingerprint,
        args.embedding_model,
        capsule_ids,
        search_texts,
    )
    if output_path.exists() and not args.refresh:
        current = np.load(output_path, allow_pickle=True)
        stored = str(current["build_fingerprint"].item()) if "build_fingerprint" in current.files else ""
        if stored == build_fingerprint:
            print(json.dumps({"cache_hit": True, "artifact": str(output_path), "capsules": len(capsule_ids)}, indent=2))
            return

    options = {"num_gpu": 999}
    delegate = OllamaEmbeddingAdapter(
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
        delegate,
        cache_path=args.embedding_cache,
        namespace=local_embedding_cache_namespace(args.embedding_model),
        model_name=args.embedding_model,
    )

    vectors: list[list[float]] = []
    for start in range(0, len(search_texts), args.checkpoint_batch_size):
        stop = min(start + args.checkpoint_batch_size, len(search_texts))
        result = embedding_adapter.embed_texts(search_texts[start:stop])
        vectors.extend(result.vectors)
        print(
            f"embedded capsules={stop}/{len(search_texts)} "
            f"cache_hits={result.usage.get('embedding_cache_hits', 0)} "
            f"cache_misses={result.usage.get('embedding_cache_misses', 0)}",
            flush=True,
        )

    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix = matrix / norms
    projector_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        capsule_ids=np.asarray(capsule_ids),
        embeddings=matrix,
        search_texts=np.asarray(search_texts, dtype=object),
        capsule_fingerprint=np.asarray(capsule_fingerprint),
        embedding_model=np.asarray(args.embedding_model),
        build_fingerprint=np.asarray(build_fingerprint),
    )
    temporary_path.replace(output_path)
    print(
        json.dumps(
            {
                "cache_hit": False,
                "artifact": str(output_path),
                "capsules": len(capsule_ids),
                "dimension": int(matrix.shape[1]),
                "build_fingerprint": build_fingerprint,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached semantic capsule targets for query projection."
    )
    parser.add_argument("--structural-artifact", default="metaqa_shared_graph_train")
    parser.add_argument("--projector-artifact", default="metaqa_unified_vanilla_train")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-batch-size", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def _fingerprint(
    capsule_fingerprint: str,
    model: str,
    capsule_ids: list[str],
    search_texts: list[str],
) -> str:
    digest = hashlib.sha256()
    digest.update(capsule_fingerprint.encode())
    digest.update(model.encode())
    for capsule_id, text in zip(capsule_ids, search_texts):
        digest.update(capsule_id.encode())
        digest.update(text.encode(errors="replace"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
