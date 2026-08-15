"""Inspect M4 embedding-cache coverage without constructing a registry or calling Ollama."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.datasets import get_dataset_adapter
from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
    local_embedding_cache_namespace,
)
from src.helpers import TextChunker, normalize_documents
from src.rag.types import RAGConfig


def main() -> None:
    args = parse_args()
    bundle = _load_metaqa_bundle(args)
    rag_config = RAGConfig(chunk_size=900, chunk_overlap=120, index_batch_size=args.index_batch_size)
    chunks = TextChunker(chunk_size=rag_config.chunk_size, overlap=rag_config.chunk_overlap).chunk_documents(
        normalize_documents(bundle.documents)
    )
    # The wrapped adapter is never called by this script. It is supplied only
    # because CachingEmbeddingAdapter owns the canonical cache schema/API.
    cache = CachingEmbeddingAdapter(
        OllamaEmbeddingAdapter(OllamaEmbeddingConfig(model=args.embedding_model)),
        cache_path=args.embedding_cache,
        namespace=local_embedding_cache_namespace(args.embedding_model),
        model_name=args.embedding_model,
    )
    coverage = cache.inspect_texts([chunk.text for chunk in chunks])
    missing = int(coverage["missing_unique_texts"])
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "documents": len(bundle.documents),
        "normal_rag_chunks": len(chunks),
        "cache": cache.cache_summary(),
        "coverage": coverage,
        "collector_configuration": {
            "embedding_model": args.embedding_model,
            "embedding_batch_size": args.embedding_batch_size,
            "normal_index_batch_size": args.index_batch_size,
        },
        "estimated_work_if_registry_starts_now": {
            "new_vectors": missing,
            "normal_index_batches": math.ceil(len(chunks) / args.index_batch_size) if chunks else 0,
            "ollama_embed_requests_at_configured_batch_size": math.ceil(missing / args.embedding_batch_size) if missing else 0,
        },
        "interpretation": _interpret(coverage),
        "network_calls_made": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), **report}, indent=2))


def _load_metaqa_bundle(args):
    hop = args.dataset.removeprefix("metaqa_").removesuffix("hop")
    adapter = get_dataset_adapter("metaqa", hop=f"{hop}-hop", variant="vanilla", split=args.split)
    return adapter.load(limit_documents=None, limit_eval_samples=0)


def _interpret(coverage: dict[str, int | float]) -> str:
    hit_rate = float(coverage["cache_hit_rate"])
    if hit_rate >= 0.99:
        return "Cache coverage is sufficient; registry delay is likely graph/capsule construction or in-memory index rebuilding."
    return (
        "Cache coverage is incomplete. HybridRAG indexes first, so its NormalRAG embedding stage will create the "
        "reported missing vectors before GraphRAG can reuse them from the shared content-addressed cache."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure full-corpus embedding-cache coverage without any embedding requests.")
    parser.add_argument("--dataset", choices=("metaqa_1hop", "metaqa_2hop", "metaqa_3hop"), default="metaqa_2hop")
    parser.add_argument("--split", default="train")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--index-batch-size", type=int, default=512)
    parser.add_argument("--output", default="reports/router/data_audit/metaqa_2hop_m4_embedding_cache.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
