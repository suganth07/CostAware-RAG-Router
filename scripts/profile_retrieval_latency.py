from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grag_evaluation import (
    DATASETS,
    _build_pipeline,
    _index_pipeline,
    _retrieval_configs,
)
from src.adapters.datasets import get_dataset_adapter
from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    LlamaCppEmbeddingAdapter,
    LlamaCppEmbeddingConfig,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
    local_embedding_cache_namespace,
)
from src.adapters.llm import OllamaLLM, OllamaLLMConfig
from src.evaluation import fingerprint_bundle, fingerprint_run


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    bundle = get_dataset_adapter(
        str(dataset["adapter_name"]), **dict(dataset["adapter_kwargs"])
    ).load(
        limit_documents=dataset["limit_documents"],
        limit_eval_samples=args.questions,
    )
    samples = bundle.eval_samples[: args.questions]
    options = {"num_gpu": 999}
    if args.embedding_backend == "llama_cpp":
        base_embedding = LlamaCppEmbeddingAdapter(
            LlamaCppEmbeddingConfig(
                model=args.embedding_model,
                base_url=args.llama_cpp_url,
                timeout_seconds=180.0,
                batch_size=min(max(args.embedding_batch_size, 16), 256),
                max_concurrent_requests=args.embedding_request_workers,
            )
        )
        embedding_namespace = local_embedding_cache_namespace(args.embedding_model)
    else:
        base_embedding = OllamaEmbeddingAdapter(
            OllamaEmbeddingConfig(
                model=args.embedding_model,
                base_url=args.ollama_url,
                keep_alive="30m",
                options=options,
                timeout_seconds=180.0,
                batch_size=args.embedding_batch_size,
            )
        )
        embedding_namespace = local_embedding_cache_namespace(args.embedding_model)
    embedding = CachingEmbeddingAdapter(
        base_embedding,
        cache_path=args.embedding_cache,
        namespace=embedding_namespace,
        model_name=args.embedding_model,
    )
    llm = OllamaLLM(OllamaLLMConfig(model="qwen3:4b-instruct"))
    max_hops = args.max_hops or int(dataset["retrieval_max_hops"])
    configs = _retrieval_configs(dataset, max_hops)
    if args.candidate_paths is not None or args.final_paths is not None:
        optimized = configs["optimized_grag"]
        configs["optimized_grag"] = replace(
            optimized,
            max_candidate_paths=(
                args.candidate_paths
                if args.candidate_paths is not None
                else optimized.max_candidate_paths
            ),
            final_paths_to_llm=(
                args.final_paths
                if args.final_paths is not None
                else optimized.final_paths_to_llm
            ),
        )

    corpus_hash = fingerprint_bundle(bundle, [])
    normal_config = configs["normal_grag"]
    vector_index_key = fingerprint_run(
        {
            "corpus": corpus_hash,
            "embedding_namespace": embedding.namespace,
            "chunk_size": normal_config.chunk_size,
            "chunk_overlap": normal_config.chunk_overlap,
            "store": "numpy_exact_cosine_v1",
        }
    )
    vector_index_cache = Path(args.vector_index_cache) / vector_index_key
    embedding.embed_texts([sample.question for sample in samples])

    systems = (
        ["normal_grag", "optimized_grag"]
        if args.system == "both"
        else [args.system]
    )
    reports = {}
    for system in systems:
        pipeline = _build_pipeline(
            system=system,
            dataset=dataset,
            config=configs[system],
            embedding_adapter=embedding,
            llm=llm,
            vector_index_cache=(
                vector_index_cache if system == "normal_grag" else None
            ),
        )
        index_started = time.perf_counter()
        index_report = _index_pipeline(pipeline, bundle, dataset)
        index_seconds = time.perf_counter() - index_started
        if samples:
            pipeline.retrieve(samples[0].question)

        tracker = pipeline.instrumentation
        rows: list[dict[str, float]] = []
        for sample in samples:
            tracker.metrics.clear()
            started = time.perf_counter()
            pipeline.retrieve(sample.question)
            row = {"total": (time.perf_counter() - started) * 1000}
            for metric in tracker.metrics:
                stage = metric.stage.split(".")[-1]
                row[stage] = row.get(stage, 0.0) + metric.elapsed_ms
            rows.append(row)
        stages = sorted({key for row in rows for key in row})
        reports[system] = {
            "index_seconds": index_seconds,
            "index_report": _jsonable(index_report),
            "config": asdict(configs[system]),
            "stage_ms": {
                stage: _summary([row.get(stage, 0.0) for row in rows])
                for stage in stages
            },
        }

    report = {
        "dataset": args.dataset,
        "questions": len(samples),
        "warm_query_cache": True,
        "max_hops": max_hops,
        "normal_vector_index": str(vector_index_cache),
        "systems": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile normal and optimized GraphRAG under their full retrieval contracts."
    )
    parser.add_argument(
        "--dataset",
        choices=("metaqa_1hop", "metaqa_2hop", "metaqa_3hop"),
        default="metaqa_3hop",
    )
    parser.add_argument(
        "--system",
        choices=("both", "normal_grag", "optimized_grag"),
        default="both",
    )
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--max-hops", type=int)
    parser.add_argument("--candidate-paths", type=int)
    parser.add_argument("--final-paths", type=int)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument(
        "--embedding-backend",
        choices=("ollama", "llama_cpp"),
        default="ollama",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--llama-cpp-url", default="http://127.0.0.1:11435")
    parser.add_argument("--embedding-request-workers", type=int, default=8)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--vector-index-cache", default="artifacts/cache/vector_indexes")
    parser.add_argument(
        "--output",
        default="reports/latency/production_retrieval_comparison.json",
    )
    return parser.parse_args()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": max(values, default=0.0),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(values[int(round((len(values) - 1) * fraction))])


def _jsonable(value):
    return {
        name: getattr(value, name)
        for name in getattr(value, "__dataclass_fields__", {})
    }


if __name__ == "__main__":
    main()
