from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_grag_evaluation import DATASETS, _build_pipeline, _index_pipeline
from src.adapters.datasets import get_dataset_adapter
from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
    local_embedding_cache_namespace,
)
from src.adapters.llm import OllamaLLM, OllamaLLMConfig


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    bundle = get_dataset_adapter(
        str(dataset["adapter_name"]), **dict(dataset["adapter_kwargs"])
    ).load(limit_documents=dataset["limit_documents"], limit_eval_samples=args.questions)
    samples = bundle.eval_samples[: args.questions]

    options = {"num_gpu": 999}
    embedding = CachingEmbeddingAdapter(
        OllamaEmbeddingAdapter(
            OllamaEmbeddingConfig(
                model=args.embedding_model,
                base_url=args.ollama_url,
                keep_alive="30m",
                options=options,
                timeout_seconds=180.0,
                batch_size=args.embedding_batch_size,
            )
        ),
        cache_path=args.embedding_cache,
        namespace=local_embedding_cache_namespace(args.embedding_model),
        model_name=args.embedding_model,
    )
    llm = OllamaLLM(OllamaLLMConfig(model="qwen3:4b-instruct"))
    optimized_config = replace(
        dataset["optimized_config"],
        max_candidate_paths=(
            args.candidate_paths
            if args.candidate_paths is not None
            else dataset["optimized_config"].max_candidate_paths
        ),
        final_paths_to_llm=(
            args.final_paths
            if args.final_paths is not None
            else dataset["optimized_config"].final_paths_to_llm
        ),
    )
    pipeline = _build_pipeline(
        system="optimized_grag",
        dataset=dataset,
        config=optimized_config,
        embedding_adapter=embedding,
        llm=llm,
    )
    embedding.embed_texts([sample.question for sample in samples])
    index_started = time.perf_counter()
    _index_pipeline(pipeline, bundle, dataset)
    index_seconds = time.perf_counter() - index_started

    if samples:
        pipeline.retrieve(samples[0].question)
    tracker = pipeline.instrumentation
    rows: list[dict[str, float]] = []
    for sample in samples:
        tracker.metrics.clear()
        started = time.perf_counter()
        pipeline.retrieve(sample.question)
        total_ms = (time.perf_counter() - started) * 1000
        row = {"total": total_ms}
        for metric in tracker.metrics:
            stage = metric.stage.split(".")[-1]
            row[stage] = row.get(stage, 0.0) + metric.elapsed_ms
        rows.append(row)

    stages = sorted({key for row in rows for key in row})
    summary = {
        stage: _summary([row.get(stage, 0.0) for row in rows]) for stage in stages
    }
    report = {
        "dataset": args.dataset,
        "questions": len(samples),
        "warm_cache": True,
        "index_seconds": index_seconds,
        "retrieval_mode": pipeline.capsule_index.retrieval_mode,
        "config": {
            "candidate_paths": pipeline.path_searcher.max_candidate_paths,
            "final_paths": pipeline.path_searcher.final_paths,
            "seed_expansion_nodes": pipeline.path_searcher.max_seed_expansion_nodes,
        },
        "stage_ms": summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile warm optimized retrieval stages.")
    parser.add_argument(
        "--dataset",
        choices=("metaqa_1hop", "metaqa_2hop", "metaqa_3hop"),
        default="metaqa_3hop",
    )
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--candidate-paths", type=int)
    parser.add_argument("--final-paths", type=int)
    parser.add_argument("--output", default="reports/latency/optimized_profile.json")
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
    position = int(round((len(values) - 1) * fraction))
    return float(values[position])


if __name__ == "__main__":
    main()
