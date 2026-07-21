from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore, NumpyVectorStore
from src.compute_eval import InMemoryComputeTracker
from src.evaluation import (
    CachedEvaluationRunner,
    EvaluationCache,
    context_answer_recall,
    fingerprint_bundle,
    fingerprint_run,
    normalize_text,
    summarize_metrics,
    write_evaluation_report,
)
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig
from src.optimized_grag.models import TrainedStructuralEncoder


def _metaqa_dataset(hop: int) -> dict:
    artifact_name = "metaqa_unified_vanilla_train"
    return {
        "label": f"MetaQA {hop}-hop vanilla test",
        "artifact_name": artifact_name,
        "structural_artifact_name": "metaqa_shared_graph_train",
        "report_name": f"metaqa_{hop}hop_vanilla_test",
        "adapter_name": "metaqa",
        "adapter_kwargs": {
            "hop": f"{hop}-hop",
            "variant": "vanilla",
            "split": "test",
        },
        "limit_documents": None,
        "limit_eval_samples": 500,
        "max_eval_questions": 500,
        "native_graph_only": True,
        "normal_index_documents": True,
        "min_answer_coverage": 0.99,
        "retrieval_max_hops": 3,
        "normal_config": GraphRAGConfig(
            chunk_size=900,
            chunk_overlap=120,
            top_k=12,
            vector_candidate_k=64,
            vector_seed_k=12,
            max_vector_seed_entities=64,
            graph_depth=3,
            graph_limit=1200,
            ranked_graph_limit=240,
            # Large corpus batches let a multi-sequence embedding server keep the
            # GPU busy.  The cache commits after each batch, so an interrupted
            # first build loses at most one batch of work.
            index_batch_size=2048,
        ),
        "optimized_config": OptimizedGraphRAGConfig(
            chunk_size=900,
            chunk_overlap=120,
            max_region_nodes=1000,
            max_capsule_nodes=250,
            max_capsule_edges=900,
            anchors_per_region=5,
            max_capsules_per_region=5,
            first_stage_candidate_capsules=100,
            top_k_capsules=12,
            max_merged_nodes=2400,
            max_candidate_paths=11,
            max_path_length=3,
            final_paths_to_llm=11,
            max_seed_neighbors=200,
            seed_expansion_depth=3,
            max_seed_expansion_nodes=6000,
        ),
    }


DATASETS = {
    "metaqa_1hop": _metaqa_dataset(1),
    "metaqa_2hop": _metaqa_dataset(2),
    "metaqa_3hop": _metaqa_dataset(3),
    "2wiki": {
        "label": "2WikiMultiHopQA train sample",
        "artifact_name": "2wiki_train_sample",
        "adapter_name": "2wiki",
        "adapter_kwargs": {"split": "train"},
        "limit_documents": 800,
        "limit_eval_samples": 200,
        "max_eval_questions": 40,
        "native_graph_only": False,
        "retrieval_max_hops": 2,
        "normal_config": GraphRAGConfig(
            chunk_size=850,
            chunk_overlap=100,
            top_k=8,
            graph_depth=2,
            graph_limit=120,
        ),
        "optimized_config": OptimizedGraphRAGConfig(
            chunk_size=850,
            chunk_overlap=100,
            max_region_nodes=700,
            max_capsule_nodes=180,
            max_capsule_edges=650,
            anchors_per_region=4,
            max_capsules_per_region=4,
            first_stage_candidate_capsules=100,
            top_k_capsules=10,
            max_candidate_paths=200,
            final_paths_to_llm=12,
        ),
    },
    "multihop_rag": {
        "label": "MultiHop-RAG train sample",
        "artifact_name": "multihop_rag_train_sample",
        "adapter_name": "multihop_rag",
        "adapter_kwargs": {"split": "train"},
        "limit_documents": 1000,
        "limit_eval_samples": 200,
        "max_eval_questions": 40,
        "native_graph_only": False,
        "retrieval_max_hops": 2,
        "normal_config": GraphRAGConfig(
            chunk_size=850,
            chunk_overlap=100,
            top_k=8,
            graph_depth=2,
            graph_limit=120,
        ),
        "optimized_config": OptimizedGraphRAGConfig(
            chunk_size=850,
            chunk_overlap=100,
            max_region_nodes=700,
            max_capsule_nodes=180,
            max_capsule_edges=650,
            anchors_per_region=4,
            max_capsules_per_region=4,
            first_stage_candidate_capsules=100,
            top_k_capsules=10,
            max_candidate_paths=200,
            final_paths_to_llm=12,
        ),
    },
}


def main() -> None:
    args = parse_args()
    dataset = DATASETS[args.dataset]
    retrieval_max_hops = int(
        args.max_hops
        if args.max_hops is not None
        else dataset.get("retrieval_max_hops", 3)
    )
    runtime_configs = _retrieval_configs(dataset, retrieval_max_hops)
    if args.max_candidate_paths is not None or args.final_paths is not None:
        optimized = runtime_configs["optimized_grag"]
        runtime_configs["optimized_grag"] = replace(
            optimized,
            max_candidate_paths=(
                args.max_candidate_paths
                if args.max_candidate_paths is not None
                else optimized.max_candidate_paths
            ),
            final_paths_to_llm=(
                args.final_paths
                if args.final_paths is not None
                else optimized.final_paths_to_llm
            ),
        )
    if args.full_eval and args.max_questions is not None:
        raise ValueError("Use either --full-eval or --max-questions, not both.")
    max_questions = (
        None
        if args.full_eval
        else args.max_questions or int(dataset["max_eval_questions"])
    )
    load_limit = (
        None
        if max_questions is None
        else max(int(dataset["limit_eval_samples"]), max_questions)
    )
    bundle = get_dataset_adapter(
        str(dataset["adapter_name"]),
        **dict(dataset["adapter_kwargs"]),
    ).load(
        limit_documents=dataset["limit_documents"],
        limit_eval_samples=load_limit,
    )
    samples = bundle.eval_samples if max_questions is None else bundle.eval_samples[:max_questions]
    _validate_corpus_coverage(bundle, samples, dataset.get("min_answer_coverage"))

    embedding_model = args.embedding_model
    llm_model = args.llm_model
    ollama_options = {"num_gpu": 999}
    if args.embedding_backend == "llama_cpp":
        base_embedding = LlamaCppEmbeddingAdapter(
            LlamaCppEmbeddingConfig(
                model=embedding_model,
                base_url=args.llama_cpp_url,
                timeout_seconds=args.timeout,
                batch_size=min(max(args.embedding_batch_size, 16), 256),
                max_concurrent_requests=args.embedding_request_workers,
            )
        )
        # The cache key describes embedding semantics, not the HTTP transport.
        # Ollama and the bundled llama.cpp server were numerically verified to
        # expose the same local nomic model/pooling (sample cosine >= 0.99998),
        # so a GPU-server switch must not recompute the entire corpus.
        embedding_namespace = local_embedding_cache_namespace(embedding_model)
    else:
        base_embedding = OllamaEmbeddingAdapter(
            OllamaEmbeddingConfig(
                model=embedding_model,
                base_url=args.ollama_url,
                keep_alive="30m",
                options=ollama_options,
                timeout_seconds=args.timeout,
                batch_size=args.embedding_batch_size,
            )
        )
        embedding_namespace = local_embedding_cache_namespace(embedding_model)
    embedding_adapter = CachingEmbeddingAdapter(
        base_embedding,
        cache_path=args.embedding_cache,
        namespace=embedding_namespace,
        model_name=embedding_model,
    )
    llm = OllamaLLM(
        OllamaLLMConfig(
            model=llm_model,
            base_url=args.ollama_url,
            temperature=0.0,
            num_predict=160,
            keep_alive="30m",
            default_options=ollama_options,
            timeout_seconds=args.timeout,
        )
    )

    bundle_hash = fingerprint_bundle(bundle, samples)
    corpus_hash = fingerprint_bundle(bundle, [])
    source_hash = _source_fingerprint()
    cache = EvaluationCache(root=args.evaluation_cache)
    runner = CachedEvaluationRunner(cache=cache, max_workers=args.max_workers)
    systems = ["normal_grag", "optimized_grag"] if args.system == "both" else [args.system]
    all_records = []
    all_rows = []
    index_results = {}
    fingerprints = {}
    evaluation_name = str(dataset.get("report_name") or dataset["artifact_name"])
    vector_index_fingerprint = fingerprint_run(
        {
            "corpus": corpus_hash,
            "embedding_namespace": embedding_adapter.namespace,
            "chunk_size": asdict(runtime_configs["normal_grag"])["chunk_size"],
            "chunk_overlap": asdict(runtime_configs["normal_grag"])["chunk_overlap"],
            "store": "numpy_exact_cosine_v1",
        }
    )
    normal_vector_index_directory = (
        Path(args.vector_index_cache) / vector_index_fingerprint
    )
    query_embeddings_precached = False

    for system in systems:
        config = runtime_configs[system]
        run_fingerprint = fingerprint_run(
            {
                "bundle": bundle_hash,
                "source": source_hash,
                "structural_artifacts": (
                    _structural_artifact_fingerprint(
                        str(dataset.get("structural_artifact_name") or dataset["artifact_name"]),
                        str(dataset["artifact_name"]),
                    )
                    if system == "optimized_grag"
                    else None
                ),
                "system": system,
                "config": asdict(config),
                "embedding_model": embedding_model,
                "embedding_backend": args.embedding_backend,
                "embedding_namespace": embedding_namespace,
                "llm_model": llm_model if args.run_generation else None,
                "run_generation": args.run_generation,
            }
        )
        fingerprints[system] = run_fingerprint
        namespace = f"{evaluation_name}/{system}"
        needs_execution = runner.needs_execution(
            namespace=namespace,
            fingerprint=run_fingerprint,
            system=system,
            samples=samples,
            refresh=args.refresh_cache,
        )
        pipeline = None
        if needs_execution:
            if not query_embeddings_precached:
                _precache_query_embeddings(embedding_adapter, samples)
                query_embeddings_precached = True
            pipeline = _build_pipeline(
                system=system,
                dataset=dataset,
                config=config,
                embedding_adapter=embedding_adapter,
                llm=llm,
                vector_index_cache=(
                    normal_vector_index_directory
                    if system == "normal_grag"
                    else None
                ),
            )
            started = time.perf_counter()
            index_report = _index_pipeline(pipeline, bundle, dataset)
            index_seconds = time.perf_counter() - started
            index_results[system] = {
                "cache_skipped": False,
                "seconds": index_seconds,
                "report": _jsonable_report(index_report),
            }
        else:
            index_results[system] = {
                "cache_skipped": True,
                "seconds": 0.0,
                "report": None,
            }

        if pipeline is None:
            def execute(_sample):
                raise RuntimeError("Cached evaluation unexpectedly requested execution")
        elif args.run_generation:
            execute = lambda sample, pipeline=pipeline: pipeline.answer(sample.question)
        else:
            execute = lambda sample, pipeline=pipeline: pipeline.retrieve(sample.question)
        records, rows = runner.run(
            namespace=namespace,
            fingerprint=run_fingerprint,
            system=system,
            samples=samples,
            execute=execute,
            refresh=args.refresh_cache,
        )
        all_records.extend(records)
        all_rows.extend(rows)

    output_dir = Path(args.output_root) / evaluation_name
    run_config = {
        "dataset": args.dataset,
        "dataset_config": _dataset_config_json(dataset),
        "systems": systems,
        "questions": len(samples),
        "full_eval": bool(args.full_eval),
        "retrieval_max_hops": retrieval_max_hops,
        "runtime_configs": {
            system: asdict(runtime_configs[system]) for system in systems
        },
        "run_generation": args.run_generation,
        "embedding_model": embedding_model,
        "embedding_backend": args.embedding_backend,
        "embedding_namespace": embedding_namespace,
        "llm_model": llm_model,
        "bundle_fingerprint": bundle_hash,
        "corpus_fingerprint": corpus_hash,
        "source_fingerprint": source_hash,
        "normal_vector_index": {
            "fingerprint": vector_index_fingerprint,
            "directory": str(normal_vector_index_directory),
            "manifest": _read_json_object(
                normal_vector_index_directory / "manifest.json"
            ),
        },
        "run_fingerprints": fingerprints,
        "index": index_results,
    }
    paths = write_evaluation_report(
        output_dir,
        records=all_records,
        rows=all_rows,
        run_config=run_config,
    )
    summaries = {
        system: summarize_metrics([row for row in all_rows if row["system"] == system])
        for system in systems
    }
    print(json.dumps({"summary": summaries, "reports": {k: str(v) for k, v in paths.items()}}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached, evidence-aware GraphRAG evaluation."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--system",
        choices=("both", "normal_grag", "optimized_grag"),
        default="both",
    )
    parser.add_argument("--max-questions", type=int)
    parser.add_argument(
        "--max-hops",
        type=int,
        help=(
            "Shared query-time traversal ceiling for both systems. Defaults to "
            "the dataset benchmark setting. Does not change offline capsule radius."
        ),
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Evaluate the complete selected split instead of the configured sample.",
    )
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "Concurrent question workers. Defaults to 1 to avoid duplicate GPU "
            "model work/OOM; increase only after measuring available memory."
        ),
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument(
        "--embedding-backend",
        choices=("ollama", "llama_cpp"),
        default="ollama",
        help="Embedding API. llama_cpp enables true multi-sequence bulk indexing.",
    )
    parser.add_argument(
        "--llama-cpp-url",
        default="http://127.0.0.1:11435",
        help="OpenAI-compatible llama.cpp server used by --embedding-backend llama_cpp.",
    )
    parser.add_argument(
        "--embedding-request-workers",
        type=int,
        default=8,
        help=(
            "Concurrent llama.cpp embedding requests. Eight 128-text requests "
            "is the measured 6 GB GPU default; ignored by Ollama."
        ),
    )
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--llm-model", default="qwen3:4b-instruct")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=16,
        help=(
            "Maximum texts per embedding request. Use a small value for Ollama; "
            "128 is the measured RTX 4050 llama.cpp setting (hard cap 256)."
        ),
    )
    parser.add_argument(
        "--max-candidate-paths",
        type=int,
        help="Optional optimized path-search ablation override.",
    )
    parser.add_argument(
        "--final-paths",
        type=int,
        help="Optional optimized final evidence-path ablation override.",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument(
        "--vector-index-cache",
        default="artifacts/cache/vector_indexes",
        help="Persistent vectorized normal-GraphRAG indexes keyed by corpus/model/config.",
    )
    parser.add_argument("--evaluation-cache", default="artifacts/cache/evaluations")
    parser.add_argument("--output-root", default="reports/evaluation")
    return parser.parse_args()


def _build_pipeline(
    *,
    system,
    dataset,
    config,
    embedding_adapter,
    llm,
    vector_index_cache=None,
):
    if system == "normal_grag":
        return GraphRAG(
            graph_store=InMemoryGraphStore(),
            vector_store=NumpyVectorStore(cache_directory=vector_index_cache),
            embedding_adapter=embedding_adapter,
            llm=llm,
            instrumentation=InMemoryComputeTracker(log_stages=False, namespace="normal"),
            config=config,
        )
    projector_dir = ROOT / "artifacts" / "rgcn" / str(dataset["artifact_name"])
    structural_dir = ROOT / "artifacts" / "rgcn" / str(
        dataset.get("structural_artifact_name") or dataset["artifact_name"]
    )
    structural_encoder = (
        TrainedStructuralEncoder(
            structural_dir,
            projector_artifact_dir=projector_dir,
        )
        if _has_structural_artifact(structural_dir)
        else None
    )
    return OptimizedGraphRAG(
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
        embedding_adapter=embedding_adapter,
        llm=llm,
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="optimized"),
        config=config,
        structural_encoder=structural_encoder,
    )


def _retrieval_configs(dataset: dict, max_hops: int) -> dict[str, object]:
    """Apply one query-time hop ceiling to both comparison systems.

    `OptimizedGraphRAGConfig.max_hops` controls offline capsule construction and
    is deliberately unchanged; changing it would invalidate trained artifacts.
    """
    if max_hops < 1:
        raise ValueError("--max-hops must be at least 1")
    return {
        "normal_grag": replace(dataset["normal_config"], graph_depth=max_hops),
        "optimized_grag": replace(
            dataset["optimized_config"],
            max_path_length=max_hops,
            seed_expansion_depth=max_hops,
        ),
    }


def _index_pipeline(pipeline, bundle, dataset):
    native_only = bool(dataset.get("native_graph_only"))
    if bundle.graph_nodes and bundle.graph_edges:
        include_documents = (
            bool(dataset.get("normal_index_documents", True))
            if isinstance(pipeline, GraphRAG)
            else not native_only
        )
        return pipeline.index_dataset_bundle(
            bundle,
            include_documents=include_documents,
        )
    return pipeline.index_documents(bundle.documents)


def _has_structural_artifact(artifact_dir: Path) -> bool:
    metadata_path = artifact_dir / "capsule_rgcn_metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = str(metadata.get("projection_manifest") or "")
            if manifest and (artifact_dir / manifest.split(":", 1)[0]).exists():
                return True
        except (OSError, json.JSONDecodeError):
            pass
    return (artifact_dir / "capsule_rgcn_embeddings.npz").exists()


def _read_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _validate_corpus_coverage(bundle, samples, minimum):
    if minimum is None:
        return
    # Normalize the multi-megabyte corpus once. Calling
    # context_answer_recall for every sample re-tokenized the entire MetaQA KB
    # hundreds of times and made validation slower than retrieval evaluation.
    corpus = normalize_text("\n".join(document.text for document in bundle.documents))
    measured = [
        float(
            any(
                normalized in corpus
                for normalized in (normalize_text(answer) for answer in sample.answers)
                if normalized
            )
        )
        for sample in samples
        if sample.answers
    ]
    coverage = sum(measured) / len(measured) if measured else 1.0
    if coverage < float(minimum):
        raise RuntimeError(
            f"Invalid benchmark: corpus answer coverage {coverage:.1%} is below {minimum:.1%}."
        )


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    roots = [
        ROOT / "src" / "adapters",
        ROOT / "src" / "compute_eval",
        ROOT / "src" / "evaluation",
        ROOT / "src" / "g_rag",
        ROOT / "src" / "helpers",
        ROOT / "src" / "optimized_grag",
    ]
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _structural_artifact_fingerprint(
    structural_artifact_name: str,
    projector_artifact_name: str | None = None,
) -> str | None:
    structural_directory = ROOT / "artifacts" / "rgcn" / structural_artifact_name
    projector_directory = ROOT / "artifacts" / "rgcn" / (
        projector_artifact_name or structural_artifact_name
    )
    metadata_path = structural_directory / "capsule_rgcn_metadata.json"
    manifest_path = structural_directory / "capsule_rgcn_embeddings.npz"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            manifest = str(metadata.get("projection_manifest") or "")
            if manifest:
                manifest_path = structural_directory / manifest.split(":", 1)[0]
        except (OSError, json.JSONDecodeError):
            pass
    paths = [
        manifest_path,
        metadata_path,
        projector_directory / "query_projection_mlp.pt",
        projector_directory / "query_projection_metadata.json",
    ]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in existing:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _precache_query_embeddings(
    embedding_adapter: CachingEmbeddingAdapter,
    samples,
    *,
    batch_size: int = 256,
) -> None:
    """Batch query embedding work before per-question retrieval.

    This turns a full MetaQA run from tens of thousands of one-text Ollama
    requests into bounded batches while preserving the persistent cache.
    """
    questions = [sample.question for sample in samples]
    for start in range(0, len(questions), batch_size):
        embedding_adapter.embed_texts(questions[start : start + batch_size])


def _jsonable_report(report):
    if report is None:
        return None
    result = {}
    for name in getattr(report, "__dataclass_fields__", {}):
        result[name] = getattr(report, name)
    return result


def _dataset_config_json(dataset):
    return {
        key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
        for key, value in dataset.items()
    }


if __name__ == "__main__":
    main()
