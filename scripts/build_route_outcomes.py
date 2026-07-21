"""Measure Hybrid RAG, GraphRAG, and non-learned Optimized GraphRAG routes.

The generated JSONL is a counterfactual training set: every selected question
has one observed quality/cost outcome for each of the three possible routes.
It intentionally does not load R-GCN weights or the query-projection MLP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapters.datasets import get_dataset_adapter
from src.adapters.datasets.base import DatasetBundle
from src.adapters.embedding import CachingEmbeddingAdapter, OllamaEmbeddingAdapter, OllamaEmbeddingConfig, local_embedding_cache_namespace
from src.adapters.llm import OllamaLLM, OllamaLLMConfig
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.compute_eval import InMemoryComputeTracker
from src.evaluation import evaluate_record
from src.evaluation.runner import retrieval_record_from_output
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig
from src.optimized_grag.models import FeatureStructuralEncoder
from src.rag.bm25 import BM25RAG
from src.rag.hybrid import HybridRAG
from src.rag.normal import NormalRAG
from src.rag.types import RAGConfig
from src.rl_router import CostWeights, RewardConfig, cheapest_acceptable_route, query_features, route_cost, route_quality, route_reward
from src.rl_router.outcomes import run_cost_maxima


ROUTES = ("hybrid_rag", "graph_rag", "optimized_grag")


def main() -> None:
    args = parse_args()
    bundle = _load_bundle(args)
    samples = bundle.eval_samples[: args.max_questions]
    if not samples:
        raise RuntimeError("The selected dataset split has no evaluation questions")

    embedding = CachingEmbeddingAdapter(
        OllamaEmbeddingAdapter(
            OllamaEmbeddingConfig(
                model=args.embedding_model,
                base_url=args.ollama_url,
                keep_alive="30m",
                timeout_seconds=args.timeout,
                batch_size=args.embedding_batch_size,
                options={"num_gpu": 999},
            )
        ),
        cache_path=args.embedding_cache,
        namespace=local_embedding_cache_namespace(args.embedding_model),
        model_name=args.embedding_model,
    )
    llm = OllamaLLM(
        OllamaLLMConfig(
            model=args.llm_model,
            base_url=args.ollama_url,
            temperature=0.0,
            num_predict=160,
            keep_alive="30m",
            default_options={"num_gpu": 999},
            timeout_seconds=args.timeout,
        )
    )
    pipelines, index_reports = _build_and_index_pipelines(bundle, embedding, llm)

    raw_outcomes: list[dict] = []
    for sample in samples:
        features = query_features(sample.question)
        for route, pipeline in pipelines.items():
            started = time.perf_counter()
            output = pipeline.answer(sample.question) if args.run_generation else pipeline.retrieve(sample.question)
            elapsed = time.perf_counter() - started
            record = retrieval_record_from_output(
                system=route,
                sample=sample,
                output=output,
                latency_seconds=elapsed,
            )
            metrics = evaluate_record(sample, record)
            raw_outcomes.append(
                {
                    "sample_id": sample.id,
                    "question": sample.question,
                    "route": route,
                    "features": features,
                    "metrics": _jsonable(metrics),
                    "latency_seconds": record.latency_seconds,
                    "retrieved_units": len(record.evidence_units),
                    "context_chars": len(record.context_text),
                    "details": _jsonable(record.details),
                    "generation_enabled": bool(args.run_generation),
                    "learned_structural_models_enabled": False,
                }
            )

    maxima = run_cost_maxima(raw_outcomes)
    weights = CostWeights()
    reward_config = RewardConfig(acceptable_quality=args.acceptable_quality)
    by_sample: dict[str, list[dict]] = {}
    for outcome in raw_outcomes:
        outcome["quality_score"] = route_quality(outcome["metrics"])
        outcome["normalized_cost"] = route_cost(outcome, maxima=maxima, weights=weights)
        outcome["reward"] = route_reward(
            outcome["quality_score"], outcome["normalized_cost"], config=reward_config
        )
        by_sample.setdefault(outcome["sample_id"], []).append(outcome)
    for outcomes in by_sample.values():
        if {item["route"] for item in outcomes} != set(ROUTES):
            raise RuntimeError("Outcome collection did not produce all three routes for a question")
        oracle = cheapest_acceptable_route(outcomes, config=reward_config)
        for outcome in outcomes:
            outcome["cheapest_acceptable_route"] = oracle["route"]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for outcome in raw_outcomes:
            file.write(json.dumps(outcome, ensure_ascii=False) + "\n")

    run_config_path = output_path.with_suffix(".run_config.json")
    run_config_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "dataset_split": args.split,
                "questions": len(samples),
                "max_documents": None if args.full_corpus else args.max_documents,
                "routes": ROUTES,
                "generation_enabled": bool(args.run_generation),
                "learned_structural_models_enabled": False,
                "optimized_structural_encoder": "FeatureStructuralEncoder",
                "quality_definition": "0.40 answer + 0.25 faithfulness + 0.20 evidence + 0.15 completeness",
                "cost_maxima": maxima,
                "cost_weights": weights.to_dict(),
                "reward_config": reward_config.to_dict(),
                "index_reports": index_reports,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"outcomes": str(output_path), "run_config": str(run_config_path), "records": len(raw_outcomes)}, indent=2))


def _load_bundle(args: argparse.Namespace):
    if args.dataset.startswith("metaqa_"):
        hop = args.dataset.removeprefix("metaqa_").removesuffix("hop")
        adapter = get_dataset_adapter("metaqa", hop=f"{hop}-hop", variant="vanilla", split=args.split)
        # MetaQA's adapter uses one triple list to create both its graph and
        # fact documents.  Always load the entire KB so GraphRAG and Optimized
        # GraphRAG never receive an invalid sliced graph; only then trim the
        # documents used by the Hybrid-RAG smoke route.
        bundle = adapter.load(limit_documents=None, limit_eval_samples=args.max_questions)
        if not args.full_corpus:
            bundle = DatasetBundle(
                name=bundle.name,
                documents=bundle.documents[: args.max_documents],
                eval_samples=bundle.eval_samples,
                graph_nodes=bundle.graph_nodes,
                graph_edges=bundle.graph_edges,
                metadata={**bundle.metadata, "hybrid_document_limit": args.max_documents},
            )
        return bundle
    elif args.dataset == "2wiki":
        adapter = get_dataset_adapter("2wiki", split=args.split)
    else:
        adapter = get_dataset_adapter("multihop_rag", split=args.split)
    document_limit = None if args.full_corpus else args.max_documents
    return adapter.load(limit_documents=document_limit, limit_eval_samples=args.max_questions)


def _build_and_index_pipelines(bundle, embedding, llm):
    rag_config = RAGConfig(chunk_size=900, chunk_overlap=120, top_k=12)
    normal = NormalRAG(
        vector_store=InMemoryVectorStore(), embedding_adapter=embedding, llm=llm,
        config=rag_config, instrumentation=InMemoryComputeTracker(log_stages=False, namespace="dense"),
    )
    bm25 = BM25RAG(
        llm=llm, config=rag_config,
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="bm25"),
    )
    hybrid = HybridRAG(
        normal_rag=normal, bm25_rag=bm25, llm=llm, config=rag_config,
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="hybrid"),
    )
    graph = GraphRAG(
        graph_store=InMemoryGraphStore(), vector_store=InMemoryVectorStore(),
        embedding_adapter=embedding, llm=llm,
        config=GraphRAGConfig(top_k=12, graph_depth=3, graph_limit=1200, ranked_graph_limit=240),
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="graph"),
    )
    optimized = OptimizedGraphRAG(
        graph_store=InMemoryGraphStore(), vector_store=InMemoryVectorStore(),
        embedding_adapter=embedding, llm=llm,
        config=OptimizedGraphRAGConfig(
            max_region_nodes=1000, max_capsule_nodes=250, max_capsule_edges=900,
            first_stage_candidate_capsules=100, top_k_capsules=12,
            max_candidate_paths=11, max_path_length=3, final_paths_to_llm=11,
            max_seed_neighbors=200, seed_expansion_depth=3, max_seed_expansion_nodes=6000,
        ),
        # Explicitly disables exported R-GCN and query-projection artifacts.
        structural_encoder=FeatureStructuralEncoder(),
        instrumentation=InMemoryComputeTracker(log_stages=False, namespace="optimized"),
    )
    native_graph = bool(bundle.graph_nodes and bundle.graph_edges)
    reports = {
        "hybrid_rag": _jsonable(hybrid.index_documents(bundle.documents)),
        "graph_rag": _jsonable(graph.index_dataset_bundle(bundle, include_documents=True)),
        # MetaQA uses its native graph only; document datasets still build their graph from text.
        "optimized_grag": _jsonable(optimized.index_dataset_bundle(bundle, include_documents=not native_graph)),
    }
    return {"hybrid_rag": hybrid, "graph_rag": graph, "optimized_grag": optimized}, reports


def _jsonable(value):
    if hasattr(value, "__dataclass_fields__"):
        return {key: _jsonable(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build three-route counterfactual RAG outcomes without R-GCN/projector models.")
    parser.add_argument("--dataset", choices=("metaqa_1hop", "metaqa_2hop", "metaqa_3hop", "2wiki", "multihop_rag"), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--max-documents", type=int, default=1000, help="Smoke-run corpus limit. Use --full-corpus for reportable MetaQA results.")
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--acceptable-quality", type=float, default=0.65)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--llm-model", default="qwen3:4b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--output", default="artifacts/rl_router/outcomes.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    main()
