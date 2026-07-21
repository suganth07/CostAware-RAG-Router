# GraphRAG System Documentation

This folder explains the project from first principles through implementation
details. Start with the overview, then follow the numbered guides.

1. [System overview](01-system-overview.md) — the problem, the intuition, and the complete workflow.
2. [Graph construction and capsules](02-graph-and-capsules.md) — how data becomes a bounded graph hierarchy.
3. [Retrieval algorithm](03-retrieval.md) — how one unknown-complexity question becomes ranked evidence.
4. [R-GCN and unified projector](04-model-training.md) — what is learned, how it is trained, and artifact safety.
5. [Normal vs optimized GraphRAG](05-normal-vs-optimized.md) — algorithmic and operational comparison.
6. [Evaluation and caching](06-evaluation.md) — honest splits, metrics, cache keys, and interpretation.
7. [Operations runbook](07-operations.md) — exact training/evaluation order and GPU guidance.
8. [Examples and troubleshooting](08-examples-and-troubleshooting.md) — worked questions and failure diagnosis.
9. [Glossary and configuration](09-glossary-and-configuration.md) — terminology and important limits.
10. [Codebase tour](10-codebase-tour.md) — modules, interfaces, storage, updates, and extension points.
11. [Performance and storage](11-performance-and-storage.md) — measured latency, optimizations, and when Qdrant helps.
12. [Three-route RL router](12-three-route-rl-router.md) — R-GCN-free Hybrid/Graph/Optimized comparison and contextual-bandit training.
13. [RL pipeline and intuition](13-rl-router-pipeline-and-intuition.md) — why the router exists and how data flows through it.
14. [RL outcomes, quality, cost, and reward](14-rl-outcomes-quality-cost-and-reward.md) — training-data schema and the quality-first objective.
15. [Contextual-bandit algorithm](15-contextual-bandit-algorithm.md) — policy mathematics, training, and limitations.
16. [Running and checking RL results](16-rl-running-and-checking-results.md) — commands, verification checks, and troubleshooting.

## One-sentence description

The system compresses a large knowledge graph into overlapping, bounded
“capsules,” learns how text questions relate to those capsules, then performs a
small relation-aware graph search around linked entities while preserving
evidence from multiple possible reasoning depths.

## The most important design rule

Runtime receives only a question:

```python
context = optimized_grag.retrieve(question)
```

It never receives `hop=1`, `hop=2`, or `hop=3`. MetaQA hop labels are used only
to create separate held-out evaluation reports.

## Source map

| Concern | Primary implementation |
|---|---|
| Normal GraphRAG | `src/g_rag/pipeline.py` |
| Optimized pipeline | `src/optimized_grag/pipeline.py` |
| Capsule construction | `src/optimized_grag/partitioning/` |
| Capsule routing | `src/optimized_grag/retrieval/capsule_index.py`, `router.py` |
| Path search | `src/optimized_grag/retrieval/path_search.py` |
| R-GCN | `src/optimized_grag/models/rgcn.py` |
| Model artifact loading | `src/optimized_grag/models/structural.py` |
| Metrics/cache | `src/evaluation/` |
| Reproducible CLI | `scripts/run_grag_evaluation.py` |
| Fair retrieval profiler | `scripts/profile_retrieval_latency.py` |
| Optimized-only path ablations | `scripts/profile_optimized_latency.py` |

# Project Structure

```
├── artifacts
│   ├── rgcn
│   │   ├── metaqa_shared_graph_train
│   │   │   ├── capsule_rgcn_embeddings_v3.npz
│   │   │   ├── capsule_rgcn_embeddings_v4.npz
│   │   │   ├── capsule_rgcn_metadata.json
│   │   │   └── capsule_rgcn.pt
│   │   ├── metaqa_unified_vanilla_train
│   │   │   ├── capsule_semantic_embeddings_nomic_embed_text.npz
│   │   │   ├── query_projection_metadata.json
│   │   │   └── query_projection_mlp.pt
│   │   └── multihop_rag_train_sample
│   │       ├── capsule_rgcn_embeddings_v4.npz
│   │       ├── capsule_rgcn_metadata.json
│   │       └── capsule_rgcn.pt
│   └── rl_report_render
├── docs
│   ├── 01-system-overview.md
│   ├── 02-graph-and-capsules.md
│   ├── 03-retrieval.md
│   ├── 04-model-training.md
│   ├── 05-normal-vs-optimized.md
│   ├── 06-evaluation.md
│   ├── 07-operations.md
│   ├── 08-examples-and-troubleshooting.md
│   ├── 09-glossary-and-configuration.md
│   ├── 10-codebase-tour.md
│   ├── 11-performance-and-storage.md
│   ├── 12-three-route-rl-router.md
│   └── README.md
├── logger
│   ├── __init__.py
│   └── config.py
├── notebooks
│   ├── 01_compare_grag_vs_optimized.ipynb
│   ├── 02_diagnose_rgcn_projection.ipynb
│   └── README.md
├── reports
│   ├── evaluation
│   │   ├── metaqa_1hop_vanilla_test
│   │   │   ├── run_config.json
│   │   │   └── summary_metrics.json
│   │   ├── metaqa_2hop_vanilla_test
│   │   │   ├── run_config.json
│   │   │   └── summary_metrics.json
│   │   └── metaqa_3hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── evaluation_fast_verified
│   │   └── metaqa_2hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── grag_comparison
│   │   ├── metaqa_3hop_vanilla_test
│   │   │   ├── coverage_metrics.csv
│   │   │   ├── index_metrics.csv
│   │   │   ├── run_config.json
│   │   │   ├── stage_totals.csv
│   │   │   └── summary_metrics.csv
│   │   └── multihop_rag_train_sample
│   │       ├── coverage_metrics.csv
│   │       ├── index_metrics.csv
│   │       ├── run_config.json
│   │       ├── stage_totals.csv
│   │       └── summary_metrics.csv
│   ├── latency
│   │   ├── optimized_exact_seed_precomputed_terms_500_profile.json
│   │   ├── optimized_profile_after.json
│   │   ├── optimized_profile_candidate1000.json
│   │   ├── optimized_profile_candidate500.json
│   │   ├── optimized_profile_final.json
│   │   ├── optimized_profile.json
│   │   ├── production_retrieval_comparison_500.json
│   │   └── README.md
│   ├── production_comparison_verified
│   │   └── metaqa_2hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── production_normal_verified
│   │   └── metaqa_2hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── projection_evaluation
│   │   ├── metaqa_1hop_raw_semantic_path_500.json
│   │   ├── metaqa_1hop_semantic_path_projector_500.json
│   │   ├── metaqa_2hop_raw_semantic_path_500.json
│   │   ├── metaqa_2hop_semantic_path_projector_500.json
│   │   ├── metaqa_3hop_raw_semantic_path_500.json
│   │   ├── metaqa_3hop_semantic_path_projector_500.json
│   │   └── README.md
│   ├── accuracy_analysis_and_recovery_plan.md
│   └── metaqa_verified_assessment.md
├── scripts
│   ├── build_capsule_semantic_artifact.py
│   ├── build_route_outcomes.py
│   ├── evaluate_query_projection.py
│   ├── profile_optimized_latency.py
│   ├── profile_retrieval_latency.py
│   ├── repool_rgcn_artifact.py
│   ├── run_grag_evaluation.py
│   ├── start_batched_embedding_server.ps1
│   └── train_cost_aware_router.py
├── src
│   ├── adapters
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── hf_disk.py
│   │   │   ├── metaqa.py
│   │   │   ├── multihop_rag.py
│   │   │   ├── paths.py
│   │   │   ├── registry.py
│   │   │   └── wiki_multihop_qa.py
│   │   ├── embedding
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py
│   │   │   ├── cache.py
│   │   │   ├── llama_cpp.py
│   │   │   └── ollama.py
│   │   ├── llm
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py
│   │   │   └── ollama.py
│   │   ├── storage
│   │   │   ├── graph_store
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── in_memory.py
│   │   │   │   └── neo4j.py
│   │   │   ├── vector_store
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── in_memory.py
│   │   │   │   ├── numpy_store.py
│   │   │   │   └── qdrant.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── compute_eval
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── in_memory.py
│   │   ├── metrics.py
│   │   └── noop.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── core.py
│   │   ├── README.md
│   │   └── runner.py
│   ├── g_rag
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── context.py
│   │   ├── extractors.py
│   │   ├── instrumentation.py
│   │   ├── pipeline.py
│   │   ├── ranking.py
│   │   ├── README.md
│   │   └── types.py
│   ├── helpers
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   └── documents.py
│   ├── optimized_grag
│   │   ├── graph
│   │   │   ├── __init__.py
│   │   │   ├── algorithms.py
│   │   │   └── local.py
│   │   ├── graph_construction
│   │   │   ├── __init__.py
│   │   │   ├── builder.py
│   │   │   ├── fallbacks.py
│   │   │   ├── README.md
│   │   │   └── types.py
│   │   ├── graph_updates
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   ├── types.py
│   │   │   └── updater.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── projection_supervision.py
│   │   │   ├── README.md
│   │   │   ├── rgcn.py
│   │   │   └── structural.py
│   │   ├── notebooks
│   │   │   ├── projection
│   │   │   │   ├── 01_train_query_projection.ipynb
│   │   │   │   └── README.md
│   │   │   └── rgcn
│   │   │       ├── 01_train_capsule_rgcn.ipynb
│   │   │       └── README.md
│   │   ├── partitioning
│   │   │   ├── __init__.py
│   │   │   ├── anchors.py
│   │   │   ├── capsule_builder.py
│   │   │   ├── hubs.py
│   │   │   ├── pipeline.py
│   │   │   ├── README.md
│   │   │   ├── regions.py
│   │   │   ├── supergraph.py
│   │   │   ├── types.py
│   │   │   └── validation.py
│   │   ├── retrieval
│   │   │   ├── __init__.py
│   │   │   ├── capsule_index.py
│   │   │   ├── context.py
│   │   │   ├── path_search.py
│   │   │   ├── README.md
│   │   │   ├── router.py
│   │   │   ├── scoring.py
│   │   │   └── types.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── Idea.md
│   │   ├── pipeline.py
│   │   └── types.py
│   ├── rag
│   │   ├── bm25
│   │   │   ├── __init__.py
│   │   │   ├── index.py
│   │   │   └── pipeline.py
│   │   ├── hybrid
│   │   │   ├── __init__.py
│   │   │   ├── fusion.py
│   │   │   └── pipeline.py
│   │   ├── normal
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py
│   │   │   └── README.md
│   │   ├── __init__.py
│   │   ├── context.py
│   │   └── types.py
│   ├── rl_router
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── outcomes.py
│   │   └── policy.py
│   └── __init__.py
├── tests
│   ├── smoke_dataset_adapters.py
│   ├── smoke_graph_construction_updates.py
│   ├── smoke_in_memory_grag.py
│   ├── smoke_optimized_grag.py
│   ├── smoke_rag_variants.py
│   ├── test_accuracy_pipeline.py
│   ├── test_metaqa_retrieval_correctness.py
│   ├── test_ollama_embedding_adapter.py
│   ├── test_performance_regressions.py
│   ├── test_production_grag.py
│   └── test_rl_router.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── Run.md
└── to-check.md
```

# File Contents

## docs/README.md

````markdown
# GraphRAG System Documentation

This folder explains the project from first principles through implementation
details. Start with the overview, then follow the numbered guides.

1. [System overview](01-system-overview.md) — the problem, the intuition, and the complete workflow.
2. [Graph construction and capsules](02-graph-and-capsules.md) — how data becomes a bounded graph hierarchy.
3. [Retrieval algorithm](03-retrieval.md) — how one unknown-complexity question becomes ranked evidence.
4. [R-GCN and unified projector](04-model-training.md) — what is learned, how it is trained, and artifact safety.
5. [Normal vs optimized GraphRAG](05-normal-vs-optimized.md) — algorithmic and operational comparison.
6. [Evaluation and caching](06-evaluation.md) — honest splits, metrics, cache keys, and interpretation.
7. [Operations runbook](07-operations.md) — exact training/evaluation order and GPU guidance.
8. [Examples and troubleshooting](08-examples-and-troubleshooting.md) — worked questions and failure diagnosis.
9. [Glossary and configuration](09-glossary-and-configuration.md) — terminology and important limits.
10. [Codebase tour](10-codebase-tour.md) — modules, interfaces, storage, updates, and extension points.
11. [Performance and storage](11-performance-and-storage.md) — measured latency, optimizations, and when Qdrant helps.
12. [Three-route RL router](12-three-route-rl-router.md) — R-GCN-free Hybrid/Graph/Optimized comparison and contextual-bandit training.

## One-sentence description

The system compresses a large knowledge graph into overlapping, bounded
“capsules,” learns how text questions relate to those capsules, then performs a
small relation-aware graph search around linked entities while preserving
evidence from multiple possible reasoning depths.

## The most important design rule

Runtime receives only a question:

```python
context = optimized_grag.retrieve(question)
```

It never receives `hop=1`, `hop=2`, or `hop=3`. MetaQA hop labels are used only
to create separate held-out evaluation reports.

## Source map

| Concern | Primary implementation |
|---|---|
| Normal GraphRAG | `src/g_rag/pipeline.py` |
| Optimized pipeline | `src/optimized_grag/pipeline.py` |
| Capsule construction | `src/optimized_grag/partitioning/` |
| Capsule routing | `src/optimized_grag/retrieval/capsule_index.py`, `router.py` |
| Path search | `src/optimized_grag/retrieval/path_search.py` |
| R-GCN | `src/optimized_grag/models/rgcn.py` |
| Model artifact loading | `src/optimized_grag/models/structural.py` |
| Metrics/cache | `src/evaluation/` |
| Reproducible CLI | `scripts/run_grag_evaluation.py` |
| Fair retrieval profiler | `scripts/profile_retrieval_latency.py` |
| Optimized-only path ablations | `scripts/profile_optimized_latency.py` |

````

# Project Structure

```
├── artifacts
│   ├── rgcn
│   │   ├── metaqa_shared_graph_train
│   │   │   ├── capsule_rgcn_embeddings_v3.npz
│   │   │   ├── capsule_rgcn_embeddings_v4.npz
│   │   │   ├── capsule_rgcn_metadata.json
│   │   │   └── capsule_rgcn.pt
│   │   ├── metaqa_unified_vanilla_train
│   │   │   ├── capsule_semantic_embeddings_nomic_embed_text.npz
│   │   │   ├── query_projection_metadata.json
│   │   │   └── query_projection_mlp.pt
│   │   └── multihop_rag_train_sample
│   │       ├── capsule_rgcn_embeddings_v4.npz
│   │       ├── capsule_rgcn_metadata.json
│   │       └── capsule_rgcn.pt
│   └── rl_report_render
├── docs
│   ├── 01-system-overview.md
│   ├── 02-graph-and-capsules.md
│   ├── 03-retrieval.md
│   ├── 04-model-training.md
│   ├── 05-normal-vs-optimized.md
│   ├── 06-evaluation.md
│   ├── 07-operations.md
│   ├── 08-examples-and-troubleshooting.md
│   ├── 09-glossary-and-configuration.md
│   ├── 10-codebase-tour.md
│   ├── 11-performance-and-storage.md
│   ├── 12-three-route-rl-router.md
│   └── README.md
├── logger
│   ├── __init__.py
│   └── config.py
├── notebooks
│   ├── 01_compare_grag_vs_optimized.ipynb
│   ├── 02_diagnose_rgcn_projection.ipynb
│   └── README.md
├── reports
│   ├── evaluation
│   │   ├── metaqa_1hop_vanilla_test
│   │   │   ├── run_config.json
│   │   │   └── summary_metrics.json
│   │   ├── metaqa_2hop_vanilla_test
│   │   │   ├── run_config.json
│   │   │   └── summary_metrics.json
│   │   └── metaqa_3hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── evaluation_fast_verified
│   │   └── metaqa_2hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── grag_comparison
│   │   ├── metaqa_3hop_vanilla_test
│   │   │   ├── coverage_metrics.csv
│   │   │   ├── index_metrics.csv
│   │   │   ├── run_config.json
│   │   │   ├── stage_totals.csv
│   │   │   └── summary_metrics.csv
│   │   └── multihop_rag_train_sample
│   │       ├── coverage_metrics.csv
│   │       ├── index_metrics.csv
│   │       ├── run_config.json
│   │       ├── stage_totals.csv
│   │       └── summary_metrics.csv
│   ├── latency
│   │   ├── optimized_exact_seed_precomputed_terms_500_profile.json
│   │   ├── optimized_profile_after.json
│   │   ├── optimized_profile_candidate1000.json
│   │   ├── optimized_profile_candidate500.json
│   │   ├── optimized_profile_final.json
│   │   ├── optimized_profile.json
│   │   ├── production_retrieval_comparison_500.json
│   │   └── README.md
│   ├── production_comparison_verified
│   │   └── metaqa_2hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── production_normal_verified
│   │   └── metaqa_2hop_vanilla_test
│   │       ├── run_config.json
│   │       └── summary_metrics.json
│   ├── projection_evaluation
│   │   ├── metaqa_1hop_raw_semantic_path_500.json
│   │   ├── metaqa_1hop_semantic_path_projector_500.json
│   │   ├── metaqa_2hop_raw_semantic_path_500.json
│   │   ├── metaqa_2hop_semantic_path_projector_500.json
│   │   ├── metaqa_3hop_raw_semantic_path_500.json
│   │   ├── metaqa_3hop_semantic_path_projector_500.json
│   │   └── README.md
│   ├── accuracy_analysis_and_recovery_plan.md
│   └── metaqa_verified_assessment.md
├── scripts
│   ├── build_capsule_semantic_artifact.py
│   ├── build_route_outcomes.py
│   ├── evaluate_query_projection.py
│   ├── profile_optimized_latency.py
│   ├── profile_retrieval_latency.py
│   ├── repool_rgcn_artifact.py
│   ├── run_grag_evaluation.py
│   ├── start_batched_embedding_server.ps1
│   └── train_cost_aware_router.py
├── src
│   ├── adapters
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── hf_disk.py
│   │   │   ├── metaqa.py
│   │   │   ├── multihop_rag.py
│   │   │   ├── paths.py
│   │   │   ├── registry.py
│   │   │   └── wiki_multihop_qa.py
│   │   ├── embedding
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py
│   │   │   ├── cache.py
│   │   │   ├── llama_cpp.py
│   │   │   └── ollama.py
│   │   ├── llm
│   │   │   ├── __init__.py
│   │   │   ├── adapter.py
│   │   │   └── ollama.py
│   │   ├── storage
│   │   │   ├── graph_store
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── in_memory.py
│   │   │   │   └── neo4j.py
│   │   │   ├── vector_store
│   │   │   │   ├── __init__.py
│   │   │   │   ├── base.py
│   │   │   │   ├── in_memory.py
│   │   │   │   ├── numpy_store.py
│   │   │   │   └── qdrant.py
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── compute_eval
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── in_memory.py
│   │   ├── metrics.py
│   │   └── noop.py
│   ├── evaluation
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── core.py
│   │   ├── README.md
│   │   └── runner.py
│   ├── g_rag
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── context.py
│   │   ├── extractors.py
│   │   ├── instrumentation.py
│   │   ├── pipeline.py
│   │   ├── ranking.py
│   │   ├── README.md
│   │   └── types.py
│   ├── helpers
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   └── documents.py
│   ├── optimized_grag
│   │   ├── graph
│   │   │   ├── __init__.py
│   │   │   ├── algorithms.py
│   │   │   └── local.py
│   │   ├── graph_construction
│   │   │   ├── __init__.py
│   │   │   ├── builder.py
│   │   │   ├── fallbacks.py
│   │   │   ├── README.md
│   │   │   └── types.py
│   │   ├── graph_updates
│   │   │   ├── __init__.py
│   │   │   ├── README.md
│   │   │   ├── types.py
│   │   │   └── updater.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── projection_supervision.py
│   │   │   ├── README.md
│   │   │   ├── rgcn.py
│   │   │   └── structural.py
│   │   ├── notebooks
│   │   │   ├── projection
│   │   │   │   ├── 01_train_query_projection.ipynb
│   │   │   │   └── README.md
│   │   │   └── rgcn
│   │   │       ├── 01_train_capsule_rgcn.ipynb
│   │   │       └── README.md
│   │   ├── partitioning
│   │   │   ├── __init__.py
│   │   │   ├── anchors.py
│   │   │   ├── capsule_builder.py
│   │   │   ├── hubs.py
│   │   │   ├── pipeline.py
│   │   │   ├── README.md
│   │   │   ├── regions.py
│   │   │   ├── supergraph.py
│   │   │   ├── types.py
│   │   │   └── validation.py
│   │   ├── retrieval
│   │   │   ├── __init__.py
│   │   │   ├── capsule_index.py
│   │   │   ├── context.py
│   │   │   ├── path_search.py
│   │   │   ├── README.md
│   │   │   ├── router.py
│   │   │   ├── scoring.py
│   │   │   └── types.py
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── Idea.md
│   │   ├── pipeline.py
│   │   └── types.py
│   ├── rag
│   │   ├── bm25
│   │   │   ├── __init__.py
│   │   │   ├── index.py
│   │   │   └── pipeline.py
│   │   ├── hybrid
│   │   │   ├── __init__.py
│   │   │   ├── fusion.py
│   │   │   └── pipeline.py
│   │   ├── normal
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py
│   │   │   └── README.md
│   │   ├── __init__.py
│   │   ├── context.py
│   │   └── types.py
│   ├── rl_router
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── outcomes.py
│   │   └── policy.py
│   └── __init__.py
├── tests
│   ├── smoke_dataset_adapters.py
│   ├── smoke_graph_construction_updates.py
│   ├── smoke_in_memory_grag.py
│   ├── smoke_optimized_grag.py
│   ├── smoke_rag_variants.py
│   ├── test_accuracy_pipeline.py
│   ├── test_metaqa_retrieval_correctness.py
│   ├── test_ollama_embedding_adapter.py
│   ├── test_performance_regressions.py
│   ├── test_production_grag.py
│   └── test_rl_router.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── Run.md
└── to-check.md
```

# File Contents

## scripts/build_route_outcomes.py

```python
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

```

## scripts/train_cost_aware_router.py

```python
"""Train a three-action contextual-bandit router from measured route outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_router import RewardConfig, cheapest_acceptable_route, feature_names, route_reward, train_bandit


ACTIONS = ("hybrid_rag", "graph_rag", "optimized_grag")


def main() -> None:
    args = parse_args()
    outcomes = _read_jsonl(Path(args.outcomes))
    reward_config = RewardConfig(acceptable_quality=args.acceptable_quality)
    examples = _complete_examples(outcomes, reward_config=reward_config)
    split = max(1, int(len(examples) * (1.0 - args.validation_fraction)))
    train_examples, validation_examples = examples[:split], examples[split:]
    policy = train_bandit(
        train_examples,
        actions=ACTIONS,
        feature_names=feature_names(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    report = {
        "examples": len(examples),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "actions": ACTIONS,
        "reward_config": reward_config.to_dict(),
        "train": _evaluate(policy, train_examples),
        "validation": _evaluate(policy, validation_examples),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"policy": policy.to_dict(), "report": report}, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(output), "report": report}, indent=2))


def _complete_examples(outcomes: list[dict], *, reward_config: RewardConfig) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for outcome in outcomes:
        grouped[str(outcome["sample_id"])].append(outcome)
    examples = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        by_route = {str(row["route"]): row for row in rows}
        if set(by_route) != set(ACTIONS):
            continue
        for row in rows:
            row["reward"] = route_reward(
                float(row["quality_score"]),
                float(row["normalized_cost"]),
                config=reward_config,
            )
        oracle = cheapest_acceptable_route(rows, config=reward_config)
        examples.append(
            {
                "sample_id": sample_id,
                "features": dict(rows[0]["features"]),
                "rewards": {route: float(by_route[route]["reward"]) for route in ACTIONS},
                "oracle_route": str(oracle["route"]),
            }
        )
    if not examples:
        raise RuntimeError("No questions have a complete Hybrid/Graph/Optimized outcome set")
    return examples


def _evaluate(policy, examples: list[dict]) -> dict:
    if not examples:
        return {"route_accuracy": None, "mean_selected_reward": None, "mean_oracle_reward": None}
    selected_rewards = []
    oracle_rewards = []
    correct = 0
    for example in examples:
        selected = policy.choose(example["features"])
        correct += int(selected == example["oracle_route"])
        selected_rewards.append(example["rewards"][selected])
        oracle_rewards.append(max(example["rewards"].values()))
    return {
        "route_accuracy": correct / len(examples),
        "mean_selected_reward": sum(selected_rewards) / len(selected_rewards),
        "mean_oracle_reward": sum(oracle_rewards) / len(oracle_rewards),
        "mean_reward_regret": (sum(oracle_rewards) - sum(selected_rewards)) / len(examples),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cost-aware full-feedback contextual-bandit router.")
    parser.add_argument("--outcomes", default="artifacts/rl_router/outcomes.jsonl")
    parser.add_argument("--output", default="artifacts/rl_router/cost_aware_bandit.json")
    parser.add_argument("--acceptable-quality", type=float, default=0.65)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main()

```

