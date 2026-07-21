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
