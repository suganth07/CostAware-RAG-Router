# 11. Query Latency and Storage Choices

## Why the old optimized run was slow

The screenshot's MetaQA 2-hop report used 500 questions. Its optimized path
search took 66.7 seconds in total, or 133.4 ms/query: 92.5% of optimized query
latency. Capsule vector storage was not the bottleneck.

The same report's normal side was not a production normal-RAG baseline. Its
index report says `documents=0`, `chunks=0`, and `vectors=0`; it only expanded
the authoritative graph. That explains its 0.249-second index time and much of
its apparently exceptional query speed.

## Current fair production comparison

The fast path now combines exact-seed membership routing, direct traversal of
the original graph, per-seed budgets, relation-first fan-out pruning, cached
neighbor ordering, and index-time lexical feature preparation.

The production normal index contains all 134,741 source facts and its run report
records `documents=134741`, `chunks=134741`, and `vectors=134741`. On the same
first 500 held-out MetaQA 2-hop test questions with a shared three-hop ceiling:

| Metric | Production normal | Current optimized |
|---|---:|---:|
| Mean query latency | 36.707 ms | 13.192 ms |
| P95 query latency | 43.404 ms | 23.026 ms |
| Context answer hit | 94.0% | 82.0% |
| Answer Recall@10 | 18.0% | 73.6% |
| Evidence MRR | 0.097 | 0.526 |
| Supporting endpoint recall | 89.3% | 54.6% |
| Mean selected capsules | 0 | 12 |
| Mean retrieved evidence units | 252.0 | 9.694 |

Optimized is 64.1% faster by mean latency, 46.9% faster at p95, and preserves
87.2% of normal context-answer recall. It also ranks answer-bearing evidence far
earlier (4.1x Recall@10 and 5.4x MRR), while the much broader normal context has
better exhaustive supporting-endpoint coverage. The fair combined report is
`reports/production_comparison_verified/metaqa_2hop_vanilla_test/`.

A second profiler executed all 500 queries live in one process, bypassing the
evaluation-record cache while reusing only immutable indexes/embeddings:

| Live warm profile | Production normal | Current optimized |
|---|---:|---:|
| Mean query latency | 30.740 ms | 8.031 ms |
| P95 query latency | 33.648 ms | 14.998 ms |
| Warm index/startup | 1.947 s | 10.961 s |

That same-process profile makes optimized 73.9% faster by mean and 55.4% faster
at p95. Optimized startup is still slower because capsules are reconstructed;
query savings amortize that one-time cost. The raw live profile is
`reports/latency/production_retrieval_comparison_500.json`.

The original screenshot is still useful as a regression clue, but not as the
final baseline: its normal index report contained zero documents, chunks, and
vectors. Even there, the unoptimized path enumerator—not capsule vector
storage—caused the optimized slowdown.

An instrumented 500-query run reports 9.465 ms total mean and attributes it as:

| Optimized stage | Mean |
|---|---:|
| Query embedding | 1.276 ms |
| Entity extraction | 0.039 ms |
| Capsule routing | 0.104 ms |
| Bounded path search | 7.760 ms |
| Context construction | 0.192 ms |

Stage timing varies with instrumentation and machine load. Use the same-run
13.192-vs-36.707 ms result above for the system claim; the stage profile is for
bottleneck attribution.

## Persistent normal index

The comparison notebook refuses to benchmark MetaQA normal GraphRAG with zero
vectors when document indexing is enabled. The production normal pipeline
embeds all 134,741 source facts, stores the normalized matrix under
`artifacts/cache/vector_indexes/`, retrieves dense candidates, reranks them with
lexical/entity signals, then performs capped graph expansion. Its first index
build is expensive; later runs memory-map the same persisted matrix.

The completed MetaQA matrix is now cached locally. Restart the notebook kernel
and rerun every cell so it memory-maps this index and cannot reuse old in-memory
configs.

## Would Qdrant make this faster?

Not at the present scale or bottleneck. The capsule index contains only 13,717
vectors; exact-seed questions now bypass its dense scoring entirely. Replacing
the local NumPy matrix with Qdrant would add serialization and service overhead
without accelerating the remaining 7.76 ms graph-path stage.

Qdrant becomes useful when there are hundreds of thousands or millions of
capsules, approximate search is acceptable, multiple processes need shared
persistent vectors, or metadata filtering/multi-tenant access dominates. It is
a deployment choice, not a fix for Python graph traversal.

## Reproduce the measurements

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_batched_embedding_server.ps1
python scripts/run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3 --max-questions 500 --embedding-backend llama_cpp --embedding-batch-size 128 --output-root reports/production_comparison_verified
python scripts/profile_retrieval_latency.py --dataset metaqa_2hop --system optimized_grag --questions 500 --max-hops 3 --embedding-backend llama_cpp --embedding-batch-size 128 --output reports/latency/optimized_exact_seed_precomputed_terms_500_profile.json
```

For the fair production comparison, first run the batched embedding server and
then execute the notebook or CLI commands in `Run.md`. Do not lower path budgets
without recording a quality ablation.
