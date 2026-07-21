# Optimized Query-Latency Profiles

## Fair production comparison (2026-07-16)

On the same 500 held-out MetaQA 2-hop test questions, with the normal pipeline's
full 134,741-vector source index present:

| Metric | Normal | Optimized |
|---|---:|---:|
| Mean latency | 36.707 ms | 13.192 ms |
| P95 latency | 43.404 ms | 23.026 ms |
| Context answer recall | 94.0% | 82.0% |

Optimized is 64.1% faster by mean, 46.9% faster at p95, and retains 87.2% of
normal context-answer recall. The combined report is under
`reports/production_comparison_verified/metaqa_2hop_vanilla_test/`.

The uncached live profiler then ran both retrievers in one process over the same
500 questions:

| Live profile | Normal | Optimized |
|---|---:|---:|
| Mean latency | 30.740 ms | 8.031 ms |
| P95 latency | 33.648 ms | 14.998 ms |
| Warm index/startup | 1.947 s | 10.961 s |

This is a 73.9% mean and 55.4% p95 query-latency reduction. The raw result is
`production_retrieval_comparison_500.json`. Startup remains slower on optimized
because deterministic capsules are reconstructed once per process.

## Optimized stage profile

On 500 held-out MetaQA 2-hop test questions, the fast exact-seed path measured
9.465 ms mean, 9.100 ms p50, and 18.139 ms p95 with stage instrumentation.
Path search was 7.760 ms/query and capsule routing only 0.104 ms/query. The raw
profile is `optimized_exact_seed_precomputed_terms_500_profile.json`.

An earlier evaluation runner pass, without per-stage instrumentation and under
lighter machine load, measured 7.144 ms mean and 14.259 ms p95. Use the combined
13.192 ms result above for the fair same-run system comparison.

## Historical profile (2026-07-12)

The former 500-candidate implementation took roughly 118 ms mean on the first
30 MetaQA 3-hop questions, with path enumeration dominant. Those raw JSON files
are historical optimization evidence, not the current runtime configuration.
Timing varies with machine load; compare identical warm query sets and inspect
the full index contract before drawing system-level conclusions.
