# Verified MetaQA Assessment

Updated: 2026-07-16

## Current 2-hop production comparison

The current adaptive exact-seed configuration was evaluated on the first 500
held-out MetaQA 2-hop `test` questions with `--max-hops 3`, 12 capsules, 11
candidate paths, and 11 final paths:

| Metric | Production normal | Current optimized |
|---|---:|---:|
| Context answer recall | 94.0% | 82.0% |
| Answer Recall@10 | 18.0% | 73.6% |
| Evidence MRR | 0.097 | 0.526 |
| Supporting endpoint recall | 89.3% | 54.6% |
| Mean latency | 36.707 ms | 13.192 ms |
| P95 latency | 43.404 ms | 23.026 ms |
| Mean evidence units | 252.0 | 9.694 |

The normal run indexed every one of the 134,741 MetaQA facts into its persistent
source-vector matrix. Optimized is 64.1% faster by mean and 46.9% faster at p95,
while retaining 87.2% of normal context-hit accuracy. It sharply improves early
answer ranking and MRR; normal's much broader 252-unit context still retrieves
more total answer endpoints. The prior zero-vector normal plot is superseded.

An independent same-process live profiler measured normal/optimized means of
30.740/8.031 ms and p95 values of 33.648/14.998 ms: optimized was 73.9% faster
by mean and 55.4% faster at p95 without using cached retrieval records.

## Historical 100-question assessment (superseded latency configuration)

These are current retrieval-only checks over the first 100 held-out `test`
questions in each MetaQA slice. Both systems use the same `--max-hops 3` query
ceiling. Query embeddings are pre-cached, graph traversal is deterministic, and
optimized runtime reports `trained_semantic_projection`, 100% capsule ID match,
and matching fingerprints.

## End-to-end results

| Test slice / metric | Normal GraphRAG | Optimized GraphRAG |
|---|---:|---:|
| 1-hop context answer recall | 100.0% | 100.0% |
| 1-hop Answer Recall@1 | 98.0% | 95.0% |
| 1-hop supporting endpoint recall | 99.0% | 98.3% |
| 1-hop mean latency | 2.4 ms | 76.0 ms |
| 1-hop mean evidence units | 455.6 | 49.5 |
| 2-hop context answer recall | 100.0% | 95.0% |
| 2-hop Answer Recall@1 | 1.0% | 49.0% |
| 2-hop Answer Recall@20 | 89.0% | 85.0% |
| 2-hop evidence MRR | 0.179 | 0.535 |
| 2-hop supporting endpoint recall | 99.4% | 77.9% |
| 2-hop mean latency | 2.1 ms | 101.0 ms |
| 2-hop mean evidence units | 491.4 | 49.9 |
| 3-hop context answer recall | 57.0% | 84.0% |
| 3-hop Answer Recall@1 | 15.0% | 27.0% |
| 3-hop Answer Recall@20 | 49.0% | 57.0% |
| 3-hop evidence MRR | 0.217 | 0.341 |
| 3-hop supporting endpoint recall | 30.6% | 40.3% |
| 3-hop mean latency | 2.7 ms | 185.6 ms |
| 3-hop mean evidence units | 503.8 | 50.2 |

These rows used the former 500-candidate/50-final-path implementation. Keep them
as a historical quality comparison only; their optimized latency is not the
current runtime performance.

The strongest defensible end-to-end claim is on 3-hop MetaQA: optimized improves
context answer recall by 27 percentage points, Answer Recall@20 by 8 points,
MRR by 57%, and supporting endpoint recall by almost 10 points while returning about
one-tenth as many evidence units. On 2-hop it sharply improves early ranking and
MRR, but normal GraphRAG still has better exhaustive multi-answer coverage.
Normal remains the appropriate fast baseline for simple 1-hop questions.

## Why the first projector failed

The old projector mapped language into a 64-dimensional R-GCN space trained for
link reconstruction. That space represented graph topology but had no direct
semantic alignment with question text. Multi-positive loss alone could not fix
the target-space mismatch.

The earlier evaluator was also misaligned: it called every answer-containing
capsule positive. For multi-hop routing, the useful capsule is normally the one
containing the query seed and the first correct path edge, not the final answer.

## Corrected projector result

The current model uses:

- gold-path first-edge capsule labels;
- up to 16 positives and 20 hard negatives per question;
- cached `nomic-embed-text` capsule targets;
- an identity-residual MLP initialized to raw semantic retrieval;
- multi-positive log-sum-exp contrastive loss;
- early stopping on full-capsule any-positive MRR.

On 500 held-out questions per hop:

| Slice | Raw semantic MRR | Learned MRR | Learned Recall@10 | Random MRR |
|---|---:|---:|---:|---:|
| 1-hop | 0.2864 | 0.2873 | 31.7% | 0.00119 |
| 2-hop | 0.2643 | 0.3035 | 33.6% | 0.00135 |
| 3-hop | 0.3449 | 0.4079 | 45.5% | 0.00230 |

This is now a demonstrated learned advantage for multi-hop capsule ranking:
MRR improves 14.8% over raw semantic retrieval on 2-hop and 18.3% on 3-hop.
The 1-hop change is neutral, so no learning advantage should be claimed there.
R-GCN is still useful as a structural representation, but the evidence does not
support claiming that pure text-to-R-GCN projection is effective.

Canonical reports are under `reports/evaluation/`; standalone projector reports
and ablations are under `reports/projection_evaluation/`.
