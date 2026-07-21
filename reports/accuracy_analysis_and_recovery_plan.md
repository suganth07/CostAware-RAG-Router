# GraphRAG Accuracy Analysis and Recovery Plan

## Implementation update

July 2026 benchmark protocol update: MetaQA is now evaluated as three separate
held-out targets (`metaqa_1hop`, `metaqa_2hop`, and `metaqa_3hop`). One unified
query projector trains on a shuffled mixture of all three train splits; final
reports use separate test slices. The shared MetaQA KB/capsule R-GCN and unified
projector are reused for every question without a runtime hop label. Multi-hop retrieval uses bounded exact seed
expansion, relation-aware edge scoring, depth-balanced beam search, and reserves
half of final evidence capacity for the requested maximum reasoning depth.

All generated legacy comparison reports, train-split evaluation reports, stale
evaluation caches, unversioned MetaQA embeddings, and incompatible legacy
2Wiki/MultiHop-RAG artifacts were removed. The historical numbers below remain
only as diagnosis of the original failure and must not be reported as current
results.

The first accuracy-recovery phase is now implemented:

- complete capsule node coverage with bounded fallback ego capsules
- one reserved capsule and a fair path budget per linked query seed
- bounded exact multi-depth seed neighborhoods across capsule boundaries
- query-ranked source chunks and evidence-first context ordering
- 100/10/200/12 candidate/capsule/path/final-path defaults
- evidence Recall@K, MRR, EM/F1, and separate boolean accuracy
- persistent embedding and evaluation caches with fingerprints
- relation-aware, leakage-safe R-GCN training metrics
- evidence-aware projector mining and retrieval validation metrics

A deterministic full-graph MetaQA verification with local hash embeddings reached
50/50 context-answer hits, 100% linked-seed coverage, and 100% graph-node capsule
coverage. Full capsule indexing took about 12 seconds and mean query retrieval
about 0.22 seconds on this workspace. Official Ollama-backed reports still need to
be regenerated using the documented CLI.

## Bottom line

The old MetaQA comparison is invalid as an accuracy benchmark. It indexed the
first 5,000 triples from a 134,741-triple KB but evaluated an independent slice
of questions. Only 10 of the first 50 questions had even one gold answer in the
indexed slice, so the 10% normal / 4% optimized context-hit rates were dominated
by absent evidence.

The corrected setup loads the complete MetaQA KB and indexes its native graph
without redundantly embedding every one-line triple document. Gold-answer
coverage is 50/50 for the first 50 questions, and a direct native-graph
regression check retrieves a gold answer for 50/50. The notebook now rejects a
MetaQA run below 99% corpus answer coverage.

The old saved metrics must be treated as stale and regenerated after the
corrected optimized index and trained artifacts are rebuilt.

## Evidence from the existing reports

### MetaQA

| Measure | Old value |
|---|---:|
| KB triples available | 134,741 |
| KB triples indexed | 5,000 |
| First-50 questions with any gold answer in indexed documents | 10/50 (20%) |
| Normal context hits | 5/50 (10%) |
| Optimized context hits | 2/50 (4%) |
| Normal recall among answer-covered questions | 5/10 (50%) |
| Optimized recall among answer-covered questions | 2/10 (20%) |

The primary failure was corpus construction. The secondary optimized failure
was routing and evidence preservation.

### MultiHop-RAG

The reported 77.5% normal hit rate is inflated because 20/40 questions have a
gold answer of `Yes` or `No`, and the old metric searched for those literal
strings inside the retrieved context. The current notebook excludes those from
the span metric.

For the 13 answer-bearing inference questions, all 13 answers exist in the
corpus. Normal GraphRAG retrieved 12/13 (92.3%); optimized GraphRAG retrieved
2/13 (15.4%). This is the cleanest evidence that optimized retrieval is losing
answer-bearing evidence.

The optimized system reduced mean context from 9,487 to 1,705 characters and
selected only three capsules/five paths in the old run. That 82% context
reduction was too aggressive for the current router and path ranker.

## RGCN and projector health

### What looks healthy

- All three exported RGCN embedding files are finite and L2-normalized.
- Capsule dimensions are consistent at 64.
- RGCN validation loss fell materially:
  - MetaQA: 0.6463 to a best 0.3468.
  - MultiHop-RAG: 0.5477 to a best 0.2707.
- MetaQA projector separation is positive: final validation positive cosine
  0.1647 versus negative cosine -0.1525.

### What is not healthy enough to trust

- The MetaQA RGCN was trained on 20,000 KB triples while the old comparison
  rebuilt capsules from 5,000. Only 1 of 719 capsules in a diagnostic rebuild
  matched an exported capsule ID (0.14%). The artifact was effectively unused
  or was mixing incompatible spaces.
- Capsule construction used unordered set traversal, so IDs could change across
  processes. This is now deterministic, but old artifacts must be retrained.
- MultiHop-RAG projector validation loss was exactly zero in 7/8 epochs. Its
  negatives are too easy, so the loss does not demonstrate useful top-k routing.
- Mean random-pair cosine for capsule embeddings is high (about 0.57-0.59),
  suggesting partial over-smoothing even though embeddings have not collapsed.
- RGCN validation currently leaks validation edges into message passing, uses a
  relation-agnostic bilinear decoder, resamples validation negatives, and does
  not report filtered MRR or Hits@K.
- Projector positives are weakly mined by token overlap and negatives are sampled
  from ranks 6-50. This can label a query-token capsule as positive even when it
  does not contain supporting evidence.
- The 2Wiki artifact has no projector, so structural query routing is incomplete
  for that dataset.

Runtime now measures artifact-to-runtime capsule match, disables the trained
space below 95% match, and uses a structural fallback instead of zero vectors.

## Root causes in optimized GraphRAG

1. **No hard capsule coverage invariant.** Anchor-based capsules can leave graph
   nodes uncovered. A query entity can exist in the graph but in no capsule.
2. **Semantic reranking could discard exact entity matches.** Exact linked query
   seeds are now preserved ahead of merely similar capsules.
3. **Evidence is discarded inside a selected capsule.** A capsule may contain
   many chunk nodes, but only the first three snippets are rendered. Correct
   routing therefore does not guarantee that answer-bearing text reaches the
   context.
4. **Path budget is consumed by the first seed.** Query extraction includes
   generic entities such as `Who`/`Which`; the first seed can generate the full
   candidate budget before useful seeds are searched.
5. **Path scoring is query-token overlap plus shortness.** It has no evidence
   relevance, relation intent, source quality, or endpoint-answer likelihood.
6. **The learned structural score is not validated as a retrieval feature.**
   Training loss is being used as a proxy for routing quality, but Recall@K and
   MRR are not measured.
7. **Old budgets over-compressed retrieval.** Three capsules and five final paths
   are not enough while routing/path ranking remain weak.

## Architecture candidates

### 1. Evidence-preserving capsule retriever (strongest recommendation)

- **Files:** `retrieval/capsule_index.py`, `retrieval/router.py`,
  `retrieval/path_search.py`, `retrieval/context.py`, partitioning types.
- **Problem:** capsule selection, local path search, and source-text selection are
  separate shallow decisions with no invariant that selected evidence reaches
  the LLM.
- **Solution:** expose one deeper retrieval interface that guarantees exact-seed
  inclusion, node coverage, per-seed path quotas, and query-ranked source chunks
  from selected capsules.
- **Benefits:** directly targets answer recall; keeps correctness knowledge and
  tests local; permits adaptive budgets without changing callers.
- **Risk:** context and latency increase until ranking is tuned.

### 2. Versioned structural artifact bundle

- **Files:** training notebooks, `models/structural.py`, capsule construction.
- **Problem:** vectors are keyed by unstable/config-dependent capsule IDs without
  a graph/config fingerprint.
- **Solution:** export graph fingerprint, partition config, capsule manifest,
  model metrics, and projector metrics as one versioned artifact; fail closed on
  mismatch.
- **Benefits:** prevents silent invalid experiments and makes RGCN deployment
  reproducible.
- **Risk:** all old artifacts become intentionally incompatible and need retraining.

### 3. Evaluation harness by question type

- **Files:** comparison notebook plus a reusable evaluation module.
- **Problem:** answer coverage, span retrieval, boolean generation, null answers,
  and end-to-end QA are mixed into one substring metric.
- **Solution:** separate corpus coverage, supporting-evidence recall, span
  ContextRecall@K, boolean accuracy, null accuracy, EM/F1, and latency/compute.
- **Benefits:** optimization decisions become trustworthy and regressions fail
  early.
- **Risk:** historical headline numbers will no longer be directly comparable.

## Prioritized implementation plan

### Phase 0 - Re-establish valid baselines

1. Regenerate MetaQA with the complete native graph and no triple-document
   embeddings.
2. Add hard report gates:
   - MetaQA answer coverage >= 99%.
   - runtime capsule/artifact match >= 95% before enabling trained space.
   - graph-node capsule coverage >= 99.5%.
3. Report span retrieval separately from boolean/null generation metrics.
4. Save per-query selected capsule IDs, exact-seed coverage, source chunk IDs,
   path scores, and failure reason.

Acceptance: MetaQA native baseline context hit >= 95%, and every reported run
passes the coverage gates.

### Phase 1 - Highest-return optimized retrieval changes

1. Guarantee every graph node belongs to at least one capsule; add small fallback
   ego capsules for uncovered nodes.
2. Keep exact linked query-entity capsules in the selected set (partly implemented).
3. Filter generic interrogative entities and allocate path candidates per seed so
   one noisy seed cannot consume the whole budget.
4. Within selected capsules, rank source chunks against the query and include the
   best 3-5 snippets per capsule instead of the first three by storage order.
5. Increase the working budget while ranking improves:
   - first-stage candidates: 80-120,
   - selected capsules: 8-12,
   - candidate paths: 150-250,
   - final paths: 10-20,
   - context: 6k-10k characters.
6. Add relation-intent and source-evidence features to path scoring.

Acceptance targets: MetaQA context hit >= 90% for optimized retrieval and
MultiHop inference context hit >= 70%, while remaining at least 5x faster than
the old normal GraphRAG query path.

### Phase 2 - Retrain RGCN and projector correctly

1. Rebuild deterministic capsules from the exact full-corpus/config used at
   runtime and export a fingerprinted manifest.
2. Train RGCN with train-only message-passing edges for validation, inverse edge
   types, a relation-aware decoder (DistMult/ComplEx), filtered corruptions, and
   fixed validation negatives.
3. Evaluate RGCN using filtered MRR and Hits@1/3/10, not loss alone.
4. Mine projector positives from gold supporting evidence/source chunks and use
   in-batch plus mined hard negatives from the text retriever.
5. Split projector data by question before mining and evaluate capsule Recall@K,
   MRR, and exact supporting-capsule recall.
6. Tune structural weight by ablation; retain it only if it improves held-out
   Recall@K over text + entity routing.

Acceptance: capsule artifact match = 100%, projector Recall@10 materially beats
text/entity routing alone, and no near-zero validation loss from trivial negatives.

### Phase 3 - Recover compute after accuracy

1. Replace quadratic capsule-supergraph construction with inverted indexes over
   shared nodes/source IDs.
2. Use adaptive budgets: 1-hop entity questions get exact graph expansion;
   multi-hop questions get more capsules and longer paths.
3. Cache capsule embeddings, query-independent source rankings, and graph
   fingerprints.
4. Run ablations for text-only, entity-only, text+entity, +RGCN, +path reranker,
   and full system on fixed question sets.

The recommended order is Phase 0, then candidate 1, then candidate 2. Increasing
compute alone will not fix the current evidence-loss mechanisms; modestly larger
budgets should be used together with seed guarantees and source-chunk reranking.
