# 6. Evaluation, Metrics, and Caching

## Honest benchmark protocol

MetaQA uses one runtime model but three held-out report slices:

| CLI target | Test questions | Model artifact |
|---|---:|---|
| `metaqa_1hop` | 9,947 | `metaqa_unified_vanilla_train` |
| `metaqa_2hop` | 14,872 | same |
| `metaqa_3hop` | 14,274 | same |

The target chooses data to score. It does not alter the runtime retriever.
Default runs sample 500 test questions; `--full-eval` uses the complete split.
The report records `retrieval_max_hops` and both resolved runtime configs. Cache
fingerprints include those configs, so results from different caps cannot mix.

## Retrieval metrics

- **Context answer recall:** at least one normalized gold span appears anywhere
  in final context.
- **Answer Recall@K:** a gold span appears in the first K evidence units.
- **Supporting-evidence recall:** fraction of annotated evidence items retrieved.
  MetaQA has no fact annotations, so answer endpoints are evidence proxies.
- **Evidence Recall@K:** evidence recall restricted to the first K units.
- **Evidence MRR:** reciprocal rank of the first matching evidence unit.
- **Capsule Recall/MRR:** available when relevant capsule IDs are annotated.

## Standalone learned-projector evaluation

`scripts/evaluate_query_projection.py` evaluates capsule routing separately
from entity linking, path search, and final evidence ranking. For MetaQA, a gold
capsule contains the first edge of a valid path from the bracketed query seed to
any answer. This measures the launch decision the router actually controls,
instead of requiring a capsule to contain a several-hop-away answer. For every
held-out query it computes the best rank of any gold capsule:

- any-positive MRR;
- any-positive Recall@1/5/10/20;
- mean best-positive rank and number of valid positive capsules;
- analytical random-ranker MRR and Recall@K baselines;
- label coverage.

Use `--scoring-mode raw_semantic` for the unlearned semantic baseline. Comparing
that result with the default residual projector isolates the gain from learning;
comparing only with random would overstate the model contribution.

The random baseline uses the actual number of valid path capsules. Reports are fingerprinted over the test
questions, evaluator/mining code, embedding model, and both model artifacts; an
unchanged rerun reuses the JSON result unless `--refresh` is supplied.

## Generation metrics

- **Exact match:** normalized prediction equals a gold answer.
- **Token F1:** token overlap precision/recall harmonic mean.
- **Boolean accuracy:** parses a generated yes/no answer separately.

Boolean context substring matching is intentionally forbidden: unrelated context
often contains “yes” or “no” and would inflate retrieval accuracy.

## Why report several metrics

Consider a question with ten valid answers. Context answer recall can be 100%
after retrieving only one. Supporting-evidence recall reveals that nine endpoints
are still missing. Recall@K shows whether useful evidence appears early enough
for a context-limited LLM.

## Cache layers

### Embedding cache

SQLite stores vectors by model namespace and SHA-256 text hash. Full evaluation
precomputes question embeddings in bounded batches, avoiding tens of thousands
of one-question model calls. The namespace represents embedding semantics rather
than API transport. The local Ollama and llama.cpp `nomic-embed-text` outputs
were checked over 500 shared texts (minimum cosine above 0.99998), so both use
`local:nomic-embed-text:mean_pooling_v1` and do not recompute the same corpus
when the transport changes.

### Evaluation cache

Raw retrieval/generation records are JSONL. The fingerprint includes:

- dataset content and selected questions;
- source code and configuration;
- embedding/LLM model and generation mode;
- structural and projector artifact bytes.

Records checkpoint every 25 completed questions. A killed full run resumes from
the last checkpoint; a partial final JSON line is ignored safely. Metric code can
be changed and recomputed from cached raw records.

### Normal source-vector index

Normal GraphRAG persists `vectors.npy`, `documents.jsonl`, and a versioned
manifest under a corpus/model/chunk fingerprint in
`artifacts/cache/vector_indexes/`. Vectors are exact normalized float32 values;
completed indexes are memory-mapped rather than rebuilt into Python float
objects. Embedding-cache checkpoints protect the long first build, while the
index itself is written atomically only after every source is present.

The evaluation fingerprint still records the selected backend, so benchmark
reports remain auditable even though the validated vector cache is shared.

## Reading results correctly

Always report:

1. dataset split and question count;
2. whether generation was enabled;
3. context/evidence recall and Recall@K;
4. EM/F1 or boolean accuracy when generated;
5. mean and tail latency;
6. index time separately from query time;
7. trained-space enabled status and capsule match rate.
