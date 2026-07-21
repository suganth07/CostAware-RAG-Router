# Evaluation and Caching

`src.evaluation` is the reusable evaluation interface for every retriever.

`CachedEvaluationRunner` stores raw `RetrievalRecord` values and recomputes
metrics from those records. This keeps expensive retrieval/generation caching
separate from metric definitions.

MetaQA exposes explicit `metaqa_1hop`, `metaqa_2hop`, and `metaqa_3hop` targets.
They evaluate held-out test questions and use separate report/cache namespaces.
Use `--full-eval` for every question in the chosen test split; query embeddings
are precomputed in bounded batches and persisted before retrieval begins.
Evaluation records are appended every 25 completed questions, so a stopped
full run resumes from its last checkpoint instead of restarting the split.

## Metrics

- **Supporting-evidence recall:** fraction of annotated evidence items recovered.
- **Evidence Recall@K:** supporting-evidence recall within the first K ranked units.
- **Evidence MRR:** reciprocal rank of the first supporting-evidence unit.
- **Answer Recall@K:** whether a span answer appears in the first K units.
- **Context answer recall:** whether a span answer appears anywhere in context.
- **Exact match / token F1:** normalized generated-answer quality.
- **Boolean accuracy:** explicit Yes/No generation accuracy; context substring
  matches are never used for boolean questions.

MultiHop-RAG evidence uses evidence URLs/titles/facts. 2Wiki uses supporting
titles and evidence triples. MetaQA has no supporting-fact annotation, so gold
answer endpoints are used as graph-evidence proxies.

## Cache invalidation

The CLI fingerprints dataset content, selected questions, retrieval source code,
pipeline configuration, models, generation mode, and structural artifacts.
Changing any of them creates a new cache entry.
