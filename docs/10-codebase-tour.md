# 10. Codebase Tour and Extension Points

## Layered structure

```text
src/
  adapters/          external datasets, embedding/LLM, graph/vector storage
  helpers/           documents, chunks, normalization, chunking
  g_rag/             conventional GraphRAG baseline
  optimized_grag/    graph construction, capsules, models, retrieval, updates
  evaluation/        records, metrics, fingerprints, persistent caches
  compute_eval/      stage timing and operation counters
scripts/
  run_grag_evaluation.py
  repool_rgcn_artifact.py
notebooks/
  comparison and diagnostics
docs/
  this handbook
```

## Core interfaces

### Dataset adapters

A dataset adapter returns a `DatasetBundle` containing documents, evaluation
samples, and optionally native graph nodes/edges. Adding a dataset means
implementing the adapter interface and registering it in
`src/adapters/datasets/registry.py`.

### Embedding and LLM adapters

`EmbeddingAdapter` exposes `embed_texts` and a one-text convenience call.
`OllamaEmbeddingAdapter` performs bounded HTTP batches;
`LlamaCppEmbeddingAdapter` uses the OpenAI-compatible endpoint for true
multi-sequence bulk indexing. The caching wrapper is content-addressed and can
wrap either implementation without changing a RAG pipeline. `LLM` similarly
hides the generation provider.

### Storage adapters

`GraphStore` handles node/edge upserts and bounded neighborhood queries.
`VectorStore` handles vector-document upserts, existing-ID checks, flushes, and
similarity search. The production comparison baseline uses `NumpyVectorStore`:
contiguous normalized float32 arrays, partial top-K exact cosine search, atomic
persistence, and memory-mapped reloads. The optimized capsule path retains its
small in-memory index. Qdrant or another service can implement the same contract
when multi-process durability or ANN scale is actually needed.

## Important data objects

- `Document` and `Chunk`: source text plus provenance.
- `GraphNode` and `GraphEdge`: typed graph facts.
- `Region` and `GraphCapsule`: offline hierarchy.
- `CapsuleCandidate`: all component scores for a routed capsule.
- `EvidencePath`: ordered nodes/edges, rendered text, score, and seed metadata.
- `SourceSnippet`: ranked source text and provenance.
- `RetrievalRecord`: provider-independent evaluation representation.

## Instrumentation

Both pipelines wrap stages with `ComputeTracker`. Timings distinguish indexing,
query embedding, vector/capsule routing, path search, context building, and
generation. This separation exposed the old Python capsule-cosine bottleneck
and the redundant summary-embedding index cost.

## Graph updates

`GraphUpdateManager` accepts new documents or explicit nodes/edges, updates local
and configured stores, embeds new source vectors, and rebuilds capsules. The
current implementation performs a full capsule rebuild to preserve deterministic
coverage and fingerprints. Incremental region-only rebuilding is a future
optimization and must preserve the same artifact safety rules.

## Model extension points

`StructuralEncoder` requires two operations:

```python
encode_capsule(capsule) -> list[float]
project_query(query_vector) -> list[float]
```

`FeatureStructuralEncoder` is the safe handcrafted fallback.
`TrainedStructuralEncoder` loads R-GCN vectors plus an optional projector and
enforces fingerprints. A different GNN or projector can be introduced behind
this interface without changing capsule routing.

## Retrieval extension points

- Replace entity extraction while keeping seed IDs stable.
- Add a learned cross-encoder after the 100-capsule first stage.
- Improve query-relation parsing used by path search.
- Replace the beam score while retaining seed/depth fairness.
- Add an evidence-sufficiency stopping policy above the fixed depth ceiling.
- Replace in-memory matrices with an ANN index when capsule count becomes large.

Every change should be evaluated with the existing evidence metrics, latency,
coverage invariants, and artifact fingerprints rather than only answer examples.
