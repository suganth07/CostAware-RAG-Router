# GraphRAG / Optimized GraphRAG Experiments

Research code for comparing normal GraphRAG with bounded, evidence-preserving
capsule GraphRAG on MetaQA, 2WikiMultiHopQA, and MultiHop-RAG.

The optimized path now enforces four retrieval invariants:

1. Every graph node belongs to at least one capsule.
2. Every linked query seed is represented in selected capsules and receives a
   fair path-search budget.
3. Exact seed neighborhoods are expanded with bounded, multi-depth search.
4. Source chunks are ranked against the query before they enter context.

Unified MetaQA retrieval uses an adaptive router: exact linked seeds go straight
to their known capsules, while unlinked/ambiguous questions use learned capsule
projection. It selects 12 capsules and keeps 11 query-ranked evidence paths
across depths 1–3. Runtime is never given a benchmark hop label.

On the verified 500-question MetaQA 2-hop production comparison, optimized
retrieval is 64.1% faster than normal by mean latency (13.19 vs 36.71 ms) while
retaining 87.2% of normal context-answer recall (82% vs 94%).

## Documentation

The complete layman-to-technical handbook starts at [docs/README.md](docs/README.md).
It covers architecture, algorithms, capsule construction, the R-GCN and unified
projector, retrieval, evaluation, caching, operations, worked examples,
troubleshooting, and configuration.

## Setup

```powershell
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
python -m pytest -q
```

Start Ollama before running model-backed evaluation.

For the first production-baseline build on a 6 GB NVIDIA GPU, start the bundled
batched embedding server in a separate PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_batched_embedding_server.ps1
```

Then add `--embedding-backend llama_cpp --embedding-batch-size 128` to the
evaluation command. The helper reuses Ollama's local model and CUDA runtime;
it does not download another model. It also disables per-request server logs,
which otherwise become a major bulk-indexing bottleneck.

## Recommended run order

### 1. Establish retrieval baselines

Run retrieval-only evaluation first. This measures evidence recall without LLM
answer-style noise.

```powershell
python scripts\run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3
python scripts\run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3
python scripts\run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3
python scripts\run_grag_evaluation.py --dataset multihop_rag --system both
python scripts\run_grag_evaluation.py --dataset 2wiki --system both
```

For a fast smoke run, add `--max-questions 5`.

The default MetaQA run evaluates 500 held-out test questions. For final reported
numbers over every test question, use `--full-eval`. See `Run.md` for the exact
1-hop, 2-hop, and 3-hop sequence.

### 2. Inspect reports

Each run writes to `reports/evaluation/<dataset-artifact-name>/`:

- `summary_metrics.json`: aggregate retrieval and answer metrics.
- `query_metrics.csv`: per-question failure analysis.
- `retrieval_records.jsonl`: raw ranked evidence, context, and answers.
- `run_config.json`: data, source, model, artifact, and config fingerprints.

Key retrieval metrics are supporting-evidence recall, evidence Recall@1/3/5/10/20,
answer Recall@K, and evidence MRR. Retrieval-only runs leave EM/F1 and boolean
accuracy empty because no answer was generated.

### 3. Retrain structural artifacts

Old R-GCN/projector artifacts are intentionally rejected because they predate
deterministic capsule fingerprints.

For each graph dataset:

1. Open `src/optimized_grag/notebooks/rgcn/01_train_capsule_rgcn.ipynb`.
2. Run setup, exactly one dataset block, then every remaining cell.
3. Build cached capsule semantics with `scripts/build_capsule_semantic_artifact.py`.
4. Open `src/optimized_grag/notebooks/projection/01_train_query_projection.ipynb`.
5. Select the same dataset block and run every remaining cell.
6. Repeat for the next graph dataset. MetaQA uses one mixed-hop projector.

MetaQA 1/2/3-hop share one knowledge graph, deterministic capsule build, R-GCN
structural space, and unified query projector. Train the MetaQA R-GCN once and
the mixed-hop projector once. Hop labels only choose held-out evaluation files;
they are never passed to runtime retrieval. Final evaluation uses `test`
questions while projector training uses a shuffled mixture of all three `train`
question sets.

The updated R-GCN notebook exports the exact projection manifest. If an artifact
was trained before this manifest was added, rerun the R-GCN notebook once; the
projection notebook will clearly reject the legacy artifact.

If compatible relation-aware weights already exist, repool them without
retraining:

```powershell
python scripts\repool_rgcn_artifact.py --artifact-name metaqa_shared_graph_train
```

R-GCN validation now excludes validation edges from message passing, uses inverse
message relations and a relation-aware DistMult decoder, and reports MRR plus
Hits@1/3/10. Projection training uses gold-path launch capsules, semantic target
vectors, multi-positive hard negatives, and reports full-capsule
Recall@1/5/10/20 plus MRR.

R-GCN relation aggregation is grouped and message-batched, so it no longer
materializes an `[edges, 64, 64]` tensor. `MESSAGE_BATCH_SIZE=16384` is the
6 GB GPU default in the notebook; lower it if another process is sharing the
GPU. Projection validation also scores capsules in batches of 2,048.

### 4. Re-run retrieval evaluation

```powershell
python scripts\run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3
python scripts\run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3
python scripts\run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3
python scripts\run_grag_evaluation.py --dataset multihop_rag --system both
python scripts\run_grag_evaluation.py --dataset 2wiki --system both
```

Artifact hashes are part of the cache key, so retraining automatically causes a
fresh optimized evaluation.

### 5. Run end-to-end answer evaluation

```powershell
python scripts\run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3 --run-generation
python scripts\run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3 --run-generation
python scripts\run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3 --run-generation
python scripts\run_grag_evaluation.py --dataset multihop_rag --system both --run-generation
python scripts\run_grag_evaluation.py --dataset 2wiki --system both --run-generation
```

This adds normalized exact match, token F1, and separate boolean accuracy.

## Caching

Three persistent caches avoid repeated work:

- `artifacts/cache/embeddings.sqlite3` caches embeddings by model namespace and
  text hash. Ollama and the bundled llama.cpp server share the verified
  `local:nomic-embed-text:mean_pooling_v1` semantic namespace, so switching
  transport does not re-embed the corpus.
- `artifacts/cache/vector_indexes/` stores the normal pipeline's contiguous,
  normalized source-evidence matrix and metadata. The full MetaQA corpus is
  embedded once and then memory-mapped on later runs.
- `artifacts/cache/evaluations/` caches raw retrieval/generation records by
  dataset, code, config, model, and structural-artifact fingerprints. A fully
  cached evaluation skips indexing and model calls.

Use `--refresh-cache` only when intentionally rerunning identical inputs. Normal
code, config, data, model, or artifact changes invalidate the appropriate cache
automatically.

Evaluation defaults to one question worker because concurrent local-model calls
can exhaust a 6 GB GPU. Increase `--max-workers` only after measuring VRAM. For
Ollama use `--embedding-batch-size 8` (or 4) after an OOM. The batched server's
measured 6 GB setting is 128 sequences. Lower `-Parallelism` to 64 only if the
GPU is shared; it uses less VRAM but was slower on the RTX 4050.
With matching R-GCN and projector artifacts, optimized indexing searches in the
trained capsule space and skips redundant embedding of every capsule summary.

## Notebook use

`notebooks/01_compare_grag_vs_optimized.ipynb` remains available for exploratory
plots and uses the same embedding/evaluation caches. The CLI is the canonical
reproducible evaluation path.
