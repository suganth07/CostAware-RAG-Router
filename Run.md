# Clean Run Order

## 0. Verify the code

```powershell
python -m pytest -q
```

## 0.5. Start the optional high-throughput embedding server

The first production normal-GraphRAG run embeds all 134,741 MetaQA source
facts. Ollama remains the default and needs no extra setup, but its embedding
runner processes this corpus almost serially. On a 6 GB NVIDIA GPU, open a
second PowerShell terminal and run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_batched_embedding_server.ps1
```

Leave that terminal open. Add the following flags to evaluation/profile
commands while it is running:

```text
--embedding-backend llama_cpp --embedding-batch-size 128
```

The first run checkpoints embeddings after every 2,048 texts and persists the
completed exact-cosine NumPy index under `artifacts/cache/vector_indexes/`.
Interrupted runs resume from the embedding cache; later hop/test runs reuse the
same memory-mapped source index. The measured RTX 4050 default is 128 server
slots. If another GPU application is open, restart it with `-Parallelism 64`.
Ollama and this server use one verified semantic cache namespace, so changing
the local transport does not discard already-computed `nomic-embed-text`
vectors.

## 1. Train structural models

Open `src/optimized_grag/notebooks/rgcn/01_train_capsule_rgcn.ipynb`.

- MetaQA: the existing `metaqa_shared_graph_train` R-GCN is shared by 1/2/3-hop.
  Do not retrain it unless its folder is missing or you intentionally change the
  capsule-construction configuration.
- 2Wiki and MultiHop-RAG: select each dataset cell and train once. Their removed
  legacy artifacts were incompatible with deterministic capsule fingerprints.

Run setup, exactly one dataset cell, then every remaining cell. Restart the
kernel before selecting another dataset.

If compatible R-GCN weights already exist but the v4 membership manifest is
missing, repool without retraining:

```powershell
python scripts/repool_rgcn_artifact.py --artifact-name metaqa_shared_graph_train
```

## 2. Build semantic capsule targets

This is persistent and resumable through the embedding cache:

```powershell
python scripts/build_capsule_semantic_artifact.py --embedding-batch-size 8
```

On a 6 GB GPU, keep the batch at 8 or lower. An unchanged rerun is a cache hit.

## 3. Train the unified MetaQA query projector

Open `src/optimized_grag/notebooks/projection/01_train_query_projection.ipynb`,
run the MetaQA unified selection cell, then every remaining cell once. It mixes
1/2/3-hop training questions and exports one projector to
`artifacts/rgcn/metaqa_unified_vanilla_train/`.

There is no hop setting at inference. The three evaluation commands below choose
different held-out files only; all load the same model and runtime configuration.

## 4. Evaluate the projector by itself

Run this after projector training. It uses held-out test questions and reports
any-positive capsule MRR/Recall plus a random-ranker baseline. Results are
fingerprinted and cached under `reports/projection_evaluation/`.

```powershell
python scripts/evaluate_query_projection.py --dataset metaqa_1hop --max-questions 500
python scripts/evaluate_query_projection.py --dataset metaqa_2hop --max-questions 500
python scripts/evaluate_query_projection.py --dataset metaqa_3hop --max-questions 500
```

Run the raw semantic baseline for the learning ablation:

```powershell
python scripts/evaluate_query_projection.py --dataset metaqa_1hop --max-questions 500 --scoring-mode raw_semantic
python scripts/evaluate_query_projection.py --dataset metaqa_2hop --max-questions 500 --scoring-mode raw_semantic
python scripts/evaluate_query_projection.py --dataset metaqa_3hop --max-questions 500 --scoring-mode raw_semantic
```

Use `--refresh` only when you intentionally want to recompute unchanged inputs.

## 5. Smoke-test all MetaQA hops

```powershell
python scripts/run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3 --max-questions 50
python scripts/run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3 --max-questions 50
python scripts/run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3 --max-questions 50
```

Optional warm stage-level latency profile:

```powershell
python scripts/profile_retrieval_latency.py --dataset metaqa_3hop --system both --questions 30
```

When using the batched server, add the backend flags from step 0.5. The profiler
measures both complete retrieval contracts after a warm-up query; indexing time
is reported separately.

The verified fast MetaQA defaults are 11 candidate paths and 11 final paths.
Exact linked seeds use direct bounded graph traversal, so materializing hundreds
of speculative paths is no longer necessary. Use `--max-candidate-paths` or
`--final-paths` only for recorded ablations and recheck retrieval quality.

## 6. Run the actual held-out MetaQA test sets

These use all 9,947 1-hop, 14,872 2-hop, and 14,274 3-hop test questions.
Embeddings and completed evaluation records are cached, so interrupted runs can
be restarted without repeating finished work.

```powershell
python scripts/run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3 --full-eval
python scripts/run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3 --full-eval
python scripts/run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3 --full-eval
```

Reports are written separately to:

```text
reports/evaluation/metaqa_1hop_vanilla_test/
reports/evaluation/metaqa_2hop_vanilla_test/
reports/evaluation/metaqa_3hop_vanilla_test/
```

`--max-hops 3` is one shared query-time ceiling. Normal GraphRAG uses it as
`graph_depth`; optimized GraphRAG uses it as both evidence-path length and exact
seed-expansion depth. Use `--max-hops 1` or `2` for controlled ablations; never
give the systems different caps in a comparison.

## 7. Evaluate the other datasets

After training both their R-GCN and matching projector:

```powershell
python scripts/run_grag_evaluation.py --dataset multihop_rag --system both
python scripts/run_grag_evaluation.py --dataset 2wiki --system both
```

For a 6 GB GPU, keep the default `--max-workers 1`. If Ollama embeddings run out
of memory, add `--embedding-batch-size 8` or `--embedding-batch-size 4`.
