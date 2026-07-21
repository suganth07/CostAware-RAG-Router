# 7. Operations Runbook

The concise command list lives in the repository root `Run.md`. This guide
explains why the order matters.

## Environment check

```powershell
python -m pytest -q
```

Start Ollama before projector training or model-backed evaluation. Keep
`--max-workers 1` on a 6 GB GPU.

For the first production normal-GraphRAG corpus build, start
`scripts/start_batched_embedding_server.ps1` in another terminal and add
`--embedding-backend llama_cpp --embedding-batch-size 128` to the evaluation
commands. The measured helper default is 128 server slots and remains below
6 GB on the RTX 4050. Later runs memory-map the completed source-vector index.

## MetaQA training order

### 1. Shared R-GCN

The existing shared structural files under
`artifacts/rgcn/metaqa_shared_graph_train/` remain valid. Retrain only if the
graph/capsule configuration changes or those files are missing.

Notebook: `src/optimized_grag/notebooks/rgcn/01_train_capsule_rgcn.ipynb`

### 2. Unified projector

First build the cached semantic capsule targets:

```powershell
python scripts/build_capsule_semantic_artifact.py --embedding-batch-size 8
```

Notebook: `src/optimized_grag/notebooks/projection/01_train_query_projection.ipynb`

Run setup, the MetaQA unified selection cell, and all remaining cells once. Do
not choose a hop. Expected inventory is 5,000 questions per slice; path mining
currently yields about 13,400 multi-positive examples.

## Smoke evaluation

```powershell
python scripts/run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3 --max-questions 50
python scripts/run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3 --max-questions 50
python scripts/run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3 --max-questions 50
```

Check `run_config.json` for:

```text
trained_structural_space_enabled: true
trained_capsule_match_rate: 1.0 (or at least 0.95)
retrieval_mode: trained_semantic_projection
```

## Full held-out evaluation

```powershell
python scripts/run_grag_evaluation.py --dataset metaqa_1hop --system both --max-hops 3 --full-eval
python scripts/run_grag_evaluation.py --dataset metaqa_2hop --system both --max-hops 3 --full-eval
python scripts/run_grag_evaluation.py --dataset metaqa_3hop --system both --max-hops 3 --full-eval
```

Each command writes a separate folder under `reports/evaluation/` while loading
the same unified projector.

For hop-cap ablations, repeat with `--max-hops 1`, `2`, and `3`. The option is
shared automatically, so a single command cannot accidentally compare unequal
normal/optimized traversal depths.

## GPU memory controls

- R-GCN `MESSAGE_BATCH_SIZE`: default 16,384; reduce to 8,192 or 4,096.
- Projection embedding batch: default 16; reduce to 8 or 4.
- Projection capsule score batch: 2,048.
- Evaluation workers: 1.
- Close stale notebook kernels and stop unused models before R-GCN training.

Reducing a batch size changes peak memory and speed, not the objective.

## When retraining is mandatory

Retrain/repool R-GCN and projector when capsule fingerprint changes. Retrain only
the projector when its text encoder, mining logic, or training curriculum
changes while capsule vectors remain identical. Code/config/artifact changes
automatically invalidate relevant evaluation caches.
