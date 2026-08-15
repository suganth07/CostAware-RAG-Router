# 09 — How to run the current project safely

This chapter is only about the current `src/router` workflow. It does not start
future M6/M7 training.

## Before a large run

Run from the repository root in the project environment. Make sure your local
embedding and generation services are available. First verify cache coverage:

```bash
python scripts/router/inspect_embedding_cache.py \
  --dataset metaqa_2hop \
  --split train \
  --embedding-model nomic-embed-text \
  --embedding-cache artifacts/cache/embeddings.sqlite3 \
  --embedding-batch-size 8 \
  --index-batch-size 512 \
  --output reports/router/data_audit/metaqa_2hop_m4_embedding_cache.json
```

This command makes **zero** embedding or generation requests. It tells you how
much full-corpus indexing work is still needed.

## M4 main pilot: first command to run when ready

The current plan calls for 1,000 questions, one repeat, and all three actions:

```bash
python scripts/router/collect_outcomes.py \
  --dataset metaqa_2hop \
  --split train \
  --max-questions 1000 \
  --repeats 1 \
  --stratified \
  --full-corpus \
  --run-generation \
  --seed 7 \
  --embedding-batch-size 8 \
  --index-batch-size 512 \
  --output artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl
```

Expected final raw count: `1,000 × 1 × 3 = 3,000` outcomes.

Do not use `--no-resume`. If interrupted, run the exact same command again.
Completed candidate rows and completed 512-chunk embedding checkpoints will be
reused.

## Score and audit the main pilot

```bash
python scripts/router/score_outcomes.py \
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl \
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl \
  --output artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl

python scripts/router/build_training_dataset.py \
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl \
  --output artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_matrix.jsonl

python scripts/router/analyze_pilot.py \
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl \
  --repeats 1 \
  --output reports/router/data_audit/metaqa_2hop_m4_pilot_1000.json
```

The matrix command builds labels and features only. It does not train an RL
model.

## M4 repeated latency subset

After the main pilot succeeds, select a frozen subset and replay it three times:

```bash
python scripts/router/select_latency_subset.py \
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl \
  --max-questions 100 \
  --seed 17 \
  --output artifacts/router/manifests/metaqa_2hop_m4_latency_100.questions.jsonl

python scripts/router/collect_outcomes.py \
  --dataset metaqa_2hop \
  --split train \
  --questions-file artifacts/router/manifests/metaqa_2hop_m4_latency_100.questions.jsonl \
  --max-questions 100 \
  --repeats 3 \
  --full-corpus \
  --run-generation \
  --seed 7 \
  --index-batch-size 512 \
  --output artifacts/router/outcomes/metaqa_2hop_m4_latency_100x3.jsonl
```

Then run `score_outcomes.py` and `analyze_pilot.py` with `--repeats 3`, using
the latency file paths. Expected raw count: `100 × 3 × 3 = 900` outcomes.

## Optional critic collection

Do this only after a primary critic is explicitly enabled and validated in
`configs/router/critics.yaml`. The placeholder below is intentional:

```bash
python scripts/router/score_critics.py \
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl \
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl \
  --judge-model <configured-primary-model-id> \
  --max-questions 500 \
  --seed 7 \
  --output artifacts/router/critiques/metaqa_2hop_m4_pilot_1000_primary_500q.jsonl
```

This requests all three critic judgements for each selected question: 500
questions means 1,500 candidate records.

## What not to run yet

Do not begin outcome-model training, oracle imitation training, contextual-bandit
training, or deployment. The plan requires M4 completion and human reward
calibration first.
