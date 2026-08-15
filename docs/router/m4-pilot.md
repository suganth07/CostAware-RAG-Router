# M4 pilot collection runbook

## Purpose and boundary

M4 creates the first plan-scale, full-information router dataset. It asks the
same stratified MetaQA questions to every supported candidate:

- `hybrid_rag`
- `graph_rag`
- `optimized_graph_rag`

For every question, the artifact therefore contains the three outcomes that a
future router would otherwise have had to predict. This lets later work compare
actions fairly. M4 is data collection and evaluation only: it does **not**
train an outcome model, imitate an oracle, or train a contextual-bandit policy.

The proposed pilot is defined in `configs/router/m4_pilot.yaml`:

| Part | Setting | Why |
|---|---:|---|
| Main matrix | 1,000 stratified questions × 1 repeat × 3 actions | Captures action wins and deterministic quality at plan scale. |
| Latency calibration | 100 frozen questions × 3 repeats × 3 actions | Separates latency variation from question mix without tripling the main run. |
| Corpus | Full corpus | Makes retrieval and latency results meaningful beyond the 1,000-document engineering test. |
| Generation | Enabled | Quality must be measured on generated answers, not retrieval alone. |

Use 2,000 main questions instead of 1,000 only by changing the value in the
commands consistently. Do not mix runs with different sizes under one artifact
name.

## Before starting

Run these commands from the repository root in the same environment used for
the successful 40-question check. Ensure the configured local embedding and
generation services are running and their models are available. Collection
records model, corpus, hardware, seed, and action fingerprints in a manifest.

Do not use `--no-resume` for the M4 commands. Re-running the exact command
resumes missing outcome rows and preserves the original artifact path. If model
or corpus settings change, choose a new output name; never append a new
fingerprint to an old pilot file.

### Inspect the embedding cache first

Do this before every new full-corpus M4 run. It is read-only: it does not build
the registry, contact Ollama, generate an embedding, or start outcome
collection. It compares the exact NormalRAG chunk texts with the cache namespace
that the collector will use.

```powershell
python scripts/router/inspect_embedding_cache.py `
  --dataset metaqa_2hop `
  --split train `
  --embedding-model nomic-embed-text `
  --embedding-cache artifacts/cache/embeddings.sqlite3 `
  --embedding-batch-size 8 `
  --index-batch-size 512 `
  --output reports/router/data_audit/metaqa_2hop_m4_embedding_cache.json
```

If `missing_unique_texts` is large, the initial registry build must create
those vectors. This is not a cache bug: the cache is content-addressed and can
only reuse exact texts embedded with the same namespace. The collector now
persists completed NormalRAG batches of 512 chunks, so an interruption retains
the completed batch embeddings for a later resumed run. Do not change the
embedding model, cache namespace, or batch settings while diagnosing a run.

## 1. Collect the main full-information matrix

This is the long-running step. It executes all three RAG candidates for 1,000
stratified training questions with the full corpus. Expected raw rows:

```text
1,000 questions × 1 repeat × 3 actions = 3,000 outcomes
```

```powershell
python scripts/router/collect_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions 1000 `
  --repeats 1 `
  --stratified `
  --full-corpus `
  --run-generation `
  --seed 7 `
  --index-batch-size 512 `
  --output artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl
```

It writes these linked artifacts:

- `artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl`: raw candidate
  outcomes, timings, retrieval records, errors, and answers.
- `artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl`:
  the frozen question list.
- `artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.manifest.json`: run,
  model, corpus, hardware, seed, and index fingerprints.

## 2. Score and audit the main matrix

Scoring computes deterministic answer and evidence metrics. A configured
semantic adapter is not assumed: unavailable semantic/critic metrics remain
null and their availability masks are retained, rather than being replaced by
false zeros.

```powershell
python scripts/router/score_outcomes.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl `
  --output artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl

python scripts/router/build_training_dataset.py `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl `
  --output artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_matrix.jsonl

python scripts/router/analyze_pilot.py `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl `
  --repeats 1 `
  --output reports/router/data_audit/metaqa_2hop_m4_pilot_1000.json
```

`build_training_dataset.py` is an artifact builder, not a training command. It
checks that all actions exist for each question and turns the three scored rows
into one full-information matrix row. Do not run M6/M7 training scripts after
this step.

Check the audit report for exactly 3,000 raw outcomes, 1,000 complete matrices,
three action IDs, zero unexpected errors, non-collapsed oracle selection, and
the latency distribution for each candidate.

## 3. Freeze and collect the repeated latency subset

First select the subset from the main pilot's frozen question file. This makes
the subset deterministic, records the seed, and prevents a later re-run from
measuring a different question mix.

```powershell
python scripts/router/select_latency_subset.py `
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl `
  --max-questions 100 `
  --seed 17 `
  --output artifacts/router/manifests/metaqa_2hop_m4_latency_100.questions.jsonl
```

### Reuse the completed main-pilot measurement as logical repeat 0

When the main matrix was collected with the same corpus, embedding model,
generation model, pipeline configuration, and hardware, its outcome rows for
the frozen subset are valid first latency measurements. Do **not** collect those
300 rows again. Instead, collect two new repeats, for 600 new outcomes total:

```text
100 reused outcomes × 1 existing repeat × 3 actions = 300 existing rows
100 questions × 2 new repeats × 3 actions = 600 new rows
logical latency cohort = 900 measurements
```

```powershell
python scripts/router/collect_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --questions-file artifacts/router/manifests/metaqa_2hop_m4_latency_100.questions.jsonl `
  --max-questions 100 `
  --repeats 2 `
  --full-corpus `
  --run-generation `
  --seed 7 `
  --index-batch-size 512 `
  --output artifacts/router/outcomes/metaqa_2hop_m4_latency_100x2_new.jsonl

python scripts/router/score_outcomes.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_latency_100x2_new.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_m4_latency_100x2_new.questions.jsonl `
  --output artifacts/router/datasets/metaqa_2hop_m4_latency_100x2_new_scored.jsonl

python scripts/router/analyze_latency_subset.py `
  --source-scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl `
  --supplementary-scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_latency_100x2_new_scored.jsonl `
  --questions artifacts/router/manifests/metaqa_2hop_m4_latency_100.questions.jsonl `
  --output reports/router/data_audit/metaqa_2hop_m4_latency_100x3.json
```

`analyze_latency_subset.py` leaves both raw source artifacts unchanged. It
creates a derived analysis cohort only in memory, verifies that corpus/model/
hardware/pipeline fingerprints match, and maps the main-pilot values to logical
repeat 0 and the two new values to logical repeats 1 and 2. Its report includes
within-question latency range and standard deviation per action.

If any compatibility check fails, do not use the existing values: collect all
three repeats as a new, standalone 900-row latency artifact instead. The
latency cohort is for variance analysis only and is never mixed into the main
training matrix.

## 4. Conditional M4 critic collection

Only do this after `configs/router/critics.yaml` has an explicitly enabled,
validated primary critic and its model ID is supplied deliberately. The model
is shown below as a placeholder; it is not a repository default.

This command samples 500 complete question matrices and judges every candidate
for each selected question. It therefore requests 1,500 critic records, which
meets the plan requirement of critic coverage for every candidate on at least
500 questions.

```powershell
python scripts/router/score_critics.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl `
  --judge-model <configured-primary-model-id> `
  --max-questions 500 `
  --seed 7 `
  --output artifacts/router/critiques/metaqa_2hop_m4_pilot_1000_primary_500q.jsonl

python scripts/router/analyze_critics.py `
  --critiques artifacts/router/critiques/metaqa_2hop_m4_pilot_1000_primary_500q.jsonl `
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl `
  --output reports/router/critic_audit/metaqa_2hop_m4_pilot_1000_primary_500q.json `
  --review-output artifacts/router/review_queue/metaqa_2hop_m4_pilot_1000_primary_500q_blind_review.jsonl
```

Do not promote these critic scores to reward values yet. The required 300–500
balanced human reviews are still the calibration gate. The existing prompt,
model, run, and full-prompt cache keys preserve compatibility and resumption.

## What constitutes a successful M4 pilot

M4 is ready for reward-calibration review when:

1. Both manifests identify the same intended candidate registry and immutable
   corpus/model settings for their respective run.
2. The main matrix is complete: 1,000 questions, one repeat, and all three
   actions (3,000 rows).
3. The latency matrix is complete: the frozen 100 questions, three repeats,
   and all three actions (900 rows).
4. The audits show candidate errors, latency distributions/variance, action
   feasibility, and action-oracle imbalance explicitly.
5. Deterministic metrics are present. Optional semantic/critic values are
   present only if their configuration and validation gates were satisfied.
6. Human calibration is obtained before any optional signal or provisional
   reward weighting is used for M5/M6/M7 decisions.

At that point, the next work is human reward calibration and M5 dataset/split
preparation—not outcome-model or contextual-bandit training.
