# 08 — Files produced and what you see

The project writes nested operational data as JSONL and small summaries as JSON.
JSONL means one JSON object per line, which makes long jobs resumable and easy
to inspect incrementally.

## Main artifact path

```text
artifacts/router/
  outcomes/        raw candidate executions
  datasets/        scored outcomes and full-information matrices
  critiques/       optional structured critic records
  cache/           critic cache
  calibration/     future reviewed-label and calibration artifacts

reports/router/
  data_audit/      collection, quality, latency, feasibility, and oracle summaries
  critic_audit/    critic coverage and diagnostic reports
```

## What each command produces

| Command | Main output | What it means |
|---|---|---|
| `collect_outcomes.py` | `*.jsonl`, `*.questions.jsonl`, `*.manifest.json` | Raw comparison of each route, frozen input list, and reproducibility metadata. |
| `score_outcomes.py` | `*_scored.jsonl` | Raw outcomes plus metric vector, quality composition, and compute units. |
| `build_training_dataset.py` | `*_matrix.jsonl` | One complete three-action comparison per question/repeat. It is not model training. |
| `analyze_pilot.py` | report JSON | Matrix completeness, errors, quality/latency distributions, feasibility, and oracle action distribution. |
| `score_critics.py` | critic JSONL and failures JSONL | Optional candidate-blind critic records. |
| `analyze_critics.py` | report JSON and blind-review queue JSONL | Critic diagnostics plus records for human review. |
| `inspect_embedding_cache.py` | cache report JSON | Read-only exact cache coverage; no embedding/model call is made. |

## Manifests and fingerprints

A manifest is the identity card of a collection. It records the schema version,
row count, actions, selected questions, corpus fingerprint, model fingerprint,
hardware fingerprint, seed, and index reports.

Fingerprints answer a crucial question: “Can I safely compare or resume these
two files?” If the corpus or configured model changes, an old outcome should
not be silently mixed with the new experiment.

## What a user sees while a run executes

For a small run, the terminal shows dataset load, index reports, then a final
JSON summary with output file names and record count.

For a full-corpus M4 run, indexing progress appears first. Important messages
include cache hits/misses and the current dense-index batch. Only after all
three candidates are indexed does question-by-question outcome collection
begin. This is normal.

## How to inspect a result

Open a report first, not the raw JSONL:

```text
reports/router/data_audit/<experiment>.json
```

Check these fields:

1. `raw_outcomes` matches the expected question × repeat × 3 calculation.
2. All three action IDs are present.
3. Errors are zero or explained.
4. Per-action latency distributions are plausible.
5. The oracle chooses more than one action somewhere; a complete collapse to
   one action is a warning to inspect reward/budget/data balance.
6. Optional metrics are null only when their model/calibration gate is not yet
   enabled—not silently treated as zero.
