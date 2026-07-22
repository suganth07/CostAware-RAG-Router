# Router Operations

Run from the repository root after installing dependencies and starting Ollama.

```powershell
python scripts/router/collect_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions 100 `
  --repeats 2 `
  --stratified `
  --run-generation `
  --output artifacts/router/outcomes/metaqa_2hop_smoke.jsonl
```

Score the raw collection:

```powershell
python scripts/router/score_outcomes.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_smoke.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_smoke.questions.jsonl `
  --output artifacts/router/datasets/metaqa_2hop_smoke_scored.jsonl
```

Build budget-aware training examples. Leave budget flags absent until warm
hardware measurements exist; then set them explicitly.

```powershell
python scripts/router/build_training_dataset.py `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_smoke_scored.jsonl `
  --max-request-latency-ms 100 `
  --compute-budget 1.0 `
  --output artifacts/router/datasets/metaqa_2hop_smoke_training.jsonl
```

Check the outcome manifest before trusting results: the action list must match,
every question/repeat must have three actions, and
`optimized_structural_encoder` must be `FeatureStructuralEncoder`.

Audit a pilot before collecting more data or training a policy:

```powershell
python scripts/router/analyze_pilot.py `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_pilot_100_scored.jsonl `
  --repeats 2 `
  --output reports/router/data_audit/metaqa_2hop_pilot_100.json
```

If a hard request budget is supplied, the report must show at least one feasible
action for a meaningful fraction of questions. A report where every action is
infeasible is a budget-configuration problem, not evidence that every pipeline
is unusable.

## Collect the first structured-critic pilot

After deterministic scoring, create 100 candidate-blind judgements. The
command balances records over the three actions and can safely resume. It uses
your local Ollama judge; pull the model first if it is not installed.

If `localhost:11434` is refused, first open a separate terminal and run
`ollama serve` (keep it running), or start the Ollama desktop application.
The script now checks the service and model once before it starts judging.

```powershell
ollama pull qwen3:4b-instruct

python scripts/router/score_critics.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_pilot_100.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_pilot_100.questions.jsonl `
  --judge-model qwen3:4b-instruct `
  --max-records 100 `
  --output artifacts/router/critiques/metaqa_2hop_pilot_100_critic.jsonl
```

Inspect the JSONL and its optional `*.failures.jsonl` companion. Do not merge
critic scores into a reportable reward until a human audit has checked the
sample; see [semantic-metrics-and-critic.md](semantic-metrics-and-critic.md).

Audit a completed critic artifact and create a 20-record blind human-review
queue. The report compares action coverage, score distributions, the explicit
schema-repair rate, exact-match agreement, and accepted-answer coverage. These
are diagnostics, not human ground truth.

```powershell
python scripts/router/analyze_critics.py `
  --critiques artifacts/router/critiques/metaqa_2hop_pilot_100_critic.jsonl `
  --outcomes artifacts/router/outcomes/metaqa_2hop_pilot_100.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_pilot_100.questions.jsonl `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_pilot_100_scored.jsonl `
  --output reports/router/critic_audit/metaqa_2hop_pilot_100.json `
  --review-output artifacts/router/review_queue/metaqa_2hop_pilot_100_blind_review.jsonl
```

After an approved human calibration set exists, enrich a *copy* of scored
outcomes with answer-embedding metrics. Omit `--calibration` to record raw
cosine only; it will not change reward quality.

Fit the versioned calibration artifact first from the plan-required labelled
equivalent/non-equivalent answer pairs. Each input JSONL row contains
`answer_type`, `raw_similarity`, and boolean `equivalent`.

```powershell
python scripts/router/fit_embedding_calibration.py `
  --labels artifacts/router/calibration/embedding_equivalence_labels.jsonl `
  --output artifacts/router/calibration/embedding_similarity_v1.json
```

```powershell
python scripts/router/enrich_embedding_metrics.py `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_pilot_100_scored.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_pilot_100.questions.jsonl `
  --calibration artifacts/router/calibration/embedding_similarity_v1.json `
  --output artifacts/router/datasets/metaqa_2hop_pilot_100_embedding_scored.jsonl
```

## Corrected critic pilot (v3, two-pass)

The v1 pilot was rejected: it treated context-grounded abstentions as globally
correct despite non-empty accepted answers. Version 2 still over-scored the
very long GraphRAG contexts. Do not merge either rejected artifact with v3.
Version 3 uses two local judge calls per candidate: one context-free answer
correctness call and one evidence-grounding call. Collect it into a fresh,
separately versioned artifact and audit it with the same command above,
substituting the v3 input/output paths.

```powershell
python scripts/router/score_critics.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_pilot_100.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_pilot_100.questions.jsonl `
  --judge-model qwen3:4b-instruct `
  --max-records 100 `
  --output artifacts/router/critiques/metaqa_2hop_pilot_100_critic_v3.jsonl
```


## Collect the first structured-critic pilot

After deterministic scoring, create 100 candidate-blind judgements. The
command balances records over the three actions and can safely resume. It uses
your local Ollama judge; pull the model first if it is not installed.

```powershell
ollama pull qwen3:4b-instruct

python scripts/router/score_critics.py `
  --outcomes artifacts/router/outcomes/metaqa_2hop_pilot_100.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_pilot_100.questions.jsonl `
  --judge-model qwen3:4b-instruct `
  --max-records 100 `
  --output artifacts/router/critiques/metaqa_2hop_pilot_100_critic.jsonl
```

Inspect the JSONL and its optional `*.failures.jsonl` companion. Do not merge
critic scores into a reportable reward until a human audit has checked the
sample; see [semantic-metrics-and-critic.md](semantic-metrics-and-critic.md).
