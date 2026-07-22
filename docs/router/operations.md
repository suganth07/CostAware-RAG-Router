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

After an approved human calibration set exists, enrich a *copy* of scored
outcomes with answer-embedding metrics. Omit `--calibration` to record raw
cosine only; it will not change reward quality.

```powershell
python scripts/router/enrich_embedding_metrics.py `
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_pilot_100_scored.jsonl `
  --questions artifacts/router/outcomes/metaqa_2hop_pilot_100.questions.jsonl `
  --calibration artifacts/router/calibration/embedding_similarity_v1.json `
  --output artifacts/router/datasets/metaqa_2hop_pilot_100_embedding_scored.jsonl
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
