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
