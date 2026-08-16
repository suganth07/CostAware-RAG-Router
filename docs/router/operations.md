# Router operations

## Current approved operation: M4 pilot

The next plan operation is the M4 pilot, not training. Follow the complete,
resumable sequence in [M4 pilot collection runbook](m4-pilot.md). It collects
the full action matrix, a repeated latency subset, deterministic metrics, and
conditional critic artifacts. It explicitly stops before M6/M7.

Before M4 full-corpus collection, run the read-only cache diagnostic from the
[M4 runbook](m4-pilot.md#inspect-the-embedding-cache-first). It reports exact
chunk-level cache coverage and makes no Ollama requests.

## General collection pattern

Run commands from the repository root in the same environment and hardware
configuration used for the planned experiment. Start the local services used by
the embedding/generation configuration before collection.

```powershell
python scripts/router/collect_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions <count> `
  --repeats <count> `
  --stratified `
  --full-corpus `
  --run-generation `
  --index-batch-size <checkpoint-size> `
  --output artifacts/router/outcomes/<experiment>.jsonl
```

The collector is resumable by default. Keep the output path and all model,
corpus, seed, and collection settings unchanged when resuming. Use a new
artifact name for a different experiment. Each collection writes its raw
outcomes, frozen question list, and manifest with fingerprints.

## Score and audit an outcome matrix

```powershell
python scripts/router/score_outcomes.py `
  --outcomes artifacts/router/outcomes/<experiment>.jsonl `
  --questions artifacts/router/outcomes/<experiment>.questions.jsonl `
  --output artifacts/router/datasets/<experiment>_scored.jsonl

python scripts/router/analyze_pilot.py `
  --scored-outcomes artifacts/router/datasets/<experiment>_scored.jsonl `
  --repeats <count> `
  --output reports/router/data_audit/<experiment>.json
```

`analyze_pilot.py` checks the complete three-action matrix and reports quality,
latency, errors, feasibility, and oracle-action distribution. It does not train
anything.

For M4 latency calibration, reuse a compatible main-pilot measurement as the
first logical repeat and collect only two additional repeats. The dedicated
`analyze_latency_subset.py` preserves raw run fingerprints and rejects an
incompatible corpus, model, hardware, or candidate pipeline. The exact commands
are in the [M4 runbook](m4-pilot.md#3-freeze-and-collect-the-repeated-latency-subset).

## Final runtime outcome (after M6–M9)

The project is complete only when a user can enter a question in a terminal and
receive the selected candidate, routing decision logs, and final generated
answer. This is a later M9 runtime deliverable, not an M4 operation. Its CLI
will load a validated exported router artifact, extract runtime-safe features,
apply budgets/uncertainty fallback, select one candidate, execute it, and print
structured routing telemetry plus the answer.

## Critic operation is opt-in

Critic collection is allowed only after an explicit model choice has been made
in `configs/router/critics.yaml` and validated. Pass that chosen model ID
explicitly to the script:

```powershell
python scripts/router/score_critics.py `
  --outcomes artifacts/router/outcomes/<experiment>.jsonl `
  --questions artifacts/router/outcomes/<experiment>.questions.jsonl `
  --judge-model <configured-primary-model-id> `
  --max-questions <complete-question-count> `
  --output artifacts/router/critiques/<experiment>_primary.jsonl
```

`--max-questions` always includes every candidate for each selected complete
question/repeat matrix. For a 500-question M4 critic sample, that means 1,500
candidate judgments. The critic cache is stored at
`artifacts/router/cache/critic_responses.sqlite3` and includes prompt/model/
stage version keys, so compatible reruns reuse results.

Critic and semantic values remain optional/nullable until their human
calibration gates are complete. Never replace missing optional metrics with
zero, and never start M6/M7 merely because an M4 matrix exists.

For the blind human-calibration queue and the completed-label validation
command, see [Semantic metrics and structured critic](semantic-metrics-and-critic.md#blinded-human-calibration-workflow).

## Configuration boundaries

- `configs/router/m4_pilot.yaml` records the planned pilot scale and seeds.
- `configs/router/semantic.yaml` is disabled until a semantic model is selected
  and validated.
- `configs/router/critics.yaml` is disabled until primary/secondary critic
  choices are selected and validated.

No semantic-equivalence or critic model is built into the router architecture.
