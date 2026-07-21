# Running and Checking the RL Router Experiment

## Prerequisites

From the project root:

```powershell
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
ollama serve
```

`datasets` is required for the locally saved 2Wiki and MultiHop-RAG Hugging
Face datasets. It is already listed in `requirements.txt`.

## Step 1: run code checks

```powershell
python -m pytest -q tests/test_rl_router.py
python scripts/build_route_outcomes.py --help
python scripts/train_cost_aware_router.py --help
```

The unit test checks two critical rules: a cheap inaccurate route must not beat
an acceptable route, and the bandit must learn a clearly better action.

## Step 2: run a small wiring test

```powershell
python scripts/build_route_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions 20 `
  --max-documents 1000 `
  --output artifacts/rl_router/metaqa_2hop_smoke.jsonl
```

Check the paired file:

```text
artifacts/rl_router/metaqa_2hop_smoke.run_config.json
```

Required checks:

```json
"learned_structural_models_enabled": false
"optimized_structural_encoder": "FeatureStructuralEncoder"
"routes": ["hybrid_rag", "graph_rag", "optimized_grag"]
```

The smoke run has intentionally limited Hybrid-RAG and GraphRAG text indexes.
Use it to find software/configuration problems, not to report retrieval quality.

## Step 3: create reportable training outcomes

Use MetaQA's full knowledge-base corpus. Start with a manageable number of
questions, confirm the result, then increase it.

```powershell
python scripts/build_route_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions 5000 `
  --full-corpus `
  --output artifacts/rl_router/metaqa_2hop_train.jsonl
```

For a stronger project, collect separate 1-hop, 2-hop, and 3-hop route tables:

```powershell
python scripts/build_route_outcomes.py --dataset metaqa_1hop --split train --max-questions 5000 --full-corpus --output artifacts/rl_router/metaqa_1hop_train.jsonl
python scripts/build_route_outcomes.py --dataset metaqa_2hop --split train --max-questions 5000 --full-corpus --output artifacts/rl_router/metaqa_2hop_train.jsonl
python scripts/build_route_outcomes.py --dataset metaqa_3hop --split train --max-questions 5000 --full-corpus --output artifacts/rl_router/metaqa_3hop_train.jsonl
```

Train one initial bandit per slice. This keeps the first experiment easy to
interpret. Combine slices only after you define a documented mixed-slice split
and retrain/validate it as one new experiment.

## Step 4: train the policy

```powershell
python scripts/train_cost_aware_router.py `
  --outcomes artifacts/rl_router/metaqa_2hop_train.jsonl `
  --output artifacts/rl_router/metaqa_2hop_bandit.json
```

The command prints the policy's train and validation route accuracy, selected
reward, oracle reward, and reward regret. The JSON output preserves both the
policy and report.

## Step 5: inspect the output

Open the `report` inside `metaqa_2hop_bandit.json` and check:

| Field | Desired direction | Meaning |
|---|---|---|
| `validation.mean_selected_reward` | Higher | Quality/cost trade-off achieved by the router. |
| `validation.mean_reward_regret` | Lower, near zero | Distance from the per-question oracle. |
| `validation.mean_oracle_reward` | Reference only | Best possible outcome after knowing all route results. |
| `validation.route_accuracy` | Higher | Agreement with cheapest acceptable route. |

Also inspect route distribution from the JSONL. A router that always chooses
one route may be valid only if the data proves it is best; otherwise it can
indicate a cost/quality bug or insufficient features.

## Step 6: held-out test protocol

Do not train on a MetaQA test split. For a final comparison:

1. Fit the policy using a train outcome file.
2. Collect outcomes for the corresponding test split with exactly the same
   corpus, models, quality threshold, and cost weights.
3. Apply the saved policy features to test questions.
4. Compare selected-route reward/cost against all-three-route oracle outcomes.
5. Report absolute route metrics plus reward regret and P95 latency.

## Optional generated-answer evaluation

Add `--run-generation` during collection:

```powershell
python scripts/build_route_outcomes.py `
  --dataset metaqa_2hop `
  --split test `
  --max-questions 500 `
  --full-corpus `
  --run-generation `
  --output artifacts/rl_router/metaqa_2hop_test_generation.jsonl
```

This is slower but enables exact-match and token-F1 terms in the quality score.
Keep retrieval-only and generation-enabled outcome files separate: they answer
different questions and have different costs.

## Common problems

### `datasets` import error

Run `python -m pip install -r requirements.txt` in the same Python environment
used to launch the script.

### Ollama connection error

Start `ollama serve`, check its port is `11434`, and verify the two models with
`ollama list`.

### Optimized route unexpectedly uses trained artifacts

The new builder constructs `FeatureStructuralEncoder` directly. Check the run
configuration; it must record `learned_structural_models_enabled: false`.

### MetaQA quality is poor in a smoke run

Expected: `--max-documents 1000` limits the Hybrid and GraphRAG fact-document
indexes, while their MetaQA native graph remains complete. Use `--full-corpus`
before interpreting quality comparisons.

### Training result is unhelpful

First check that every `sample_id` occurs exactly three times. Then inspect
quality, cost, and rewards by route. If the routes have nearly identical reward
on every question, the bandit has no meaningful routing signal.
