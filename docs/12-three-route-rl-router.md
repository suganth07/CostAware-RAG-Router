# Three-Route Evaluation and Cost-Aware RL Router

## Purpose

This workflow compares exactly three retrieval routes while intentionally
disabling the trained R-GCN and query-projection model artifacts:

1. **Hybrid RAG**: BM25 plus dense/vector retrieval, fused with reciprocal-rank fusion.
2. **GraphRAG**: dense source retrieval plus query-aware graph-neighborhood expansion.
3. **Optimized GraphRAG**: bounded graph capsules and relation-aware path search.

The Optimized GraphRAG route uses `FeatureStructuralEncoder`, a deterministic
eight-feature capsule representation. It does **not** load
`TrainedStructuralEncoder`, `capsule_rgcn.pt`, or `query_projection_mlp.pt`.
Text embeddings and the optional answering LLM remain necessary; disabling
R-GCN does not mean disabling `nomic-embed-text` or `qwen3:4b-instruct`.

## Dataset readiness

### MetaQA

The installed MetaQA data is sufficient for this work. It contains the complete
movie knowledge base and the following vanilla QA split sizes:

| Split | Train questions | Test questions |
|---|---:|---:|
| 1-hop | 96,106 | 9,947 |
| 2-hop | 118,980 | 14,872 |
| 3-hop | 114,196 | 14,274 |

MetaQA is the primary graph benchmark because every fact is a typed graph
triple. No additional graph dataset is needed to begin the project.

### 2WikiMultiHopQA and MultiHop-RAG

Both datasets are present locally and are useful for cross-dataset testing:

- 2Wiki has train, validation, and test splits.
- MultiHop-RAG has a train corpus and train QA split.

Their local adapters require the Hugging Face `datasets` package. It is already
listed in `requirements.txt`; install dependencies before using these datasets.
MultiHop-RAG needs a held-out validation/test split before it can be used for a
final generalization claim. Do not create a random split silently: retain the
original dataset and write any derived split to a clearly named new artifact.

## One-time setup

```powershell
python -m pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen3:4b-instruct
```

Start Ollama before running a model-backed route collection:

```powershell
ollama serve
```

## Build the counterfactual training data

`scripts/build_route_outcomes.py` runs every selected question through all
three routes. Its output is a JSONL counterfactual table: each question has
one measured quality, latency, graph-work, and context-size outcome per route.
This is the correct data format for offline routing work because every action's
reward is observed, rather than guessed.

### Small smoke test

```powershell
python scripts/build_route_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions 20 `
  --max-documents 1000 `
  --output artifacts/rl_router/metaqa_2hop_smoke.jsonl
```

The smoke command always loads the complete MetaQA graph, but limits the
Hybrid-RAG and GraphRAG fact-document indexes. It verifies software wiring
only; do not quote its quality numbers because the text-retrieval routes have a
deliberately incomplete source corpus.

### Reportable training collection

```powershell
python scripts/build_route_outcomes.py `
  --dataset metaqa_2hop `
  --split train `
  --max-questions 5000 `
  --full-corpus `
  --output artifacts/rl_router/metaqa_2hop_train.jsonl
```

Repeat for `metaqa_1hop` and `metaqa_3hop`, then combine only after confirming
the same model/configuration is recorded in each run configuration file.

Add `--run-generation` only when you want answer-generation metrics. It is
slower because every route makes an LLM call. Retrieval-only outcomes use
Answer Recall@10 and context/evidence recall as documented quality proxies;
they are not final answer-accuracy results.

## Train the initial RL model

The initial policy is a **full-feedback contextual bandit**, not a classifier
and not yet a sequential DQN/PPO agent. This is the right first RL stage
because the collected table tells us the reward of all three actions.

```powershell
python scripts/train_cost_aware_router.py `
  --outcomes artifacts/rl_router/metaqa_2hop_train.jsonl `
  --output artifacts/rl_router/metaqa_2hop_bandit.json
```

The policy receives cheap, pre-retrieval features only:

- question length;
- bracketed-entity and capitalized-phrase counts;
- relation-word count;
- multi-hop cue count;
- comparison/temporal cue count.

Its reward is quality-first:

```text
if quality < 0.65:
    reward = strong negative quality penalty - cost penalty
else:
    reward = 1 + quality - cost penalty
```

Quality is calculated as:

```text
0.40 answer term + 0.25 faithfulness term
+ 0.20 evidence term + 0.15 completeness term
```

The generated JSON model contains the feature order, normalization values,
actions, learned linear-policy weights, and held-out reward/regret report.

## What the routes measure

Each outcome records:

- route name and question ID;
- retrieval/generation metrics;
- elapsed route latency;
- evidence-unit and context size;
- graph nodes and edges processed where applicable;
- normalized cost;
- quality score;
- reward;
- cheapest acceptable route label.

The cost combines normalized latency, graph nodes, graph edges, evidence units,
and context characters. The weights are recorded beside each JSONL run, so a
later experiment can reproduce or deliberately alter the cost policy.

## Validation protocol

Use MetaQA train data for outcome collection and policy fitting. Evaluate the
final router on MetaQA test data with the same three routes and full corpus.
Report:

1. answer and evidence quality;
2. average and P95 latency;
3. route distribution;
4. mean selected reward;
5. reward regret versus the per-question oracle;
6. excess cost versus the cheapest acceptable route.

Compare it with always-Hybrid, always-GraphRAG, always-Optimized-GraphRAG,
random selection, and a fixed escalation rule.

## Next research stage: sequential escalation

Do not use gold answers in live deployment. Gold answers are only for offline
route rewards. Once the contextual bandit is sound, add a critic that estimates
retrieval/evidence adequacy after the first route. Then the action space can
be extended to `accept`, `escalate_to_route`, and `abstain`, with a maximum of
two or three routes per question.

## Safety checks

- `learned_structural_models_enabled` must be `false` in each outcome record.
- The Optimized GraphRAG index mode should be `semantic_text`, not a
  `trained_*` mode.
- Use `--full-corpus` for all reportable MetaQA comparisons.
- Keep the same question set and generation setting for all three routes.
- Never call retrieval-only context recall “answer accuracy.”
