# 06 — How answers are scored and rewarded

Raw answers alone cannot train a reliable router. The project turns each
`CandidateOutcome` into a nullable `MetricVector`, then composes quality from
the metrics that are genuinely available.

## Deterministic metrics available now

`scripts/router/score_outcomes.py` reads raw outcomes and the frozen question
file. For each outcome it calculates:

| Metric family | Examples | Question it answers |
|---|---|---|
| Answer correctness | normalized exact match, token/set F1, boolean accuracy, numeric/date accuracy | Did the answer match an accepted answer? |
| Hard conflicts | numeric, polarity, entity conflicts | Is a semantically similar-looking answer still wrong in an important way? |
| Evidence/retrieval | supporting-evidence recall, evidence MRR, evidence nDCG, context precision | Did retrieval return useful supporting material near the top? |
| Completeness | accepted-answer coverage | Did the response contain all required accepted answer pieces? |
| Reliability | failure penalty | Did the route error or time out? |
| Cost proxy | compute units | How much graph/context/token work did the route perform? |

## Why some values are null

`null` means “not measured or not available.” It does **not** mean zero.

For example, a retrieval-only run has no generated answer, so answer correctness
is unavailable. Similarly, semantic equivalence and claim faithfulness remain
null until their configurable model adapters are chosen and validated.

Replacing unknown values with zero would say “we measured this route and it
failed,” which is false and would bias reward/training. The reward composer uses
an availability mask instead.

## Quality composition

The quality composer groups related signals into four understandable blocks:

```text
answer quality
faithfulness / groundedness
evidence quality
completeness
```

The provisional configuration weights are 0.50, 0.20, 0.20, and 0.10. Only
available blocks participate; their weights are renormalized rather than
pretending absent metrics are failures.

Then penalties are applied for contradictions, unsupported claims, execution
failures, hallucination flags, or incorrect refusals. A numeric or polarity
conflict forces quality to `-1.0`, because a response with the wrong date,
number, yes/no polarity, or equivalent hard contradiction should not receive
credit simply because it shares words with the gold answer.

## The budget-aware oracle

After scoring, `build_training_dataset.py` forms a complete three-action group
for each question/repeat. It computes action feasibility using configured:

```text
mean latency budget
tail (p95) latency budget
per-request latency budget
compute budget
```

The **oracle** is not a learned model. It is a label generator that says:

```text
Among feasible actions, which observed action had the highest quality?
```

If every action violates a budget, the oracle explicitly selects the action
with the least violation. The output stores both the selected action and the
violations, so an impossible budget cannot be hidden.

## What the built training matrix contains

One full-information row includes:

```text
question and runtime-safe question features
quality for all 3 actions
latency for all 3 actions
compute cost for all 3 actions
failure status for all 3 actions
feasibility mask
budget-aware oracle action
```

Creating this matrix is **data preparation**, not training. The current plan
does not allow M6/M7 learning to start until M4 collection and reward
calibration requirements are complete.
