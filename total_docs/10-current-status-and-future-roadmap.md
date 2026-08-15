# 10 — Current status and the remaining RL roadmap

## What has already passed

| Milestone | Status | Meaning |
|---|---|---|
| M0: contracts/configuration | Partial foundation complete | Fixed action registry, typed JSON-safe contracts, manifests, fingerprints, and budgets exist. Grouped train/validation/test manifests are still future work. |
| M1: candidate adapters/collector | Partial foundation complete | All three routes have adapters; collection resumes safely and validates a complete matrix. |
| M2: metrics/reward | Foundation complete | Deterministic metrics, evidence nDCG, completeness, nullable quality composition, and model-neutral semantic interfaces exist. |
| M3: critics | Foundation complete | Candidate-blind structured critic, caching, primary/secondary interfaces, disagreement logic, and human-calibration contract exist. |
| M4: pilot | Prepared, not complete | Commands, cache diagnostic, full-corpus collection, repeated latency subset, and audits are ready. |
| M5–M10 | Not started | Larger dataset, learned models, bandit policy, evaluation, runtime artifact, and rollout remain future work. |

## Verified small engineering run

The 40-question verification passed:

```text
40 questions × 2 repeats × 3 actions = 240 outcomes
40 questions × 2 repeats             = 80 complete matrix rows
```

There were zero candidate errors and no matrix-completeness warnings. The
budget-free oracle chose more than one action: Optimized GraphRAG for 50 groups,
GraphRAG for 20, and Hybrid RAG for 10. This proves the data path can preserve
meaningful action differences.

It is not a final benchmark, because text retrieval used a 1,000-document
engineering limit rather than the full corpus.

## Immediate next work: M4

1. Inspect cache coverage.
2. Run the 1,000-question full-corpus main matrix.
3. Score it and audit completeness, quality, errors, latency, feasibility, and
   oracle action balance.
4. Run the frozen 100-question latency subset three times.
5. If a validated critic is configured, collect three critic outcomes for at
   least 500 questions.
6. Perform 300–500 balanced manual reviews before promoting critic/semantic
   signals or selecting provisional reward weights.

## What comes after M4

### M5 — Full training dataset

Collect 15,000–30,000 stratified questions. Freeze grouped train, validation,
cross-domain, and locked-test manifests. Add selectively validated semantic and
critic labels. Create a data card that explains sources, splits, quality, and
limitations.

### M6 — Outcome model and imitation baseline

Train a compact multi-task model. Given runtime-safe question features and an
action, it should predict:

```text
quality
latency quantiles
compute cost
failure probability
uncertainty
```

Then train an imitation baseline to copy the observed budget-aware oracle. This
is a diagnostic baseline: it answers “can simple supervised learning recover a
useful route choice?”

### M7 — Constrained contextual bandit

Train a policy using complete action outcomes, with explicit constraints on
mean latency, tail latency, and compute. The policy should prefer expected
quality among feasible actions and use uncertainty/safety guards. This is the
actual RL stage.

### M8–M10 — Proof, runtime, and safe release

Evaluate against static and heuristic baselines, then test ablations, cross-
domain robustness, calibration, and confidence intervals. Export a compatible
runtime artifact, run in shadow mode, then use canary/rollback controls before
any real deployment.

## The key rule

The project must not train an RL policy merely because a JSONL matrix exists.
Quality labels, reward weights, constraints, and evaluation splits must first
be trustworthy. This is why the workflow is deliberately staged.
