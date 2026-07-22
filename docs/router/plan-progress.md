# `plan.md` Implementation Progress

## Why this project exists

The project is not simply training RL on RAG outputs. It builds a safe offline
contextual-bandit router that chooses one action per question:
`hybrid_rag`, `graph_rag`, or `optimized_graph_rag`. The router should choose
the action with the best expected answer/evidence quality while respecting
measured latency and compute budgets. Full-information collection runs all
three actions offline, so later learning has counterfactual outcomes rather
than only feedback from the action that happened to be selected.

## Status key

- **Done**: implemented and exercised on the 100-question smoke run.
- **Partial**: foundation exists, but plan exit criteria are not met.
- **Not started**: no plan-aligned implementation yet.
- **Blocked**: requires a model choice, human labels, target-hardware data, or
  a larger collection that cannot be fabricated in code.

## Milestone checklist

### M0 — Lock contracts and configuration: Partial

- **Done**: immutable three-action registry; `QuestionRecord`,
  `CandidateOutcome`, `MetricVector`, and `CritiqueRecord`; JSONL artifacts;
  manifests/fingerprints; explicit budget/reward configuration.
- **Partial**: experiment config is present, but train/validation/test manifests
  and the locked evaluation manifest have not been frozen.
- **Blocked**: final semantic model, primary/secondary critics, production
  fallback, and target-hardware budgets are explicitly unresolved in plan.md.

### M1 — Candidate adapters and shared collector: Partial

- **Done**: adapters for Hybrid RAG, GraphRAG, and Optimized GraphRAG;
  resumable full-information collector; normalized outcome/timing records;
  complete action-matrix validation; smoke collection of 100 questions × two
  repeats × three actions.
- **Not done**: randomized candidate execution order, complete repeated
  latency-calibration subset, and 1,000–2,000-question pilot collection.

### M2 — Metric and reward engine: Partial

- **Done**: deterministic lexical metrics, answer-type and hard-conflict
  guards, retrieval/evidence signals available from the existing evaluator,
  nullable metric composition, compute units, budget-aware oracle, raw answer
  embedding similarity support.
- **In progress**: semantic-equivalence scorer contract and reproducible
  embedding-calibration fitter.
- **Not done**: selected/validated semantic cross-encoder or answer-similarity
  model; learned per-answer-type embedding thresholds; contradiction model;
  evidence nDCG and claim-level faithfulness/completeness implementations;
  provisional human-calibrated final reward weights.

### M3 — Critic pipeline: Partial

- **Done**: candidate-blind strict JSON critic; prompt/model versioning;
  structured records; v1/v2 rejection history; v3 two-pass correctness versus
  grounding separation; completed 100-record V3 smoke critic artifact and
  audit.
- **Not done**: prompt-result cache keyed by prompt/model fingerprint; second
  independent critic; disagreement detection/adjudication; uncertainty weight;
  plan-required 300–500 manually reviewed balanced calibration outcomes and
  agreement/calibration metrics.

### M4 — Pilot outcome dataset: Not started

- **Done prerequisite**: 100-question smoke matrix and smoke critic run.
- **Not done**: 1,000–2,000 stratified questions, 3–5 latency repeats on a
  calibration subset, critic coverage for at least 500 questions, action-win
  analysis, and provisional reward calibration.

### M5 — Full training dataset: Not started

- **Not done**: 15,000–30,000 stratified questions; semantic and selective
  critic scores; grouped train/validation/cross-domain/locked-test manifests;
  data card.

### M6 — Outcome model and oracle imitation: Not started

- **Not done**: compact multi-task predictor for per-action quality, latency
  quantiles, compute, and failure; pairwise ranking loss; oracle labels;
  imitation warm start; residual/calibration reports.

### M7 — Constrained contextual-bandit policy: Not started

- **Not done**: full-information constrained objective; Lagrangian updates;
  constraint curriculum; probability/uncertainty calibration; policy checkpoint.

### M8 — Evaluation and ablation: Not started

- **Not done**: static/heuristic/oracle baselines; paired bootstrap tests;
  regret and constraint metrics; feature/reward/critic/embedding ablations;
  cross-domain and robustness evaluations.

### M9 — Runtime artifact and shadow deployment: Not started

- **Not done**: runtime selection policy, conservative uncertainty fallback,
  compatibility validation, telemetry, drift checks, exported artifact, shadow
  run.

### M10 — Controlled rollout: Not started

- **Not done**: promotion gates, canary rollout, rollback triggers, and
  production monitoring.

## Current next order from `plan.md`

1. Finish M2 metric foundations and select/validate the semantic model.
2. Manually review the first 100 V3 critic judgments; then create the
   plan-required 300–500 balanced critic calibration set.
3. Run the 1,000–2,000-question pilot with repeated latency measurements.
4. Calibrate provisional reward components and freeze grouped splits.
5. Collect the full training matrix before attempting M6/M7 training.

The smoke artifacts are deliberately not used to claim a trained RL router:
their purpose is to validate contracts and expose collection/metric failures
cheaply before the plan's larger collections are run.
