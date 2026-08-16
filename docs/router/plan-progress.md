# `plan.md` implementation progress

## Project purpose

This project is an offline, quality-constrained contextual-bandit router. For
each question it must select one of `hybrid_rag`, `graph_rag`, or
`optimized_graph_rag`. Offline collection runs all three candidates, producing
the counterfactual quality and cost matrix required for later supervised and
RL-style policy learning. It is not yet an RL training result.

## Status key

- **Done**: implemented and verified.
- **Prepared**: code, configuration, and run procedure are ready; the planned
  artifact has not been collected yet.
- **Partial**: foundation exists, but the plan exit criteria are incomplete.
- **Blocked**: requires a model decision, human labels, target-hardware data,
  or a larger collection. It must not be fabricated in code.

## Milestone checklist

### M0 — Lock contracts and configuration: Partial

- **Done**: immutable three-action registry; typed question/outcome/metric/
  critic contracts; JSONL artifacts; manifests and fingerprints; explicit
  budget and reward configuration.
- **Partial**: grouped train/validation/test manifests and the locked
  evaluation manifest have not been frozen.
- **Blocked**: deployment budgets, fallback, semantic model, and critic-model
  choices remain deliberately unresolved.

### M1 — Candidate adapters and shared collector: Partial

- **Done**: adapters for Hybrid RAG, GraphRAG, and Optimized GraphRAG;
  resumable full-information collection; normalized timings; complete action
  matrix validation; random action order with recorded seed; 100-question
  smoke collection and 40-question engineering verification.
- **Prepared**: frozen latency-subset selection from a main pilot question
  manifest and exact-subset replay through `--questions-file`.
- **Pending**: plan-scale 1,000–2,000-question collection and the repeated
  latency calibration subset.

### M2 — Metric and reward engine: Complete foundation

- **Done**: deterministic lexical/answer-type metrics, evidence metrics,
  compute units, budget-aware oracle, raw embedding support, semantic-
  equivalence contract, embedding-calibration fitter, contradiction contract,
  claim-faithfulness interfaces, answer completeness, evidence nDCG, and a
  nullable availability-mask reward composer.
- **Blocked**: a selected and validated semantic/contradiction model, an
  enabled claim verifier, learned answer-type thresholds, and final
  human-calibrated reward weights. These remain null/disabled until labels and
  model decisions exist.

### M3 — Critic pipeline: Complete foundation

- **Done**: candidate-blind structured critic, prompt/model/run versioning,
  two-pass correctness versus grounding separation, response cache,
  configurable primary/secondary slots, disagreement and uncertainty logic,
  model-neutral adjudication interface, and human-calibration contract.
- **Done**: the 100-record V3 critic artifact/audit was collected without
  failures. It is diagnostic only.
- **Done (15 August 2026)**: blinded human-calibration queue builder, private
  join-map handling, strict boolean label validation, and critic-versus-human
  agreement report. A 100-record V3 calibration queue is prepared; it is a
  workflow verification, not the required 300–500 reviewed-label set.
- **Blocked**: selecting a secondary critic/adjudicator and the 300–500
  balanced human-reviewed calibration set. No default critic model is encoded
  in M3 configuration.

### M4 — Pilot outcome dataset: In progress

- **Done prerequisite**: 100-question smoke matrix, 40-question engineering
  verification, and M2/M3 foundations.
- **Prepared**: `configs/router/m4_pilot.yaml`; resumable full-corpus main
  collection; deterministic scoring and audits; a frozen stratified
  latency-subset selector; complete-question critic sampling; and a read-only
  full-corpus embedding-cache inspection command.
- **Verified diagnostic (24 July 2026)**: the `nomic-embed-text` cache has
  14,817 namespace entries, but only 1,000 unique full MetaQA NormalRAG chunks
  match. The next registry build would need 132,582 new unique vectors
  (roughly 16,573 local embedding requests at a service batch size of eight).
  HybridRAG's NormalRAG indexing is therefore the expected initial bottleneck.
  NormalRAG now checkpoints embeddings every 512 chunks so interruption does
  not discard completed cache work.
- **Done (15 August 2026)**: main full-corpus MetaQA 2-hop matrix collected,
  deterministically scored, converted to 1,000 full-information rows, and
  audited with no warnings: 1,000 questions × 1 repeat × 3 actions = 3,000
  outcomes.
- **Pending execution**: select the frozen 100-question latency subset; reuse
  its 300 compatible main-pilot rows as logical repeat 0; collect only two
  additional repeats (600 new outcomes); score them; and create the derived
  three-repeat latency-variation report. Raw runs remain separate and their
  fingerprints are checked before analysis.
- **Deferred by current scope decision (15 August 2026)**: the additional 600
  latency outcomes will not be collected now. The main pilot latency
  distributions are available, but within-question latency variance remains
  unknown and must not be claimed as measured.
- **Conditional**: after a primary critic is explicitly enabled and validated,
  collect all three critic judgements for at least 500 complete question
  matrices. Semantic metrics are similarly conditional on a configured,
  validated semantic adapter.
- **Blocked**: provisional reward calibration and promotion of critic/semantic
  signals until the plan-required human review exists. No reward weights or
  human labels will be invented.

### M5 — Full training dataset: Partial (foundation verified)

- **Done (15 August 2026)**: grouped, leakage-audited split-manifest builder;
  cross-domain holdout support; atomic assignment/manifest output; data-card
  generator; and a pilot-scale verification. The 1,000-question pilot split
  has 800 train, 100 validation, and 100 locked-test questions across 198
  template groups, with no group leakage. Its data card records candidate
  coverage, failures, latency, and metric availability.
- **Pending**: a 15,000–30,000-question stratified full-information matrix,
  a real multi-dataset cross-domain holdout, configured semantic/critic scores,
  and a full M5 data card. The current pilot split is a tooling verification,
  not permission to train M6/M7 models.

### M6 — Outcome model and oracle imitation: Not started

- **Pending**: multi-task per-action quality/latency/compute/failure model,
  ranking loss, oracle imitation, residual reports, and calibration.

### M7 — Constrained contextual-bandit policy: Not started

- **Pending**: full-information constrained objective, Lagrangian updates,
  constraint curriculum, calibration, and policy checkpoint.

### M8 — Evaluation and ablation: Not started

- **Pending**: baseline/ablation evaluation, paired tests, regret and
  constraint reports, cross-domain checks, and robustness evaluation.

### M9 — Runtime artifact and terminal routing: Not started

- **Final user-facing acceptance criterion**: after M6–M9 are completed, a
  user enters a question in the terminal and receives the selected RAG action,
  budget/uncertainty/fallback decision logs, the generated answer, and request
  telemetry. The terminal CLI must load only a compatibility-validated exported
  router artifact and must never use gold labels or benchmark-only fields.
- **Pending**: runtime feature extraction, policy/artifact loader, budget and
  uncertainty guards, candidate execution, structured terminal output,
  compatibility validation, and telemetry.

### M10 — Controlled rollout: Not started

- **Pending**: promotion gates, canary rollout, rollback triggers, and
  production monitoring.

## Verified engineering check

The 23 July 2026 verification produced 240 outcomes from 40 questions × 2
repeats × 3 actions, 80 complete matrices, zero candidate errors, and no
completeness warnings. Its budget-free oracle chose Optimized GraphRAG for 50
groups, GraphRAG for 20, and Hybrid RAG for 10. These are engineering
diagnostics only: the test used 1,000 text documents and is not a final
benchmark or deployment latency measurement.

## What happens next

1. M4 repeated-latency calibration is deferred by scope; preserve this as a
   known limitation rather than fabricating repeat variance.
2. Reuse the verified M5 manifest/data-card tools when the full collection is
   available; do not call the current pilot split a final evaluation split.
3. If a validated primary critic is configured, collect complete three-action
   critic matrices for at least 500 questions.
4. Obtain 300–500 balanced human reviews before selecting semantic/critic
   reward signals or provisional reward weights.
5. Only then freeze M5 splits and collect the larger training matrix. Do not
   start M6 or M7 before the M4 reward-calibration gate is complete.

## Model-neutral configuration

Model choices remain outside code:

- `configs/router/semantic.yaml`
- `configs/router/critics.yaml`
- `configs/router/m4_pilot.yaml`

An enabled semantic or critic slot requires both an adapter and model ID. This
keeps models replaceable and prevents a hidden default model from becoming part
of the router architecture.
