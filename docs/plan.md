# Production Plan: Quality-Constrained RL Router for Multi-RAG Selection

## 1. Document purpose

This document specifies a production-grade query router that selects one of three retrieval-augmented generation systems for each incoming question:

1. `hybrid_rag`: dense embedding retrieval combined with BM25 through reciprocal-rank fusion.
2. `graph_rag`: the full GraphRAG pipeline.
3. `optimized_graph_rag`: the capsule-based optimized GraphRAG pipeline, which is faster but can lose quality as reasoning paths become deeper.

The router is trained to maximize answer and evidence quality while obeying explicit mean-latency, tail-latency, and compute budgets. It uses deterministic evaluation, semantic answer comparison, calibrated embedding similarity, structured LLM critiques, and constrained reinforcement learning.

The plan is designed for the existing repository. It reuses the current pipeline interfaces, dataset adapters, compute instrumentation, evaluation records, caching, and report conventions.

---

## 2. Decisions, assumptions, and unresolved configuration

### 2.1 Decisions already made

- The optimization target is **quality under a latency and compute budget**.
- Quality must include lexical correctness, semantic answer equivalence, calibrated answer-embedding similarity, retrieval evidence quality, groundedness, and completeness.
- Embedding similarity is retained as a low-weight signal rather than used as the primary definition of correctness.
- LLM critiques are structured supervision. Free-form critique text is not directly treated as a scalar reward.
- The router is trained offline before any online learning is considered.
- Training is organized as importable Python modules plus reproducible Jupyter notebooks.
- The existing RAG implementations remain independently testable. Routing is added above them rather than embedded inside them.

### 2.2 Working assumptions

- The action space contains exactly `hybrid_rag`, `graph_rag`, and `optimized_graph_rag`.
- The repository's vector-only `normal_rag` is excluded from the first production action space. It can remain an evaluation baseline.
- A routing episode contains one question, one selected RAG action, and one terminal outcome. This makes the first router a contextual bandit rather than a multi-step agent.
- Each candidate uses the same answer-generation LLM, system-level generation settings, dataset split, corpus snapshot, and maximum context policy during outcome collection unless a candidate requires a documented exception.
- The router must not receive gold hop count, gold answer, gold evidence, dataset split, or benchmark-only annotations at runtime.
- Exact latency budgets are deployment configuration. They must be measured on the target hardware before a model is promoted.

### 2.3 Decisions that remain configurable rather than architectural blockers

- Mean end-to-end latency budget.
- p95 or CVaR tail-latency budget.
- Compute budget expressed as tokens, GPU milliseconds, estimated energy, or a normalized cost unit.
- The final semantic-equivalence model.
- The primary and secondary LLM critic models.
- The production fallback candidate.
- Whether a later release permits sequential escalation from one RAG system to another.

---

## 3. Product goal

For an incoming question `x`, choose a candidate `a` from the three RAG systems such that expected answer quality is maximized while operational constraints are satisfied.

Formally:

```text
maximize     E[Q(x, a)]

subject to   E[L(x, a)] <= B_mean
             CVaR_0.95(L(x, a)) <= B_tail
             E[C(x, a)] <= B_compute
```

Where:

- `Q(x, a)` is the measured quality of candidate `a` on question `x`.
- `L(x, a)` is end-to-end latency, excluding one-time corpus indexing.
- `C(x, a)` is compute usage.
- `B_mean`, `B_tail`, and `B_compute` are deployment budgets.

The router must learn expected utility, not merely predict whether a question is one-hop, two-hop, or three-hop. Complexity is a latent explanatory factor. The correct action depends on the expected performance difference between systems under the current budget.

---

## 4. Goals and non-goals

### 4.1 Goals

- Route each question among the three candidate RAG pipelines.
- Learn from full counterfactual outcomes by executing all candidates during offline data collection.
- Optimize reference correctness, evidence retrieval, groundedness, and completeness.
- Enforce mean, tail, and compute constraints explicitly.
- Produce interpretable predictions for expected quality, latency, failure probability, uncertainty, and selected action.
- Support reproducible training, evaluation, ablation, calibration, export, and rollback.
- Preserve candidate-specific diagnostic evidence and LLM critiques.
- Generalize across datasets and question templates rather than memorize benchmark hop labels.
- Fail safely when the router model, a candidate, or an upstream dependency is unavailable.

### 4.2 Non-goals for the first release

- Training any of the three RAG pipelines jointly with the router.
- Replacing the existing R-GCN or query projector used inside optimized GraphRAG.
- Allowing the router to change capsule construction or graph indexing parameters per request.
- Performing multi-step escalation after inspecting a candidate's answer.
- Using production feedback without explicit validation and rollback controls.
- Optimizing prose style for short factual QA.
- Treating benchmark hop count as the production routing target.

---

## 5. Why the correct RL formulation is a contextual bandit

The first version makes one decision:

```text
question -> choose one RAG pipeline -> observe terminal quality and cost
```

There is no long action sequence. Standard PPO would add unnecessary variance, tuning difficulty, and failure modes. Because the offline collector can execute all three actions for every training question, training has full-information feedback rather than bandit-only feedback.

The recommended progression is:

1. Train outcome models that estimate quality, latency quantiles, compute, and failure probability for every action.
2. Warm-start a policy by imitating an offline budget-aware oracle.
3. Optimize the policy with a constrained full-information contextual-bandit objective.
4. Calibrate action probabilities and uncertainty.
5. Evaluate against static, heuristic, supervised, and oracle baselines.
6. Consider online contextual-bandit updates only after a safe offline and shadow deployment.

A sequential RL algorithm becomes appropriate only if a future action space includes steps such as `retrieve cheaply`, `inspect confidence`, `escalate`, and `stop`.

---

## 6. Existing repository integration

### 6.1 Candidate interfaces already available

The current implementations expose compatible high-level methods:

- `src.rag.hybrid.pipeline.HybridRAG.retrieve()` and `.answer()`
- `src.g_rag.pipeline.GraphRAG.retrieve()` and `.answer()`
- `src.optimized_grag.pipeline.OptimizedGraphRAG.retrieve()` and `.answer()`

The new router must wrap these APIs through candidate adapters. It must not branch on their internal concrete output classes in the policy layer.

### 6.2 Existing evaluation foundation

`src/evaluation/runner.py` already converts multiple pipeline outputs into `RetrievalRecord`. `src/evaluation/core.py` already provides:

- normalized exact match;
- token F1;
- boolean accuracy;
- context-answer recall;
- supporting-evidence recall;
- evidence Recall@K;
- answer Recall@K;
- evidence MRR;
- capsule Recall@K and MRR;
- mean, p50, and p95 latency summaries.

The router project should extend this foundation rather than create an incompatible evaluator.

### 6.3 Existing compute instrumentation

`src/compute_eval` already records per-stage elapsed time, counts, values, and stage totals. The outcome collector should attach these snapshots to candidate outcomes and add:

- total request time;
- retrieval-only time;
- generation-only time;
- router overhead;
- token usage;
- timeout and exception status;
- optional peak CPU and GPU memory;
- hardware and runtime fingerprint.

### 6.4 Existing datasets

The local repository contains sufficient raw QA volume for a strong first router:

| Dataset | Train | Validation | Test | Important characteristics |
|---|---:|---:|---:|---|
| MetaQA 1/2/3-hop combined | 329,282 | about 39,138 | about 39,093 | Large, graph-native, templated, explicit hop strata |
| 2WikiMultiHopQA | 167,454 | 12,576 | 12,576 | Multi-hop text QA with supporting facts and evidence |
| MultiHop-RAG | 2,556 | no official separate split | no official separate split | Smaller real-world-style collection with evidence lists |

The raw datasets are large enough. The missing artifact is the full candidate-outcome matrix used by the router.

---

## 7. End-to-end architecture

### 7.1 Offline flow

```text
dataset adapters
    -> split and stratified sampler
    -> shared corpus/index snapshot
    -> candidate runner
        -> hybrid_rag outcome
        -> graph_rag outcome
        -> optimized_graph_rag outcome
    -> deterministic metric engine
    -> semantic and contradiction evaluators
    -> structured LLM critic
    -> reward composer
    -> full-information router dataset
    -> outcome and latency models
    -> oracle imitation warm start
    -> constrained contextual-bandit training
    -> calibration and offline policy evaluation
    -> versioned runtime artifact
```

### 7.2 Online flow

```text
question
    -> input validation
    -> cheap feature extraction
    -> router model
        -> predicted quality per candidate
        -> predicted p50/p95 latency per candidate
        -> predicted failure probability per candidate
        -> uncertainty per candidate
        -> policy probabilities
    -> budget feasibility mask
    -> safety and uncertainty guard
    -> selected candidate adapter
    -> existing RAG pipeline
    -> answer and telemetry
```

### 7.3 Service boundary

Use a modular monolith inside the current Python package. Separate modules provide clear contracts, but the first release should not introduce networked microservices. Outcome collection can use worker processes or job queues later without changing core schemas.

---

## 8. Proposed repository structure

```text
.
|-- configs/
|   `-- router/
|       |-- data/
|       |   |-- pilot.yaml
|       |   `-- full.yaml
|       |-- reward/
|       |   |-- deterministic.yaml
|       |   |-- semantic.yaml
|       |   `-- final.yaml
|       |-- critics/
|       |   |-- local_judge.yaml
|       |   `-- ensemble_judge.yaml
|       |-- models/
|       |   |-- outcome_mlp.yaml
|       |   `-- policy_mlp.yaml
|       |-- training/
|       |   |-- supervised.yaml
|       |   `-- constrained_bandit.yaml
|       |-- runtime/
|       |   |-- development.yaml
|       |   `-- production.yaml
|       `-- experiments/
|           |-- deterministic_only.yaml
|           |-- semantic_ablation.yaml
|           |-- critic_ablation.yaml
|           `-- final_router.yaml
|-- src/
|   `-- router/
|       |-- __init__.py
|       |-- config.py
|       |-- constants.py
|       |-- exceptions.py
|       |-- types.py
|       |-- service.py
|       |-- candidates/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- hybrid.py
|       |   |-- graph.py
|       |   |-- optimized_graph.py
|       |   `-- registry.py
|       |-- features/
|       |   |-- __init__.py
|       |   |-- schemas.py
|       |   |-- extractor.py
|       |   |-- lexical.py
|       |   |-- entity.py
|       |   |-- semantic.py
|       |   |-- reasoning_cues.py
|       |   |-- retrieval_probe.py
|       |   `-- cache.py
|       |-- data/
|       |   |-- __init__.py
|       |   |-- schemas.py
|       |   |-- sampling.py
|       |   |-- splits.py
|       |   |-- collector.py
|       |   |-- validation.py
|       |   |-- serialization.py
|       |   |-- manifests.py
|       |   `-- fingerprints.py
|       |-- metrics/
|       |   |-- __init__.py
|       |   |-- normalization.py
|       |   |-- correctness.py
|       |   |-- answer_types.py
|       |   |-- semantic_equivalence.py
|       |   |-- embedding_similarity.py
|       |   |-- contradiction.py
|       |   |-- evidence.py
|       |   |-- faithfulness.py
|       |   |-- completeness.py
|       |   |-- reliability.py
|       |   `-- aggregation.py
|       |-- critics/
|       |   |-- __init__.py
|       |   |-- schemas.py
|       |   |-- prompts.py
|       |   |-- judge.py
|       |   |-- ensemble.py
|       |   |-- cache.py
|       |   |-- adjudication.py
|       |   `-- calibration.py
|       |-- reward/
|       |   |-- __init__.py
|       |   |-- config.py
|       |   |-- components.py
|       |   |-- composer.py
|       |   |-- constraints.py
|       |   |-- normalizers.py
|       |   `-- oracle.py
|       |-- models/
|       |   |-- __init__.py
|       |   |-- encoder.py
|       |   |-- outcome_model.py
|       |   |-- latency_model.py
|       |   |-- policy.py
|       |   |-- uncertainty.py
|       |   |-- calibration.py
|       |   `-- checkpoint.py
|       |-- training/
|       |   |-- __init__.py
|       |   |-- dataset.py
|       |   |-- collator.py
|       |   |-- losses.py
|       |   |-- supervised.py
|       |   |-- imitation.py
|       |   |-- constrained_bandit.py
|       |   |-- dual_optimizer.py
|       |   |-- trainer.py
|       |   |-- callbacks.py
|       |   `-- reproducibility.py
|       |-- evaluation/
|       |   |-- __init__.py
|       |   |-- policy_runner.py
|       |   |-- baselines.py
|       |   |-- regret.py
|       |   |-- constraints.py
|       |   |-- bootstrap.py
|       |   |-- calibration.py
|       |   |-- ablations.py
|       |   |-- robustness.py
|       |   `-- reports.py
|       `-- runtime/
|           |-- __init__.py
|           |-- loader.py
|           |-- router.py
|           |-- budget.py
|           |-- guardrails.py
|           |-- fallback.py
|           |-- circuit_breaker.py
|           `-- telemetry.py
|-- scripts/
|   `-- router/
|       |-- collect_outcomes.py
|       |-- score_outcomes.py
|       |-- build_training_dataset.py
|       |-- train_outcome_model.py
|       |-- train_policy.py
|       |-- calibrate_policy.py
|       |-- evaluate_policy.py
|       |-- run_ablations.py
|       |-- export_router.py
|       `-- validate_artifact.py
|-- notebooks/
|   `-- router/
|       |-- README.md
|       |-- 00_data_and_split_audit.ipynb
|       |-- 01_pilot_outcome_collection.ipynb
|       |-- 02_reward_and_critic_validation.ipynb
|       |-- 03_outcome_model_training.ipynb
|       |-- 04_oracle_imitation_warm_start.ipynb
|       |-- 05_constrained_bandit_training.ipynb
|       |-- 06_policy_evaluation.ipynb
|       |-- 07_ablation_and_error_analysis.ipynb
|       `-- 08_calibration_and_export.ipynb
|-- artifacts/
|   `-- router/
|       |-- datasets/
|       |-- outcomes/
|       |-- critiques/
|       |-- checkpoints/
|       |-- calibration/
|       `-- exported/
|-- reports/
|   `-- router/
|       |-- data_audit/
|       |-- reward_validation/
|       |-- experiments/
|       |-- policy_evaluation/
|       |-- ablations/
|       |-- robustness/
|       `-- production_shadow/
|-- tests/
|   `-- router/
|       |-- unit/
|       |-- integration/
|       |-- contract/
|       |-- regression/
|       `-- fixtures/
`-- docs/
    `-- router/
        |-- architecture.md
        |-- data-contracts.md
        |-- reward-model.md
        |-- training.md
        |-- evaluation.md
        `-- operations.md
```

### 8.1 Structure rules

- Notebooks orchestrate and visualize; reusable training logic lives under `src/router`.
- Scripts are reproducible CLI entry points and call package modules.
- Configurations contain experiment choices and never hide inside notebooks.
- Large outcomes, embeddings, critiques, and checkpoints live under `artifacts/router` and are ignored by Git.
- Small manifests, schema definitions, aggregate reports, and selected plots may be committed.
- Every artifact includes a content fingerprint and a manifest.

---

## 9. Core data contracts

### 9.1 `QuestionRecord`

Required fields:

```text
question_id: string
dataset: string
source_split: string
question: string
accepted_answers: list[string]
gold_evidence: list[GoldEvidenceItem]
answer_type: enum
sampling_strata: map[string, scalar]
group_keys: map[string, string]
dataset_metadata: map[string, value]
```

`source_split`, benchmark hop count, question type, and gold annotations are permitted during dataset construction and evaluation. They are prohibited from runtime feature tensors unless the same field is genuinely available in production.

### 9.2 `CandidateOutcome`

One record per question, candidate, repeat, and run fingerprint:

```text
question_id: string
candidate_id: enum[hybrid_rag, graph_rag, optimized_graph_rag]
repeat_id: integer
run_id: string
pipeline_fingerprint: string
corpus_fingerprint: string
model_fingerprint: string
hardware_fingerprint: string
retrieval_record: RetrievalRecord
generated_answer: string
usage: map[string, value]
retrieval_latency_ms: float
generation_latency_ms: float
end_to_end_latency_ms: float
peak_cpu_memory_mb: optional float
peak_gpu_memory_mb: optional float
input_tokens: optional integer
output_tokens: optional integer
timeout: boolean
error_type: optional string
error_message_hash: optional string
created_at_utc: timestamp
```

Raw exception messages should not be copied into long-lived artifacts if they can contain paths or private content. Store a normalized category and stable hash.

### 9.3 `MetricVector`

```text
exact_or_alias_match: optional float
token_or_set_f1: optional float
boolean_accuracy: optional float
numeric_date_accuracy: optional float
semantic_equivalence: optional float
embedding_similarity_raw: optional float
embedding_similarity_calibrated: optional float
contradiction_probability: optional float
entity_conflict: boolean
numeric_conflict: boolean
polarity_conflict: boolean
supporting_evidence_recall: optional float
context_precision: optional float
evidence_ndcg_at_k: optional float
evidence_mrr: optional float
claim_faithfulness: optional float
answer_completeness: optional float
answer_relevance: optional float
correct_refusal: optional float
failure_penalty: float
```

Missing metrics remain `null`; they must never be silently converted to zero. The reward composer uses answer-type masks and dataset-aware availability masks.

### 9.4 `CritiqueRecord`

```text
question_id: string
candidate_id: string
judge_model: string
judge_prompt_version: string
judge_run_id: string
correctness: float
groundedness: float
evidence_completeness: float
reasoning_completeness: float
answer_relevance: float
answerable_from_context: boolean
hallucination_detected: boolean
failure_type: enum
confidence: float
supporting_claims: list[string]
unsupported_claims: list[string]
missing_information: list[string]
critique: string
raw_response_hash: string
```

The critic is blind to candidate identity. Candidate names are removed from prompts and outputs are shuffled for pairwise adjudication.

### 9.5 `RouterTrainingExample`

```text
question_id: string
question: string
runtime_feature_vector: tensor
quality_by_action: vector[3]
latency_p50_by_action: vector[3]
latency_p95_by_action: vector[3]
compute_by_action: vector[3]
failure_probability_by_action: vector[3]
valid_metric_mask_by_action: matrix
oracle_action: integer
feasible_action_mask: vector[3]
split: enum[train, validation, test]
group_id: string
manifest_fingerprint: string
```

### 9.6 Storage format

- Use Parquet shards for normalized training tables and metric matrices.
- Use compressed JSONL for raw retrieval records and critique records when nested structure must be preserved.
- Use JSON manifests for fingerprints, schemas, row counts, distributions, and provenance.
- Use atomic write-then-rename for completed shards.
- A partially completed collection run must be resumable by `(question_id, candidate_id, repeat_id, run_fingerprint)`.

---

## 10. Data collection strategy

### 10.1 Full-information collection

Every selected training question is executed against all three candidates. This creates a complete outcome vector and avoids relying on inverse-propensity correction for the initial model.

For each question:

```text
for candidate in fixed_candidate_order_shuffled_per_question:
    execute retrieval or end-to-end answer generation
    capture normalized retrieval record
    capture stage timings and compute
    compute deterministic metrics
    enqueue semantic and critic evaluation
```

Candidate execution order is randomized and recorded to reduce systematic warm-cache and load bias.

### 10.2 Collection stages

#### Stage A: smoke collection

- 100 questions.
- Balanced across datasets, answer types, and available hop strata.
- Three candidates per question.
- Two repeated executions per candidate.
- Purpose: validate schemas, cache isolation, prompts, failure handling, and artifact resumption.

#### Stage B: pilot collection

- 1,000 to 2,000 questions.
- Three to five latency repeats for a smaller latency-calibration subset.
- Full deterministic scoring.
- LLM critiques for every candidate on at least 500 questions.
- Purpose: calibrate reward components, estimate action win rates, identify class imbalance, and size the full run.

#### Stage C: first full training collection

- 15,000 to 30,000 stratified questions.
- 45,000 to 90,000 candidate trajectories before repeats.
- Full deterministic and semantic scoring.
- Critic scoring for all failures, action disagreements, near ties, low-confidence cases, and a random control sample.
- Purpose: train the first production candidate.

#### Stage D: targeted enrichment

- Add strata where policy regret or uncertainty is high.
- Add real deployment-domain questions when available.
- Add unanswerable, ambiguous, noisy, temporal, comparative, and adversarial examples.

### 10.3 Stratification

Balance or explicitly weight by:

- dataset;
- source split;
- benchmark hop group for sampling only;
- question template cluster;
- entity cluster;
- answer type;
- question length;
- number of detected entities;
- boolean, temporal, numeric, comparative, and aggregation cues;
- graph entity-link confidence;
- evidence count;
- candidate disagreement in a preliminary run.

### 10.4 Split hygiene

Do not rely only on random question splits. MetaQA is strongly templated, so random splits can leak question patterns and entities.

Use:

- official train, validation, and test boundaries where available;
- group splitting by normalized question template;
- group splitting by key entity identifiers;
- corpus-aware separation where the benchmark supports it;
- an entire held-out dataset for cross-domain evaluation;
- a final locked test manifest that is never used for reward calibration, threshold selection, critic prompt changes, or early stopping.

### 10.5 Outcome comparability requirements

For candidate outcomes to be comparable:

- Use the same question string.
- Use the same corpus and graph snapshot.
- Use the same embedding model version where candidate design permits it.
- Use the same answer-generation model and decoding settings.
- Keep answer prompt requirements aligned: concise, grounded, and explicit about insufficient context.
- Record candidate-specific context limits.
- Exclude indexing time from per-question latency but report indexing separately.
- Separate cold-start from warm steady-state latency.
- Fingerprint every relevant input and configuration.

---

## 11. Runtime feature design

The router may use only information available before the expensive candidate action.

### 11.1 Text representation

- Frozen question embedding from the configured embedding adapter.
- Optional small trainable projection from embedding dimension to 128 or 256 dimensions.
- Question tokens are never sent to the policy as raw benchmark annotations.

### 11.2 Cheap scalar features

- character and token length;
- punctuation and clause count;
- number of named entities;
- exact entity-link count;
- entity-link confidence and ambiguity;
- number of relation cue phrases;
- conjunction and compositional cue counts;
- comparative, temporal, numeric, boolean, counting, superlative, and negation flags;
- expected answer-type probabilities;
- out-of-vocabulary or rare-entity score;
- lexical diversity;
- pronoun and coreference cues;
- number of distinct candidate entities;
- presence of an explicit knowledge-graph seed.

### 11.3 Optional cheap retrieval probes

If measured router overhead remains inside budget, add small fixed probes:

- BM25 top-1 score and top-1-to-top-2 margin;
- dense retrieval top-1 score and margin;
- overlap between sparse and dense top results;
- exact graph seed availability;
- estimated graph neighborhood size for linked seeds.

Probe latency is part of router overhead. A feature is rejected if its incremental policy value does not justify its runtime cost.

### 11.4 Prohibited runtime features

- gold hop count;
- gold answer;
- gold evidence;
- source split name;
- candidate output or candidate critique before the action;
- future latency observed after routing;
- benchmark-specific question type unless an equivalent production classifier generates it from the question.

### 11.5 Feature ablation requirement

Evaluate at least:

1. question embedding only;
2. scalar features only;
3. embedding plus scalar features;
4. embedding, scalars, and retrieval probes.

Promote the smallest feature set whose quality and constraint performance are statistically competitive.

---

## 12. Quality and reward design

### 12.1 Design principles

- Quality is distinct from latency and compute.
- Deterministic gold metrics outrank subjective judgments when trustworthy gold exists.
- Semantic metrics repair false negatives from lexical matching.
- Embedding similarity provides a smooth low-weight signal.
- Contradiction, entity, number, date, relation-direction, and polarity checks prevent false semantic credit.
- Retrieval quality and answer faithfulness remain separate so failure causes are diagnosable.
- Correlated metrics are grouped before the final weighted sum to avoid double counting.
- Scores are calibrated against human review before production use.

### 12.2 Answer-type-aware normalization

Determine a training-time answer type from gold answers and metadata:

- `boolean`;
- `number`;
- `date`;
- `single_entity`;
- `multi_entity_set`;
- `short_span`;
- `long_form`;
- `unanswerable`.

Apply type-specific correctness rules. For example, numeric mismatch overrides high semantic similarity, while set-valued answers require set precision and recall.

### 12.3 Answer correctness component

For ordinary span answers:

```text
C_answer =
    0.35 * exact_or_alias_match
  + 0.20 * token_or_set_f1
  + 0.25 * semantic_equivalence
  + 0.10 * calibrated_embedding_similarity
  + 0.10 * structured_critic_correctness
```

This makes calibrated embedding similarity 10% of answer correctness. Because answer correctness is 50% of the overall quality score, embedding similarity contributes approximately 5% of final quality.

For boolean, numeric, date, and set answers, use a type-specific composer. A wrong polarity or materially wrong number cannot be rescued by semantic or embedding similarity.

### 12.4 Calibrated embedding similarity

For multiple accepted answers:

```text
S_raw = max(cosine(embed(prediction), embed(reference_i)))
```

Calibrate the raw score by answer type:

```text
S_embedding = clip(
    (S_raw - threshold_low[type])
    / (threshold_high[type] - threshold_low[type]),
    0,
    1
)
```

Requirements:

- Learn calibration thresholds from human-labeled equivalent and non-equivalent answer pairs.
- Record both raw and calibrated values.
- Prefer an answer-similarity embedding model over a retrieval-only embedding model.
- If the existing `nomic-embed-text` model is reused, validate it separately for answer equivalence.
- Take the maximum over accepted aliases.
- Set the contribution to zero for hard conflicts.

Hard conflicts include:

- opposite boolean polarity;
- different material numbers, dates, or units;
- incompatible named entities;
- reversed relation direction;
- explicit contradiction;
- correct topic but wrong requested attribute.

### 12.5 Semantic equivalence

Use a question-conditioned cross-encoder or a validated semantic answer similarity model over:

```text
(question, generated_answer, accepted_answer)
```

The evaluator returns equivalence probability plus contradiction probability. BERTScore may be recorded as an auxiliary diagnostic but should not replace answer-type rules.

### 12.6 Evidence quality component

```text
E_evidence =
    0.50 * supporting_evidence_recall
  + 0.25 * context_precision
  + 0.15 * evidence_ndcg_at_k
  + 0.10 * early_evidence_recall
```

Evidence MRR remains an important report metric. It is not added again to the final score if nDCG and early recall already capture ranking quality strongly.

MetaQA lacks full annotated supporting paths, so its endpoint-based evidence proxy must be flagged as weaker supervision. Do not compare the absolute evidence score across datasets without dataset-aware analysis.

### 12.7 Claim-level faithfulness

Split non-trivial generated answers into atomic claims. For each claim, determine whether it is:

- supported by retrieved context;
- contradicted by retrieved context;
- not verifiable from retrieved context.

```text
F_faithfulness = supported_claims / total_material_claims
```

If there are no material claims because the answer is empty or malformed, faithfulness is unavailable and reliability penalties apply.

### 12.8 Completeness

Completeness measures whether the answer covers every requested component and necessary reasoning conclusion. It is separate from faithfulness: an answer can be fully grounded yet incomplete.

Use:

- set recall for multi-answer questions;
- gold fact coverage when gold reasoning facts exist;
- structured critic completeness for long-form or compositional responses;
- missing-step identification for multi-hop answers.

### 12.9 Final quality score

Initial recommended composition:

```text
Q_base =
    0.50 * C_answer
  + 0.20 * F_faithfulness
  + 0.20 * E_evidence
  + 0.10 * M_completeness
```

Initial hard penalties:

```text
P =
    0.50 * material_contradiction
  + 0.40 * unsupported_material_claims
  + 1.00 * unusable_execution_failure
  + 0.50 * incorrect_refusal
```

Final bounded score:

```text
Q = clip(Q_base - P, -1, 1)
```

The numbers above are initialization values. The production weights must be calibrated against human-reviewed outcomes and tested through ablations.

### 12.10 Metrics excluded from direct quality reward

- raw context length;
- raw answer length;
- verbosity;
- fluency when all systems share the same answer LLM;
- context-answer recall as a correctness substitute;
- dataset hop label;
- raw uncalibrated cosine similarity;
- latency and compute, because they are constraints;
- free-form critic rationale text;
- duplicate correlated metrics added independently.

---

## 13. LLM critique system

### 13.1 Critic responsibilities

The critic diagnoses:

- reference correctness;
- answer relevance;
- groundedness in retrieved context;
- evidence completeness;
- reasoning completeness;
- unsupported claims;
- contradictions;
- whether the question was answerable from the supplied context;
- whether refusal was appropriate;
- primary failure location.

### 13.2 Failure taxonomy

Use a closed taxonomy:

```text
none
retrieval_missing_evidence
retrieval_irrelevant_context
retrieval_wrong_entity
retrieval_incomplete_path
generation_ignored_evidence
generation_unsupported_claim
reasoning_missing_step
reasoning_relation_reversal
wrong_number_or_date
wrong_boolean_polarity
ambiguous_question
corpus_missing_information
appropriate_refusal
incorrect_refusal
timeout
runtime_failure
malformed_output
judge_uncertain
```

### 13.3 Prompt controls

- Do not reveal candidate identity.
- Present a stable rubric and JSON schema.
- Separate question, accepted answers, gold evidence, retrieved context, and candidate answer with explicit delimiters.
- Ask the critic to cite exact supporting or conflicting context spans.
- Use deterministic decoding where supported.
- Version every prompt.
- Reject malformed outputs rather than guessing missing fields.
- Cache by complete prompt fingerprint and judge-model fingerprint.

### 13.4 Judge ensemble and adjudication

Use one critic for broad coverage and a second independent critic for:

- near-tie candidate outcomes;
- disagreement between deterministic and critic metrics;
- low-confidence judgments;
- likely contradictions;
- policy-selected high-impact examples;
- a fixed random audit sample.

If critics disagree beyond configured tolerances, run adjudication or mark the score uncertain. Uncertainty reduces the example's training weight rather than silently averaging incompatible judgments.

### 13.5 Human calibration set

Create 300 to 500 manually reviewed candidate outcomes balanced across:

- datasets;
- candidate systems;
- answer types;
- obvious wins;
- near ties;
- contradiction cases;
- unanswerable cases;
- critic disagreement cases.

Measure critic agreement, per-dimension precision and recall, calibration error, and systematic bias. Review candidate ordering, answer length, and judge-model-family effects.

---

## 14. Latency and compute measurement

### 14.1 Latency definitions

Record separately:

- router feature extraction;
- router forward pass;
- retrieval;
- context construction;
- answer generation;
- total end-to-end request;
- cold-start initialization;
- one-time indexing.

The production constraint normally applies to warm end-to-end request latency including router overhead. Cold-start and indexing have separate operational service-level objectives.

### 14.2 Repeated measurement

- Randomize candidate order.
- Warm models and indexes before steady-state measurements.
- Repeat a calibrated subset at least three times.
- Preserve individual repeats rather than only averages.
- Record concurrent load and worker settings.
- Record cache status.
- Run final latency validation on the production-equivalent hardware.

### 14.3 Tail constraint

Use CVaR at 95% as a differentiable training surrogate for p95:

```text
CVaR_0.95(L) = expected latency among the slowest 5% of routed requests
```

Report observed p50, p90, p95, p99, timeout rate, and CVaR. Promotion requires observed constraints, not only predictions from the latency model.

### 14.4 Compute units

Define a normalized compute score from available signals:

```text
C_compute =
    w_gpu * gpu_time_ms
  + w_cpu * cpu_time_ms
  + w_in * input_tokens
  + w_out * output_tokens
  + w_mem * peak_memory_mb_seconds
```

If some signals are unavailable, the manifest records the reduced definition. Compute definitions cannot change silently between experiments.

---

## 15. Router model architecture

### 15.1 Recommended first model

Use a compact multi-task neural model:

```text
question embedding
    -> projection MLP

scalar and categorical runtime features
    -> normalized feature MLP

concatenation
    -> shared trunk
        -> quality head: 3 actions
        -> latency quantile head: 3 actions x quantiles
        -> compute head: 3 actions
        -> failure-probability head: 3 actions
        -> uncertainty head or ensemble outputs
        -> residual policy-logit head: 3 actions
```

Recommended initial scale:

- frozen text embedding encoder;
- embedding projection width 256;
- scalar feature projection width 64;
- shared trunk widths 256 and 128;
- dropout between 0.05 and 0.20 selected on validation;
- three action outputs per outcome head;
- less than a few million trainable parameters excluding the frozen encoder.

This is fast enough for routing and expressive enough to learn non-linear action differences.

### 15.2 Why predict outcomes as well as policy logits

Outcome heads provide:

- interpretable expected quality and cost;
- budget feasibility masks;
- better supervision from all three candidate outcomes;
- safer fallback when direct policy logits are uncertain;
- easier debugging and calibration;
- support for new deployment budgets without fully retraining the representation.

### 15.3 Uncertainty

Recommended order:

1. Deep ensemble of three to five small router models for final experiments.
2. Temperature-scaled policy probabilities.
3. Per-action prediction intervals from ensemble variance and residual calibration.
4. Optional conformal calibration for quality and latency bounds.

At runtime, high uncertainty triggers a configured conservative selection rule or fallback candidate.

---

## 16. Training procedure

### 16.1 Phase 0: reproducibility setup

- Freeze train, validation, and test manifests.
- Record random seeds.
- Fingerprint code, data, corpus, RAG configs, embedding model, answer LLM, critic models, prompts, and hardware.
- Validate that no runtime feature contains forbidden annotations.
- Validate complete action outcomes for training examples.

### 16.2 Phase 1: train outcome predictors

Train the multi-task model using all action outcomes.

Loss components:

```text
L_outcome =
    w_q * masked_huber(predicted_quality, observed_quality)
  + w_rank * pairwise_action_ranking_loss
  + w_lat * quantile_latency_loss
  + w_compute * masked_huber(predicted_compute, observed_compute)
  + w_fail * binary_cross_entropy(predicted_failure, observed_failure)
```

The pairwise loss teaches which action is better even when absolute reward calibration is imperfect.

### 16.3 Phase 2: construct budget-aware oracle labels

For each training example:

```text
feasible_actions = actions satisfying configured latency and compute limits

if feasible_actions is not empty:
    oracle_action = action with highest observed quality among feasible actions
else:
    oracle_action = configured least-violation action
```

Use repeated latency estimates or conservative quantiles rather than one noisy measurement.

### 16.4 Phase 3: oracle imitation warm start

Train policy logits with weighted cross-entropy against the oracle action.

Weight examples by:

- oracle advantage over the second-best feasible action;
- outcome confidence;
- critic confidence;
- metric availability;
- inverse sampling probability when stratified sampling changes natural frequencies.

Near ties receive lower classification weight because either action is acceptable.

### 16.5 Phase 4: constrained full-information contextual-bandit optimization

For policy `pi(a | x)`:

```text
J_quality = mean_x sum_a pi(a | x) * Q(x, a)

G_mean = mean_x sum_a pi(a | x) * L(x, a) - B_mean
G_tail = CVaR_surrogate(pi, L) - B_tail
G_compute = mean_x sum_a pi(a | x) * C(x, a) - B_compute

J =
    J_quality
  - lambda_mean * G_mean
  - lambda_tail * G_tail
  - lambda_compute * G_compute
  - beta_kl * KL(pi || pi_warm_start)
  + beta_entropy * entropy(pi)
```

Dual variables are non-negative and updated by projected gradient ascent:

```text
lambda_i = max(0, lambda_i + dual_lr * G_i)
```

Because all action rewards are known, compute the expectation over all three actions directly. Do not sample one action for the training loss unless deliberately simulating partial feedback.

### 16.6 Constraint curriculum

Recommended order:

1. Train with relaxed budgets until quality learning stabilizes.
2. Move budgets gradually toward production targets.
3. Increase dual penalties when validation violations persist.
4. Early-stop using a feasible-validation criterion rather than unconstrained reward.

### 16.7 Model selection criterion

A checkpoint is eligible only if it meets all validation constraints. Among eligible checkpoints, select the one with highest mean quality. If none are feasible, select no production model and continue tuning or data collection.

### 16.8 Calibration

After model selection:

- temperature-scale policy logits;
- calibrate per-action quality residuals;
- calibrate latency quantiles;
- measure expected calibration error and Brier score;
- choose uncertainty thresholds using validation data only;
- freeze calibration artifacts before test evaluation.

---

## 17. Notebook workflow

Notebook cells must call importable package functions. Core algorithms must not exist only inside notebooks.

### `00_data_and_split_audit.ipynb`

- Load all dataset manifests.
- Show counts and answer-type distributions.
- Cluster and inspect question templates.
- Audit entity overlap across splits.
- Build grouped train, validation, cross-domain, and locked-test manifests.
- Export data-audit report.

### `01_pilot_outcome_collection.ipynb`

- Load the pilot manifest.
- Build all candidate pipelines with shared fingerprints.
- Run or resume outcome collection.
- Inspect latency distributions and failures.
- Verify that all three candidate outcomes exist per question.
- Export pilot collection manifest.

### `02_reward_and_critic_validation.ipynb`

- Compute deterministic metrics.
- Fit embedding-similarity calibration.
- Evaluate semantic-equivalence and contradiction models.
- Run blinded critic scoring.
- Compare critic scores with human labels.
- Analyze correlated metrics.
- Choose preliminary reward weights and hard guards.

### `03_outcome_model_training.ipynb`

- Load normalized full-information records.
- Train quality, latency, compute, and failure heads.
- Plot per-action residuals and calibration.
- Evaluate pairwise action ranking accuracy.
- Save a versioned outcome-model checkpoint.

### `04_oracle_imitation_warm_start.ipynb`

- Apply budget configurations.
- Construct feasible masks and oracle actions.
- Train the warm-start policy.
- Inspect confusion matrices and per-stratum regret.
- Save warm-start checkpoint.

### `05_constrained_bandit_training.ipynb`

- Load the warm start.
- Train primal policy and dual constraint variables.
- Plot quality, mean latency, CVaR, compute, entropy, and dual variables.
- Perform validation feasibility checks.
- Save all eligible checkpoints and select the best feasible one.

### `06_policy_evaluation.ipynb`

- Compare all baselines.
- Compute quality, cost, regret, constraints, and action mix.
- Produce paired bootstrap confidence intervals.
- Evaluate in-domain, template-held-out, entity-held-out, and cross-domain splits.
- Run the final locked-test evaluation once the configuration is frozen.

### `07_ablation_and_error_analysis.ipynb`

- Run feature ablations.
- Run reward-component ablations.
- Remove critic supervision.
- Remove embedding similarity.
- Vary embedding weight.
- Compare imitation-only and constrained-bandit models.
- Analyze high-regret, high-uncertainty, and budget-violation cases.

### `08_calibration_and_export.ipynb`

- Fit final calibration artifacts.
- Select uncertainty and fallback thresholds.
- Validate artifact compatibility.
- Export runtime bundle.
- Generate a model card and promotion report.

---

## 18. Evaluation plan

### 18.1 Required baselines

- Always `hybrid_rag`.
- Always `graph_rag`.
- Always `optimized_graph_rag`.
- Always `normal_rag` as a non-action reference if operationally practical.
- Uniform random routing.
- Cheapest-feasible static routing.
- Highest-quality static feasible routing.
- Rule-based complexity heuristic.
- Supervised oracle classifier without constrained fine-tuning.
- Outcome-model argmax with a hard budget mask.
- Full-information oracle upper bound.

### 18.2 Policy metrics

- mean quality;
- median quality;
- quality by dataset and answer type;
- oracle action agreement;
- mean and p95 regret;
- action distribution;
- coverage of each action;
- policy entropy;
- uncertainty and selective-risk curves;
- failure and fallback rate.

### 18.3 Constraint metrics

- mean end-to-end latency;
- p50, p90, p95, and p99 latency;
- latency CVaR at 95%;
- compute per question;
- percentage of requests exceeding individual latency thresholds;
- mean and tail budget violation magnitude;
- router overhead;
- timeout and runtime-error rate.

### 18.4 Statistical analysis

- Use paired bootstrap resampling by question.
- Preserve group structure for template- or entity-grouped evaluation.
- Report 95% confidence intervals.
- Use paired comparisons because every system is evaluated on the same questions.
- Correct for multiple comparisons in large ablation tables.
- Report effect sizes, not only significance.

### 18.5 Generalization slices

- MetaQA one-hop, two-hop, and three-hop evaluation strata;
- each 2Wiki question type;
- each MultiHop-RAG question type;
- short versus long questions;
- linked versus unlinked entities;
- low versus high entity ambiguity;
- numerical, temporal, boolean, comparative, and set-valued answers;
- in-domain versus held-out templates;
- in-domain versus held-out entities;
- held-out dataset;
- answerable versus unanswerable;
- low versus high router uncertainty.

### 18.6 Required ablations

- no calibrated embedding similarity;
- embedding similarity at 2%, 5%, and 10% of final quality;
- raw cosine versus calibrated cosine;
- no contradiction guard;
- no LLM critic;
- deterministic metrics only;
- no evidence component;
- no faithfulness component;
- no retrieval probes;
- embedding-only features;
- scalar-only features;
- imitation only;
- constrained optimization without warm start;
- mean-latency constraint only;
- mean plus tail constraints;
- single model versus uncertainty ensemble.

---

## 19. Production runtime policy

### 19.1 Selection logic

1. Validate and normalize the question.
2. Extract cached and cheap runtime features.
3. Predict per-action quality, latency quantiles, compute, failure probability, and uncertainty.
4. Mask actions predicted to violate hard request-level limits using conservative bounds.
5. Apply the learned policy over feasible actions.
6. If uncertainty is above threshold, apply the configured conservative decision rule.
7. Execute the selected candidate.
8. Emit telemetry with model and pipeline fingerprints.

### 19.2 Conservative uncertainty rule

Recommended rule:

```text
choose the feasible action with the best lower confidence bound on quality
```

If no action is confidently feasible, use the validated static fallback candidate and record `fallback_reason=no_confident_feasible_action`.

### 19.3 Circuit breakers

Maintain per-candidate health state from:

- recent timeout rate;
- recent failure rate;
- latency-window violations;
- index or artifact incompatibility;
- dependency health;
- memory pressure.

An unhealthy candidate is temporarily removed from the feasible action set. Recovery requires successful probes and a cooldown period.

### 19.4 Artifact compatibility

The runtime loader rejects an artifact unless all required values match:

- router schema version;
- feature schema fingerprint;
- action registry and ordering;
- question embedding namespace and dimension;
- normalization statistics;
- policy checkpoint hash;
- calibration artifact hash;
- reward-version metadata;
- supported candidate IDs;
- minimum runtime package version.

### 19.5 Fallback behavior

Fallback candidate selection is based on validation measurements under the production budget. It is not hard-coded by assumption.

Fallback is used when:

- router artifact loading fails;
- feature extraction fails;
- all actions are masked;
- uncertainty is excessive;
- selected candidate is unhealthy;
- telemetry dependencies fail in a way that threatens request execution.

The user request must still complete if non-essential telemetry fails.

---

## 20. Observability and operations

### 20.1 Per-request telemetry

Record:

- request ID and privacy-safe question hash;
- router model version;
- feature schema version;
- predicted action probabilities;
- predicted quality, p50/p95 latency, compute, failure, and uncertainty per action;
- feasible action mask;
- selected and executed candidate;
- fallback or circuit-breaker reason;
- observed stage timings;
- observed token usage and compute;
- execution status;
- delayed quality signal when available.

Raw question text should follow the project's privacy and retention policy. Hashes must be salted or keyed if dictionary attacks are a concern.

### 20.2 Dashboards

- traffic and action mix;
- mean and tail latency by action;
- router overhead;
- timeout and failure rates;
- constraint violation rates;
- predicted versus observed latency;
- predicted versus delayed quality;
- uncertainty distribution;
- fallback and circuit-breaker rate;
- input and feature drift;
- quality and regret on a recurring labeled audit set.

### 20.3 Drift detection

Monitor:

- question embedding distribution;
- scalar feature distributions;
- entity-link success rate;
- action distribution;
- per-action latency residuals;
- outcome model calibration;
- disagreement with a shadow policy;
- recurring failure taxonomy counts.

Drift triggers investigation and targeted data collection, not automatic unreviewed retraining.

---

## 21. Testing strategy

### 21.1 Unit tests

- answer normalization and aliases;
- boolean, numeric, date, and set scoring;
- embedding-score calibration;
- contradiction and hard-conflict guards;
- evidence precision, recall, nDCG, and masks;
- reward composition with missing metrics;
- constraint calculations and CVaR;
- oracle action selection;
- dual-variable projection;
- feature prohibitions;
- schema serialization and migration;
- artifact fingerprint validation;
- circuit-breaker transitions.

### 21.2 Contract tests

Each candidate adapter must satisfy the same contract:

- stable candidate ID;
- `retrieve(question)` support;
- `answer(question)` support;
- normalized `RetrievalRecord` conversion;
- usage and instrumentation extraction;
- typed failure mapping;
- deterministic fingerprint payload.

### 21.3 Integration tests

- one question through all candidates and the collector;
- resume an interrupted collection;
- score and join all outcomes;
- build one training batch;
- run one optimizer step for each training phase;
- export and reload a router artifact;
- route a question through the actual selected pipeline;
- activate fallback when one candidate fails;
- reject incompatible artifacts.

### 21.4 Regression tests

Maintain fixed small fixtures with expected:

- reward components;
- action rankings;
- feasible masks;
- policy outputs within tolerances;
- latency report schema;
- critic JSON parsing behavior;
- no leakage of forbidden features.

### 21.5 Statistical regression tests

On a fixed small evaluation slice, fail CI or promotion when:

- quality decreases beyond the approved tolerance;
- mean or tail latency exceeds its approved tolerance;
- action collapse occurs unexpectedly;
- calibration error increases materially;
- fallback or failure rate rises materially.

---

## 22. Security, privacy, and data governance

- Treat real user questions, retrieved context, and generated answers as potentially sensitive.
- Define retention periods separately for raw text, hashes, features, outcomes, and aggregate metrics.
- Redact secrets and personal identifiers before critic calls where required.
- Do not send private data to an external judge without explicit authorization.
- Store model and prompt provenance for every critique.
- Prevent critic prompt injection by delimiting candidate content and instructing the judge to treat it as untrusted data.
- Validate JSON against a strict schema.
- Limit artifact permissions and avoid serializing executable Python objects from untrusted sources.
- Load PyTorch weights with safe loading options supported by the installed version.
- Record dataset and model licenses in the artifact manifest.

---

## 23. Rollout and rollback

### 23.1 Rollout stages

1. **Offline only:** no production effect.
2. **Shadow mode:** router predicts actions while the existing static policy serves requests.
3. **Replay analysis:** compare predicted choices with measured candidate outcomes on logged samples.
4. **Canary:** route a small approved traffic fraction.
5. **Progressive rollout:** increase traffic only after quality and constraint gates pass.
6. **Full deployment:** retain static fallback and instant artifact rollback.

### 23.2 Promotion gates

A router can be promoted only if:

- it meets mean, tail, and compute budgets on the locked evaluation manifest;
- it is statistically non-inferior in quality to the best static feasible baseline within a predeclared margin;
- it improves quality, compute, or both in a practically meaningful way;
- no critical dataset or answer-type slice regresses beyond its tolerance;
- critic and reward calibration reports are approved;
- artifact compatibility tests pass;
- failure and fallback rates are within limits;
- shadow telemetry matches offline assumptions.

### 23.3 Rollback triggers

- sustained latency-budget breach;
- increased timeout or error rate;
- candidate health instability;
- action collapse;
- significant quality regression on labeled audits;
- feature or outcome drift beyond thresholds;
- artifact mismatch;
- telemetry indicating forbidden or missing features.

Rollback switches to the validated static fallback and preserves incident evidence.

---

## 24. Implementation milestones

### Milestone 0: lock contracts and configuration

Deliverables:

- action registry;
- router configuration schema;
- data contracts;
- artifact manifest schema;
- initial budget configuration interface;
- test fixtures.

Exit criteria:

- schemas serialize and validate;
- candidate IDs and ordering are immutable within a schema version;
- forbidden runtime features are documented and tested.

### Milestone 1: candidate adapters and shared collector

Deliverables:

- three candidate adapters;
- normalized outcome conversion;
- resumable collection;
- timing and failure capture;
- collector CLI.

Exit criteria:

- 100-question smoke matrix is complete;
- every question has all candidate outcomes;
- resume and fingerprint invalidation tests pass.

### Milestone 2: metric and reward engine

Deliverables:

- answer-type normalization;
- deterministic correctness;
- evidence precision, recall, and nDCG;
- semantic equivalence;
- calibrated embedding similarity;
- contradiction guards;
- reward composer.

Exit criteria:

- unit tests cover hard conflicts and missing metrics;
- embedding calibration report is produced;
- reward components remain individually inspectable.

### Milestone 3: critic pipeline

Deliverables:

- strict JSON rubric;
- prompt versioning;
- judge cache;
- disagreement and adjudication logic;
- human calibration workflow.

Exit criteria:

- malformed responses are safely rejected;
- critic agreement and calibration meet predeclared thresholds;
- candidate identity is absent from judge prompts.

### Milestone 4: pilot outcome dataset

Deliverables:

- 1,000 to 2,000 question pilot;
- latency repeat study;
- quality and candidate-win analysis;
- revised sample-size estimate.

Exit criteria:

- no unexplained missing outcomes;
- latency measurement variance is characterized;
- candidate action imbalance is quantified;
- reward weights are provisionally calibrated.

### Milestone 5: full training dataset

Deliverables:

- 15,000 to 30,000-question outcome matrix;
- semantic and critic scores;
- grouped train, validation, cross-domain, and locked-test manifests;
- data card.

Exit criteria:

- schema and fingerprint validation passes;
- split leakage audit passes;
- action and failure coverage is adequate.

### Milestone 6: outcome model and imitation baseline

Deliverables:

- multi-task outcome model;
- budget-aware oracle labels;
- imitation policy;
- calibration and residual reports.

Exit criteria:

- outcome predictions beat simple mean-per-action baselines;
- pairwise action ranking is useful across all major strata;
- imitation policy beats static and heuristic routing where expected.

### Milestone 7: constrained bandit policy

Deliverables:

- primal-dual trainer;
- mean, tail, and compute constraints;
- checkpoint feasibility selection;
- training notebook and reports.

Exit criteria:

- at least one validation checkpoint is feasible;
- the constrained policy improves over imitation or provides a documented reason why imitation remains preferable.

### Milestone 8: evaluation and ablation

Deliverables:

- all required baselines;
- confidence intervals;
- reward, feature, training, and constraint ablations;
- error taxonomy report;
- cross-domain evaluation.

Exit criteria:

- the value of semantic equivalence, low-weight embeddings, critics, and constraints is empirically established;
- no hidden dependency on gold hop labels exists.

### Milestone 9: runtime artifact and shadow deployment

Deliverables:

- exported router bundle;
- loader and compatibility checks;
- telemetry;
- fallback and circuit breakers;
- shadow report.

Exit criteria:

- router overhead is within budget;
- shadow action and latency distributions match offline expectations;
- rollback is tested.

### Milestone 10: controlled production rollout

Deliverables:

- canary deployment;
- promotion dashboard;
- operational runbook;
- retraining and rollback policy.

Exit criteria:

- promotion gates pass under real load;
- quality audit and constraint compliance are stable.

---

## 25. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| MetaQA template leakage | Inflated test performance | Template- and entity-grouped splits plus held-out datasets |
| Optimized GraphRAG degradation at high hops | Wrong cheap routing | Outcome prediction, completeness supervision, deep-reasoning slices, conservative uncertainty |
| Raw embedding similarity rewards wrong entities or numbers | Reward corruption | Low weight, per-type calibration, hard conflict and contradiction guards |
| LLM critic bias | Noisy or systematically skewed rewards | Blind prompts, structured dimensions, second judge on difficult cases, human calibration |
| Double-counted correctness metrics | Overweighted answer surface similarity | Compose correlated metrics inside one correctness block |
| Latency noise | Unstable feasibility labels | Repeated measurements, random execution order, quantile models, hardware fingerprints |
| Action collapse | Router selects one system almost always | Entropy during training, stratified data, action coverage reports, compare with static policy honestly |
| Candidate configuration drift | Invalid historical labels | Full fingerprints and automatic outcome invalidation |
| Judge or semantic model leakage from test tuning | Optimistic evaluation | Freeze calibration before locked-test execution |
| Outcome model overconfidence | Unsafe routing | Ensembles, residual calibration, lower-confidence-bound selection, fallback |
| Missing real-domain questions | Benchmark-only router | Add deployment-domain unlabeled questions and selectively label counterfactual outcomes |
| Collection cost | Excess model calls | Pilot sizing, cached outcomes, selective critic use, resume support |
| Partial collection corruption | Biased training table | Atomic shards, completeness validation, idempotent keys |
| Online feedback loops | Policy reinforces its own choices | Preserve randomized audit traffic if online learning is introduced and require offline review |

---

## 26. Minimum production acceptance checklist

### Data

- [ ] All three candidate outcomes exist for every training question.
- [ ] Dataset, template, and entity group leakage audits pass.
- [ ] Raw and derived artifacts have manifests and fingerprints.
- [ ] Collection order, cache state, hardware, and failures are recorded.
- [ ] A locked test manifest exists.

### Reward

- [ ] Answer-type rules are active.
- [ ] Semantic equivalence is validated.
- [ ] Embedding similarity is calibrated and contributes approximately 5% of final quality unless an ablation supports another value.
- [ ] Entity, number, date, relation, and polarity conflicts nullify inappropriate semantic credit.
- [ ] Evidence quality and faithfulness remain separately observable.
- [ ] Reward weights are calibrated against human-reviewed examples.

### Critics

- [ ] Candidate identity is blinded.
- [ ] Outputs validate against a strict schema.
- [ ] Prompt and model versions are recorded.
- [ ] Low-confidence and disagreement cases are handled.
- [ ] Human calibration metrics are reported.

### Training

- [ ] Outcome-model baselines are beaten.
- [ ] Oracle imitation baseline is available.
- [ ] Constrained optimization uses full action outcomes.
- [ ] Dual variables and constraint curves are logged.
- [ ] Only feasible validation checkpoints are eligible.
- [ ] Calibration artifacts are frozen before final testing.

### Evaluation

- [ ] Static, heuristic, supervised, constrained, and oracle baselines are compared.
- [ ] Mean and tail constraints are evaluated on observed outcomes.
- [ ] Paired confidence intervals are reported.
- [ ] Cross-domain and difficult-slice results are reported.
- [ ] Required reward and feature ablations are complete.

### Runtime

- [ ] Router overhead is measured.
- [ ] Artifact compatibility checks reject invalid bundles.
- [ ] Fallback and circuit breakers are tested.
- [ ] Shadow mode passes.
- [ ] Rollback procedure is verified.

---

## 27. Recommended immediate execution order

1. Confirm the exact three-action registry and exclude `normal_rag` from routing while retaining it as a baseline.
2. Define a first experimental mean, tail, and compute budget using measured candidate latency on the target machine.
3. Implement schemas, fingerprints, and the candidate adapter contract.
4. Build the 100-question smoke outcome matrix.
5. Implement deterministic, semantic, embedding, contradiction, evidence, and reliability metrics.
6. Build the structured critic and manually review its first 100 judgments.
7. Run the 1,000-to-2,000-question pilot.
8. Calibrate reward components and estimate the required full collection size.
9. Build the grouped training and locked evaluation manifests.
10. Collect the 15,000-to-30,000-question full-information dataset.
11. Train the outcome model and imitation baseline.
12. Train and tune the constrained contextual-bandit policy.
13. Complete ablations, cross-domain tests, calibration, and locked-test evaluation.
14. Export the runtime artifact and begin shadow deployment.

---

## 28. Research references guiding the metric design

- Risch, Moeller, Gutsch, and Pietsch, **Semantic Answer Similarity for Evaluating Question Answering Models**: https://aclanthology.org/2021.mrqa-1.15/
- Zhang, Kishore, Wu, Weinberger, and Artzi, **BERTScore: Evaluating Text Generation with BERT**: https://arxiv.org/abs/1904.09675
- Es, James, Espinosa-Anke, and Schockaert, **RAGAS: Automated Evaluation of Retrieval Augmented Generation**: https://aclanthology.org/2024.eacl-demo.16/
- Saad-Falcon, Khattab, Potts, and Zaharia, **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**: https://aclanthology.org/2024.naacl-long.20/
- Zheng and colleagues, **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**: https://arxiv.org/abs/2306.05685

---

## 29. Final architectural recommendation

Build the first release as a modular offline-trained contextual-bandit router with full counterfactual supervision. Use a compact multi-task network to predict quality, latency quantiles, compute, and failures for all three systems, then optimize a calibrated policy with explicit mean, tail, and compute constraints.

Keep deterministic metrics as the anchor. Add cross-encoder semantic equivalence, retain calibrated embedding similarity at approximately 5% of total quality, use strict contradiction guards, and add structured LLM critiques for groundedness and completeness. Validate every subjective signal against a human-reviewed calibration set.

This design is more stable, interpretable, and data-efficient than beginning with generic PPO. It also leaves a clean path to sequential escalation or online contextual-bandit learning after the one-step production router is proven safe.
