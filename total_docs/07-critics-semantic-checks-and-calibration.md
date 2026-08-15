# 07 — Critics, semantic checks, and human calibration

Some quality questions cannot be answered safely by exact string matching:

- Is “the 1994 movie” equivalent to the accepted title?
- Does the answer make claims unsupported by the retrieved context?
- Is the answer incomplete even though one answer word matches?

The project has interfaces for these richer signals, but deliberately does not
pretend that an unvalidated model is ground truth.

## Semantic equivalence and embedding similarity

The architecture supports a configurable semantic-equivalence scorer and a
calibrated answer-embedding similarity metric. They are separated because raw
embedding cosine can be misleading: two answers can sound similar while naming
the wrong person, date, or number.

Current safeguards:

1. Hard entity/numeric/polarity conflicts are detected deterministically.
2. Raw embedding similarity is not automatically used as reward.
3. Calibration requires manually reviewed equivalent/non-equivalent answer
   pairs, ideally 300–500 balanced examples.
4. `configs/router/semantic.yaml` starts disabled with no model ID.

## The structured critic

The critic is an optional LLM evaluator. It sees the question, accepted answer,
retrieved context, and generated answer—but not the candidate name. This
candidate blinding prevents it from learning “GraphRAG is usually better” as a
shortcut.

For one candidate outcome, the current critic performs two separate calls:

```text
1. Answer correctness without context
2. Evidence grounding and completeness with context
```

It must return strict JSON fields such as correctness, groundedness, evidence
completeness, reasoning completeness, relevance, confidence, hallucination,
failure type, supported claims, and missing information. Invalid JSON or an
unknown taxonomy value is rejected or visibly normalized by a narrow rule.

## Caching and versions

Critic responses are cached in SQLite using the full prompt, system prompt,
stage, model fingerprint, and prompt version. The artifact also records a
judge run ID. Therefore, identical compatible runs can resume without paying
for identical judge calls, while a changed prompt/model creates fresh results.

## Primary, secondary, disagreement, and adjudication

The implemented M3 foundation provides:

```text
primary critic slot
secondary critic slot
disagreement detection
uncertainty weighting
adjudication interface
```

These are configuration contracts, not a hidden default model selection.
`configs/router/critics.yaml` leaves both slots disabled. A secondary critic
may run for every sample or only uncertain/audit samples, depending on future
configuration.

## Why human review is still required

An LLM critic can be biased or confidently wrong. The existing 100-record V3
critic artifact is useful for testing the pipeline, but it is not permission to
put critic scores into the final reward.

Before promoting semantic or critic signals, the plan requires a balanced,
manually reviewed calibration set. Human labels determine whether the chosen
critic/semantic model is accurate enough and how much influence it should have.

Until then, optional values remain null or diagnostic-only.
