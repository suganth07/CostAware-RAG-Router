# Semantic Metrics and Structured Critic

This stage improves the router's reward signal. The initial deterministic
metrics are valuable but cannot recognize every valid paraphrase and cannot
reliably separate a fluent answer from one that is grounded in retrieved
evidence. Two additional signals address those gaps, while keeping their
limitations visible.

## 1. Calibrated answer-embedding similarity

For a generated answer `a` and all accepted answers `g`, the system embeds
`[a, *g]` with the same local embedding model and keeps the maximum cosine
similarity:

```text
raw_similarity = max cosine(embed(a), embed(g_i))
```

This is useful for paraphrases such as `The film was directed by X` versus
`X`. It is **not** correctness by itself: related-but-wrong entities can have
a high cosine score. Therefore the raw value is recorded separately and is not
used by `compose_quality`.

Before it can influence reward, create a manually checked calibration set for
each relevant answer type (at minimum `span`, `boolean`, and `numeric/date`).
For each type choose a lower and upper similarity boundary from the labelled
examples. The calibrated value is linearly clipped into `[0, 1]`. Exact
numeric/date and boolean-polarity conflicts always force it to `0.0`.

Once the calibration is approved, `scripts/router/enrich_embedding_metrics.py`
adds both fields to a copy of scored JSONL. Without `--calibration`, it stores
only the raw diagnostic score and quality does not change. Its small JSONL
cache means interrupted local embedding runs resume without recomputing prior
outcomes.

## 2. Candidate-blind structured critic

`scripts/router/score_critics.py` asks a local Ollama judge to inspect one
question, the accepted answers, gold evidence, retrieved context, and the
generated answer. It intentionally does **not** send `hybrid_rag`,
`graph_rag`, or `optimized_graph_rag` to the judge. This prevents direct route
preference from contaminating the label.

The response must be exactly one JSON object with scores for correctness,
groundedness, evidence completeness, reasoning completeness, answer
relevance, confidence, hallucination detection, a failure taxonomy, and short
claim lists. The parser rejects missing fields, unknown fields, out-of-range
scores, prose around JSON, and unrecognised failure types. Parsing failures go
to a separate `*.failures.jsonl` artifact rather than entering training data.

The only narrow repair is an empty `failure_type` returned by the local judge
alongside an otherwise valid record. It is stored as `failure_type: "other"`
and `failure_type_normalized: true`; the code never guesses a more specific
failure category. This flag must be included in the critic audit because a
high repair rate is evidence that the judge prompt/model needs improvement.

Each successful row includes the question/action/repeat/run join key, judge
model, prompt version, critic run ID, and a SHA-256 hash of the raw response.
That is enough to reproduce and audit a judgement without treating the LLM's
free-text reasoning as a hidden label source.

## 3. Required human check before reward use

The next command creates a **100-record pilot judgement set**, balanced across
the three actions. It does not automatically make the critic a source of
truth. Review a random sample first and record at least:

- whether `correctness >= 0.5` agrees with a human label;
- whether the cited support is actually in the supplied context;
- the rate of malformed/failed outputs and the distribution of failure types;
- disagreement by action, answer type, and question template.

Only after this audit and a larger calibrated set should critic fields be
joined into a promoted reward/training artifact. Until then the existing
unconstrained training JSONL remains the correct artifact for inspecting full
information outcomes, not for training the final learned router.

`scripts/router/analyze_critics.py` generates both an aggregate JSON report
and a balanced blind-review queue. The queue omits the candidate ID so a human
reviewer sees the same information as the critic without being biased by the
route name. Fill in its `human_review_template` fields before deciding whether
to calibrate or promote the critic.

The audit reports two deterministic diagnostics. Exact-match agreement is
intentionally narrow. Accepted-answer coverage additionally checks whether a
verbose generated answer contains every expected answer, which is necessary
for MetaQA multi-answer questions such as an answer sentence containing two
correct actor names.

### Current pilot decision

The `router-critic-v1` pilot must **not** be promoted into the reward. Its
audit showed zero precision against the deterministic exact-match proxy and a
34% empty-taxonomy repair rate, concentrated in GraphRAG outcomes. Version 2
therefore makes an explicit distinction between global correctness against
accepted answers and groundedness in the retrieved context. Version 3 makes
this separation operational through two candidate-blind judge calls: the
correctness call has no retrieved context, while the grounding call has no
accepted answers. It must be
collected into a new artifact and audited separately; never mix versions in a
training dataset.

## 4. Why this comes before the bandit

A contextual bandit can only learn the preferences encoded in its reward. If
the reward treats an unsupported but plausible answer as correct, the policy
will learn that mistake very efficiently. Calibrating semantic metrics and
checking the critic first gives the later multi-task outcome model a more
trustworthy target.
