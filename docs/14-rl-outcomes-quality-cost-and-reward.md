# RL Outcome Data, Quality, Cost, and Reward

## The outcome file is the RL training dataset

`scripts/build_route_outcomes.py` writes JSON Lines (`.jsonl`). Each line is
one pair of a question and a route. Every question must therefore appear three
times: once for `hybrid_rag`, `graph_rag`, and `optimized_grag`.

Important fields include:

```json
{
  "sample_id": "metaqa:2-hop:vanilla:train:0",
  "question": "...",
  "route": "optimized_grag",
  "features": {"token_count": 12.0, "relation_term_count": 2.0},
  "metrics": {"answer_recall_at_10": 1.0, "evidence_recall_at_10": 0.5},
  "latency_seconds": 0.014,
  "retrieved_units": 11,
  "context_chars": 5200,
  "details": {"graph_nodes": 750, "graph_edges": 1800},
  "quality_score": 0.76,
  "normalized_cost": 0.24,
  "reward": 1.68,
  "cheapest_acceptable_route": "optimized_grag"
}
```

The matching `.run_config.json` is equally important. It records the corpus
limit, cost maxima, cost weights, quality threshold, and confirmation that
learned structural models were disabled. Keep it with its JSONL file.

## Pre-retrieval features

The router must choose before an expensive retrieval route has run. It uses
only cheap deterministic question features:

| Feature | Intuition |
|---|---|
| Token and character count | Longer questions often contain more constraints. |
| Bracketed entity count | MetaQA writes known seed entities in brackets; graph routing can benefit. |
| Capitalized phrase count | A fallback signal for named entities. |
| Relation-term count | Words such as `directed`, `author`, or `language` suggest graph relations. |
| Multi-hop cue count | Terms such as `shared`, `same`, `whose`, and `through` can signal a chain. |
| Comparison/temporal count | Terms such as `before`, `after`, `most`, and `when` identify a different reasoning pattern. |
| Bias | Lets a linear policy learn a default route. |

These features are intentionally interpretable. Later experiments can add
embedding vectors, BM25 probe scores, dense probe scores, entity-linking
confidence, and estimated graph size, but only if their extra cost is measured.

## Quality score

The report proposes a composite quality score:

```text
Q = 0.40 × answer + 0.25 × faithfulness
  + 0.20 × evidence + 0.15 × completeness
```

The implementation fills those terms from existing evaluation metrics:

| Quality term | With `--run-generation` | Retrieval-only fallback | Why |
|---|---|---|---|
| Answer | exact match | Answer Recall@10 | Generated correctness is unavailable without generation. |
| Faithfulness | supporting-evidence recall | Evidence Recall@10 | Evidence coverage is the nearest measured grounding signal. |
| Evidence | Evidence Recall@10 | supporting-evidence recall | Measures whether annotated/proxy evidence was found early. |
| Completeness | token F1 | context-answer recall | Measures answer overlap or answer presence in context. |

All components are clipped to `[0, 1]`, so `Q` also stays in `[0, 1]`.

### Correct interpretation

Retrieval-only `Q` is a *route-quality proxy*, not final QA accuracy. It is
useful for affordable initial routing experiments. For a final report, rerun a
fixed held-out set with `--run-generation` and report exact match/F1 separately.

## Cost model

Raw milliseconds, graph-edge counts, and context characters are incomparable.
The builder normalizes each value by the maximum observed in its one outcome
collection run, then combines them:

```text
C = 0.55 × latency
  + 0.15 × graph nodes
  + 0.15 × graph edges
  + 0.10 × evidence units
  + 0.05 × context characters
```

Why these parts exist:

- **Latency** is the main user-visible cost.
- **Graph nodes/edges** capture work ignored by text-only token counts.
- **Evidence units** approximate reranking/context-processing work.
- **Context characters** approximate prompt size and generation burden.

The weights are experiment choices, not universal truths. They are saved in
the run configuration so a different cost policy can be rerun honestly.

Because normalization is run-local, compare route rankings within a run. Do not
compare a `0.30` cost from one machine/run directly with `0.30` from another.
Compare the raw latency and raw graph work as well.

## Quality gate and reward

The default acceptable quality threshold is `tau = 0.65`.

```text
if Q < tau:
    R = -2.0 × (tau - Q) - 0.35 × C
else:
    R = 1 + Q - 0.35 × C
```

The first branch is deliberately negative. Without it, a router could learn
that a cheap but incorrect Hybrid answer is attractive. The second branch says
that two acceptable routes should prefer the lower-cost one.

### Cheapest acceptable route

For each question, the transparent oracle label is selected as follows:

1. Keep routes whose `Q >= tau`.
2. Select the route with the smallest normalized cost.
3. Break ties by higher quality, then by route name for determinism.
4. If no route meets the gate, select the highest-quality route instead.

This label is both a supervised baseline and a useful explanation of what the
bandit should learn. It is not available at live-query time because it requires
running all routes first.

## Data integrity rules

- A train example is valid only if all three route outcomes exist for its
  `sample_id`.
- Use the identical question set, embedding model, corpus, and generation mode
  for all three routes in one collection run.
- For reportable MetaQA comparisons, use `--full-corpus`; limited Hybrid/Graph
  text indexes are only a smoke test (although their native graph remains full).
- Preserve original train/test splits. Do not train the router on test questions.
- Never use a test gold answer in a production routing decision.
