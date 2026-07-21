# 8. Worked Examples and Troubleshooting

## Example: direct question

Question:

> Who directed *EuroTrip*?

Entity linking finds `eurotrip`. Search produces evidence at all depths, but the
relation-aware 1-edge path scores strongly:

```text
EuroTrip -[DIRECTED_BY]-> David Mandel
```

No hop label was needed.

## Example: repeated relation

Question:

> Who are movie co-directors of Delbert Mann?

The important path repeats a relation:

```text
Delbert Mann -[DIRECTED_BY]-> Jane Eyre
             -[DIRECTED_BY]-> Franco Zeffirelli
```

The relation stemmer aligns `directors` with `DIRECTED_BY`. Earlier diversity
scoring incorrectly preferred two different relation types; current scoring
rewards query-aligned relations even when repeated.

## Example: unknown three-step question

Question:

> The films sharing directors with *Catch Me If You Can* were in which languages?

```text
Catch Me If You Can
  -[DIRECTED_BY]-> director
  -[DIRECTED_BY]-> another film
  -[IN_LANGUAGE]-> language
```

Depth-balanced beam search guarantees that depth-three candidates are produced
before shallower paths can consume the budget. Final context also retains paths
from depths one and two in case the query can be answered earlier.

## Failure checklist

### Trained structural space is disabled

Check:

- unified projector files exist;
- structural/projector fingerprints match;
- runtime capsule match rate is >=95%;
- projection metadata points to the same text embedding dimension.

Never bypass this gate. Mixing incompatible vector spaces produces plausible but
incorrect rankings.

### Indexing tries to embed thousands of capsule summaries

This means trained structural mode was not enabled. Inspect fingerprints and
projector presence. With valid artifacts, summary embedding is skipped.

### Good context recall but low evidence recall

One answer may appear in capsule overview while many valid endpoints are absent
from ranked paths. Inspect per-question Recall@K, increase final paths cautiously,
and verify relation ranking rather than relying only on context hit.

### 3-hop answers are missing

Check `seed_expansion_depth`, `max_path_length`, candidate-path budget, final-path
budget, and relation words. The verified fast MetaQA defaults use a shared depth
ceiling of 3, 11 candidate paths, and 11 final paths. Each exact linked seed gets
an independent fair share of the path budget.

### GPU out of memory

Close stale kernels, reduce R-GCN message batch, reduce embedding/score batches,
and keep one evaluation worker. The relation-grouped R-GCN should never allocate
the old per-edge relation-matrix tensor.

### Evaluation restarted from zero

Confirm the dataset, code, config, model, and artifacts are unchanged. Any change
correctly creates a new fingerprint. Identical interrupted runs checkpoint every
25 questions.

## Known limitations

- Entity linking is rule-based by default and can miss aliases.
- Capsule positives are weakly mined from text/answer overlap.
- Search has a finite maximum depth and node/path budgets.
- MetaQA evidence uses answer endpoints because explicit supporting facts are absent.
- The in-memory stores are suitable for experiments; production-scale persistence
  would require durable graph/vector backends and concurrency controls.
