# Evidence-Preserving Retrieval

The retrieval package exposes one accuracy-oriented path:

```text
query -> entity linking -> seed-covered capsule routing
      -> bounded exact seed neighborhoods
      -> fair per-seed path search
      -> query-ranked source chunks
      -> evidence-first context
```

## Invariants

- `CapsuleRouter` reserves a capsule for every linked seed when capacity allows.
- `BoundedLocalPathSearcher` divides candidate and final-path capacity across
  seeds, so the first/generic seed cannot consume all work.
- A bounded, per-seed source-graph expansion protects facts that cross capsule
  boundaries. Its depth is configured to 1/2/3 for the corresponding MetaQA
  benchmark.
- Relation-aware, depth-balanced beam search prevents shallow paths from
  consuming the candidate budget before required multi-hop paths are produced.
- Final path capacity is balanced across depths 1, 2, and 3 before remaining
  slots are filled by global score. Runtime never needs a gold hop label.
- `OptimizedContextBuilder` ranks chunk text by query overlap, capsule score, and
  path participation. It returns the exact snippets that were rendered, enabling
  supporting-evidence evaluation.

Evidence paths are placed before source snippets, and source snippets before
capsule summaries, so context truncation removes metadata before answer-bearing
evidence.
