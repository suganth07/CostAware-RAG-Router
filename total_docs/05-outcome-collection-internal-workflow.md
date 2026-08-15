# 05 — What happens internally during collection

This is the most important current workflow. Collection creates the evidence
that later learning will use.

## Start and end of one collection run

```text
START: dataset selection + command-line configuration
  -> load corpus, graph, and requested question split
  -> stratify/select questions
  -> create shared embedding adapter and cache
  -> build/index the three candidates
  -> run every candidate for every question and repeat
  -> checkpoint each completed outcome
  -> validate complete matrix
END: raw JSONL outcomes + frozen questions + manifest
```

## Step 1: choose questions

`scripts/router/collect_outcomes.py` loads a dataset bundle. With `--stratified`
it selects a balanced sample using offline strata such as question/answer type.
With `--questions-file`, it replays exactly the frozen IDs in that file. The
second option is used for the repeated latency subset.

The selected questions are written to:

```text
<output>.questions.jsonl
```

That file is important: it freezes exactly which questions belong to the run.

## Step 2: build the shared candidate registry

`build_default_registry()` creates one instance of each route and indexes them
in this order:

1. Hybrid RAG
2. GraphRAG
3. Optimized GraphRAG

The routes have separate in-memory retrieval indexes, but share the same
persistent embedding cache. Therefore, after Hybrid RAG embeds source chunks,
GraphRAG can reuse those vectors for identical text rather than call the
embedding model again.

Current progress logs show:

```text
router registry stage started/completed
normal_rag indexing progress
embedding cache lookup/stored
embedding request progress
graph_rag indexing progress
optimized_grag capsule rebuild started/completed
```

### Why indexing happens before `OutcomeCollector.collect()`

Candidate pipelines must be ready before they can answer any question. For a
full corpus this initialization can be much longer than a short question run.
It is intentionally excluded from per-question answer latency; users care how
long a ready system takes to answer, not how long its one-time index build took.

## Step 3: execute the complete action matrix

`OutcomeCollector.collect()` loops over:

```text
each selected question
  -> each repeat
    -> all three actions in randomized order
```

Randomized action order helps reduce bias caused by a local model or computer
becoming warmer/slower over time. The seed is recorded in the manifest.

For each action, the adapter calls either `retrieve(question)` or
`answer(question)`, depending on `--run-generation`.

## Step 4: normalize the result

Each route has its own internal result type. The adapter converts it into a
single `CandidateOutcome` record containing:

| Field group | Examples |
|---|---|
| Identity | question ID, candidate ID, repeat ID, run ID |
| Provenance | pipeline, corpus, model, and hardware fingerprints |
| Answer/evidence | generated answer and normalized retrieval record |
| Timing | retrieval, generation, and end-to-end latency when available |
| Usage | instrumentation and token information |
| Failure information | timeout/error type and a hash, rather than a raw exception message |

An exception becomes a failure row for that candidate. It does not silently
remove the row or destroy the complete matrix.

## Step 5: resumability

After every completed candidate outcome, the collector atomically writes the
JSONL file. On a rerun, it reads the existing rows and skips completed keys:

```text
(question_id, candidate_id, repeat_id, run_id)
```

It rejects an attempt to mix records from a different run fingerprint into the
same file. Use the same settings/path to resume; use a new output name if the
dataset, models, corpus, or major configuration changes.

## Step 6: validate the matrix

A valid run contains exactly one row for each required combination. For example:

```text
40 questions × 2 repeats × 3 actions = 240 outcomes
```

If one action is absent for a question/repeat, that group must not become a
training example. This prevents a policy from learning from an unfair,
incomplete comparison.

## M4 collection sizes

The prepared M4 plan has two different collections:

| Collection | Purpose | Expected outcome rows |
|---|---|---:|
| Main pilot: 1,000 questions × 1 repeat | Quality, candidate wins, and normal operational distribution | 3,000 |
| Latency subset: 100 questions × 3 repeats | Measure repeated latency variation on the exact same questions | 900 |

The repeated subset is analyzed separately. It should not be mixed into the
main matrix as if it were ordinary independent training data.
