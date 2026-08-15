# 01 — What this project is

## The problem

Not every question needs the same kind of retrieval. A simple fact may be
answered quickly by text search. A question that connects several facts may
benefit from graph traversal. Using the most expensive route for every question
wastes time; using the cheapest route for every question can hurt correctness.

This project solves that choice problem.

For each user question, a future router will select exactly one of these three
actions:

| Router action | Plain-language meaning |
|---|---|
| `hybrid_rag` | Search text using both keyword and embedding search, then combine the results. |
| `graph_rag` | Search text and also traverse the full knowledge graph. |
| `optimized_graph_rag` | Use a compact capsule view of the graph, then do bounded local graph search. |

The router itself does **not** answer the question. It chooses which answering
system should answer it.

## What RAG means here

RAG means *retrieval-augmented generation*:

```text
question
  -> retrieve useful evidence
  -> give evidence + question to the answer model
  -> generated answer
```

The answer model is not supposed to answer from memory alone. It receives a
context built from the retrieved evidence.

## Why reinforcement learning is mentioned

The final router is a one-step reinforcement-learning problem:

```text
question -> choose route -> receive quality/cost outcome -> episode ends
```

Because there is only one decision, the planned algorithm is a **contextual
bandit**, not a multi-step game and not PPO. “Contextual” means the decision
depends on clues in the question, such as its length, wording, entities, and
relation cues. “Bandit” means it chooses one action and receives one final
result.

## The safe offline-learning idea

Before a router is allowed to choose a route, the project runs **all three
routes offline** for each sampled training question:

```text
same question
 ├─ hybrid_rag          -> answer, evidence, latency, cost
 ├─ graph_rag           -> answer, evidence, latency, cost
 └─ optimized_graph_rag -> answer, evidence, latency, cost
```

This is called a *full-information outcome matrix*. It is safer than learning
from only the route that happened to be selected, because the project can see
the alternatives directly.

## The eventual end-to-end product

When the project is complete, a user will see only this:

```text
User question -> router chooses one route -> chosen RAG answers -> answer returned
```

Internally, the router will estimate expected quality, latency, compute cost,
failure risk, and uncertainty for all three routes. It will avoid routes that
do not fit the configured operational budget, and it will have a safe fallback
for uncertainty or system failures.

That final online router is planned work. At present, the repository is in the
offline data-collection and validation stage.
