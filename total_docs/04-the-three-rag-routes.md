# 04 — The three RAG routes

All three routes receive the same question, use the same configured answer
model, and are scored by the same evaluation process. This makes the comparison
fair.

## 1. Hybrid RAG: `hybrid_rag`

Hybrid RAG combines two text-retrieval methods:

```text
documents
 ├─ NormalRAG: chunk -> embed -> vector search
 └─ BM25RAG:   chunk -> keyword/term search
                     |
                     v
           Reciprocal-rank fusion combines both result lists
                     |
                     v
        top evidence -> context -> answer-generation model
```

Dense/vector search helps when wording is different but meaning is similar.
BM25 helps when exact names or rare keywords are important. Reciprocal-rank
fusion rewards evidence that appears near the top of either list.

## 2. Full GraphRAG: `graph_rag`

GraphRAG keeps both a vector index and a graph:

```text
question
  -> embed question and retrieve vector evidence
  -> extract query entities
  -> use entity seeds and vector evidence to traverse the graph
  -> rank vector and graph evidence
  -> build one grounded context
  -> generate answer
```

For MetaQA, the supplied graph triples are treated as the authoritative native
graph. GraphRAG does not need to invent a second graph from those documents.

GraphRAG may retrieve linked facts that are not close together in text. It is
usually more capable for relation-heavy or multi-hop questions, but can cost
more time and compute.

## 3. Optimized GraphRAG: `optimized_graph_rag`

The optimized route starts from the same graph but creates smaller graph
regions called **capsules**.

```text
full graph
  -> detect hubs
  -> split into regions
  -> choose anchors
  -> build bounded capsules around anchors
  -> build capsule supergraph and indexes

question
  -> identify exact seed entities when possible
  -> select promising capsules
  -> bounded local path search inside/near capsules
  -> compact evidence context
  -> generate answer
```

A capsule is not a learned neural embedding by itself. It is a controlled,
compact graph substructure with source links and a summary. The current router
collection explicitly uses `FeatureStructuralEncoder`, not trained R-GCN or
query-projector artifacts. This keeps the three-route comparison focused on
retrieval strategy rather than a separately trained structural model.

## What is deliberately not a router action

`normal_rag` exists as a vector-only implementation and is useful as a
baseline. It is not one of the three actions in the current router because the
project chose Hybrid RAG as the text-oriented candidate.

## Why three routes are kept separate

The project is **not** trying to merge the three systems into one RAG pipeline.
It needs their independent answers and costs so that a future router can learn
when each route is the best choice.
