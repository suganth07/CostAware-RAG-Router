# 02 — Repository map and system architecture

## The important folders

| Folder | What it contains |
|---|---|
| `data/` | Local datasets: MetaQA, 2WikiMultiHopQA, and MultiHop-RAG. |
| `src/adapters/` | Common interfaces for datasets, embeddings, language models, graphs, and vector stores. |
| `src/rag/` | Normal dense RAG, BM25 RAG, and Hybrid RAG. |
| `src/g_rag/` | Full GraphRAG implementation. |
| `src/optimized_grag/` | Capsule-based optimized GraphRAG implementation. |
| `src/router/` | Current router contracts, collection, metrics, critics, rewards, and future router boundary. |
| `scripts/router/` | Commands that run the current router workflow. |
| `configs/router/` | Explicit experiment configuration; model choices are not hidden in Python code. |
| `artifacts/router/` | Generated raw outcomes, datasets, critic records, caches, and future checkpoints. |
| `reports/router/` | Small human-readable audit reports. |
| `docs/` | Existing detailed technical plan and component documentation. |
| `total_docs/` | This beginner-first guided explanation. |

## The current architecture

```text
                         OFFLINE (current work)

Dataset question + accepted answer + evidence
                |
                v
       Dataset adapter / sampler
                |
                v
   Shared corpus, graph, embedding cache, answer model
                |
                v
     Candidate registry: fixed three actions
       |              |               |
       v              v               v
  Hybrid RAG      GraphRAG     Optimized GraphRAG
       \              |               /
        \             |              /
         v            v             v
    CandidateOutcome records: answer, evidence, timing, usage, errors
                            |
                            v
          Metrics + reward composer + optional structured critic
                            |
                            v
          Full-information training matrix and audit reports
```

The collector runs all candidates. The scoring stage does not retrieve again;
it reads the outcomes and calculates metrics from their stored answers and
evidence.

## The final architecture (not implemented yet)

```text
                       ONLINE (future work)

Incoming user question
        |
        v
Runtime-safe question features
        |
        v
Trained outcome model and constrained policy
        |
        +--> predicted quality/latency/cost/failure/uncertainty for each route
        |
        v
Budget and safety mask
        |
        v
One selected RAG route -> answer -> telemetry -> user
```

## The fixed action contract

The action order is fixed in `src/router/constants.py`:

```text
hybrid_rag, graph_rag, optimized_graph_rag
```

This matters because a later model will output one value per action. Changing
the names or their order in the middle of an experiment would corrupt labels,
training examples, and comparisons.

## Why adapters exist

Each RAG system returns a different Python object internally. The router must
not need to understand all those differences. `CandidateAdapter` provides the
same simple boundary for every candidate:

```text
retrieve(question)
answer(question)
execute(...) -> CandidateOutcome
```

`execute` measures elapsed time, converts the result to a common retrieval
record, captures instrumentation, and converts an exception into a structured
failure outcome rather than crashing the entire collection.
