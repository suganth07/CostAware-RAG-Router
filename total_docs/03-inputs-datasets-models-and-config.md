# 03 — Inputs, datasets, models, and configuration

## What enters the system

During offline collection, the input is not just a sentence. A dataset adapter
creates a question record containing:

```text
question_id          stable identifier
question             text to answer
accepted_answers     one or more accepted gold answers
gold_evidence        known supporting evidence when available
answer_type          for example span, boolean, numeric/date
sampling_strata      information used only for balanced offline sampling
dataset metadata     source dataset, split, question type, hop metadata, etc.
```

Gold answer/evidence and benchmark hop labels are permitted during offline
evaluation. They are **not** allowed as features when the future online router
makes a real user-facing decision, because a real user question has no gold
answer.

## Local datasets

| Dataset | Location | Why it is useful |
|---|---|---|
| MetaQA | `data/datasets/metaqa/` | Movie knowledge-base QA with 1-, 2-, and 3-hop questions and a native graph. Main current pilot dataset. |
| 2WikiMultiHopQA | `data/datasets/2WikiMultiHopQA/` | Text-focused multi-hop QA with supporting facts. Planned cross-domain evaluation source. |
| MultiHop-RAG | `data/datasets/MultiHop-RAG/` | Smaller real-world-style QA/corpus collection. Planned later evaluation source. |

### Example: what MetaQA loading does

The MetaQA adapter reads triples such as:

```text
subject | relation | object
```

It converts each triple into both:

1. a text document, for example `Movie A directed by Person B.`; and
2. typed graph nodes and an edge connecting the two entities.

It separately loads questions and accepted answers from the selected
train/validation/test file. This is why MetaQA can support both ordinary text
retrieval and graph retrieval from the same source data.

## Models and services used now

The current collection command accepts configurable local model identifiers:

| Role | Current command option | Purpose |
|---|---|---|
| Embedding model | `--embedding-model` | Turns text into vectors for dense retrieval. |
| Answer-generation model | `--llm-model` | Produces an answer using retrieved context. |
| Critic model | `--judge-model` | Optional later structured evaluation; never silently selected. |

The default local transport is Ollama. The embedding cache is content-addressed:
the same exact text under the same semantic namespace reuses an existing vector.

## Why full-corpus indexing takes time

For the full MetaQA corpus, there are 134,741 documents. The main dense index
must have an embedding for each unique source text. A cache inspection on 24
July 2026 found only 1,000 matching document chunks cached, so the first
full-corpus build still needs many embeddings.

This is expected first-run work, not an error in outcome collection. The
collector now indexes NormalRAG in checkpoint batches of 512 chunks; completed
batches stay in `artifacts/cache/embeddings.sqlite3` and are reused if a run is
interrupted and resumed.

## Configuration files

| File | Meaning |
|---|---|
| `configs/router/pilot.yaml` | Small pilot action list, repeat count, and provisional reward block weights. |
| `configs/router/m4_pilot.yaml` | Current M4 plan: 1,000 main questions, 100 latency-subset questions, three repeats, full corpus, and seeds. |
| `configs/router/semantic.yaml` | Semantic-equivalence model slot. It starts disabled and model-neutral. |
| `configs/router/critics.yaml` | Primary/secondary critic slots. They also start disabled and model-neutral. |

Do not hardcode a semantic judge or critic model into the source code. The
project intentionally waits for a validated model choice and human calibration
before enabling either signal in reward decisions.
