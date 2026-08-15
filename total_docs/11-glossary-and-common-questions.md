# 11 — Glossary and common questions

## Glossary

| Term | Beginner explanation |
|---|---|
| RAG | Retrieve evidence first, then let a language model answer using it. |
| Embedding | A list of numbers representing text meaning, used for similarity search. |
| Vector store | Stores embeddings and finds vectors close to a query vector. |
| BM25 | A keyword-based ranking method; good for exact names and terms. |
| Graph | Entities as nodes and relations as edges. |
| Multi-hop question | A question requiring more than one relation/fact to reach an answer. |
| Capsule | A bounded, compact subgraph used by Optimized GraphRAG. |
| Candidate/action | One of the three RAG systems the router can choose. |
| Outcome | The measured result of one candidate on one question/repeat. |
| Full-information matrix | A dataset that records all three actions for the same question. |
| Metric | One measurable property, such as exact match or latency. |
| Reward/quality | Combined score used to compare actions. |
| Oracle | A rule that chooses the best observed action after seeing all outcomes. It is not a learned model. |
| Contextual bandit | One decision based on context, followed by one final reward. |
| Feasible action | An action that satisfies configured latency/compute budgets. |
| Fingerprint | A stable identity hash proving which corpus/models/configuration produced an artifact. |
| Manifest | Metadata file that describes an artifact and its provenance. |
| Calibration | Checking a model signal against human labels before relying on it. |
| Nullable metric | A metric allowed to be unknown (`null`) rather than falsely set to zero. |

## Common questions

### Is R-GCN used in the current router experiment?

No. The current router collector uses `FeatureStructuralEncoder` for Optimized
GraphRAG. Existing R-GCN/projector work remains elsewhere in the repository but
is intentionally outside the current three-route RL comparison.

### Is the RL model trained now?

No. The repository currently prepares valid full-information data and reward
signals. M6/M7 training is blocked until the M4 pilot and human calibration are
complete.

### Why run all three routes if the router should choose one?

Offline, running all three tells us what would have happened under every
choice. Online, the future router will run only its selected route to save cost.

### Why is a small 40-question success not enough?

It proves code wiring and data contracts, not statistical reliability. A model
can look good on a small templated sample and fail on new question types or
hardware conditions.

### Why does the first full-corpus run appear stuck?

Indexing happens before outcomes. The first run must create missing document
embeddings and build in-memory retrieval/graph indexes. Progress logs and
`inspect_embedding_cache.py` show exactly which part is working.

### Can I delete the embedding cache to fix a problem?

Usually no. Deleting it throws away expensive reusable vectors. First inspect
the cache report and verify the model/namespace/path. Delete only when you
intentionally changed embedding semantics or the cache is proven corrupt.

### What does the final user see?

Ideally only a normal answer experience: submit a question and receive an
answer. The router selection, budgets, telemetry, and fallbacks are internal.

### What should I explain in a viva/project review?

Explain the sequence: three independent RAG candidates → full-information
offline outcomes → quality/cost evaluation → calibrated labels → constrained
contextual-bandit router → safe runtime selection. Emphasize that the current
work has validated the data/measurement foundation and has not claimed RL
training results prematurely.
