# Router Architecture

The production router is an offline-trained contextual bandit over exactly
three actions: `hybrid_rag`, `graph_rag`, and `optimized_graph_rag`.

It sits **above** the existing RAG implementations. Candidate adapters convert
their different output classes into one `CandidateOutcome` contract, so policy
and training code never branch on RAG internals.

```text
Question -> runtime-safe features -> router policy -> one candidate adapter
         -> existing RAG pipeline -> answer + normalized telemetry
```

The first release is deliberately one-step. It does not inspect an answer and
escalate. That makes it a contextual bandit rather than PPO/sequential RL.

## Implemented modules

- `src/router/constants.py`: immutable action order and schema versions.
- `src/router/types.py`: `QuestionRecord`, `CandidateOutcome`, metric and
  artifact contracts.
- `src/router/candidates/`: adapters and the three-candidate registry.
- `src/router/data/`: sampling, fingerprints, atomic persistence, resume, and
  complete-matrix validation.
- `src/router/metrics/`: deterministic answer-type-aware metric bridge.
- `src/router/reward/`: quality composition, compute units, feasibility, and
  the budget-aware oracle.

The registry intentionally constructs Optimized GraphRAG with
`FeatureStructuralEncoder`; it does not load R-GCN/projector artifacts.

## Not implemented yet

Semantic cross-encoders, calibrated answer embeddings, structured LLM critics,
multi-task neural outcome heads, and runtime deployment guardrails require
explicit model choices and human calibration. Their fields are already present
as nullable contracts, so later work will not break collected data.
