# Complete beginner guide: GRAG-Optimized and its RL router

Welcome. This folder explains the project from the first input question to the
future deployed router. It describes the repository as it exists today and
labels future work clearly, so that planned features are never mistaken for
completed ones.

## Read in this order

1. [01 — What this project is](01-what-this-project-is.md)
2. [02 — Repository map and system architecture](02-repository-and-architecture.md)
3. [03 — Inputs, datasets, models, and configuration](03-inputs-datasets-models-and-config.md)
4. [04 — The three answer routes](04-the-three-rag-routes.md)
5. [05 — What happens internally during collection](05-outcome-collection-internal-workflow.md)
6. [06 — How answers are scored and rewarded](06-metrics-quality-and-reward.md)
7. [07 — Critics, semantic checks, and human calibration](07-critics-semantic-checks-and-calibration.md)
8. [08 — Files produced and what you see](08-artifacts-reports-and-what-you-see.md)
9. [09 — How to run the current project safely](09-running-the-current-project.md)
10. [10 — Current status and the remaining RL roadmap](10-current-status-and-future-roadmap.md)
11. [11 — Glossary and common questions](11-glossary-and-common-questions.md)

## The one-sentence summary

The project compares three ways of answering a question—Hybrid RAG, GraphRAG,
and Optimized GraphRAG—then will train a small decision-maker to choose the
best route for each future question while respecting quality, latency, and
compute limits.

## Important: which implementation to follow

The current, plan-driven implementation is:

```text
src/router/
scripts/router/
configs/router/
artifacts/router/
reports/router/
```

`src/rl_router/`, `scripts/build_route_outcomes.py`, and
`scripts/train_cost_aware_router.py` are earlier experimental work. They are
useful historical code, but they are **not** the workflow described by the
current production plan. For the final-year project, follow the `src/router`
workflow in this guide.

## Current stopping point

The shared collector, deterministic metrics, reward foundations, critic
foundations, and M4 pilot tooling are implemented. The next major execution is
the M4 full-corpus pilot. Outcome-model training and contextual-bandit training
are deliberately not implemented or run yet.

See [10 — Current status and the remaining RL roadmap](10-current-status-and-future-roadmap.md)
before running a large collection.
