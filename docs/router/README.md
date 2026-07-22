# Production Router Documentation

Read in this order:

1. [Architecture](architecture.md)
2. [Data contracts and collection](data-contracts.md)
3. [Reward model](reward-model.md)
4. [Semantic metrics and structured critic](semantic-metrics-and-critic.md)
5. [Training roadmap](training.md)
6. [Evaluation rules](evaluation.md)
7. [Operations](operations.md)
8. [Plan implementation progress](plan-progress.md)

The full long-range design remains in [../plan.md](../plan.md). The code in
`src/router` currently implements the contracts, collection/scoring core, and
the first candidate-blind structured-critic collection step (Milestones 0–3).
It does not yet implement the learned outcome model, constrained contextual
bandit policy, or deployment stages.
