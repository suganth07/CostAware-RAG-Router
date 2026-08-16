# Production router documentation

Read in this order:

1. [Architecture](architecture.md)
2. [Data contracts and collection](data-contracts.md)
3. [Reward model](reward-model.md)
4. [Semantic metrics and structured critic](semantic-metrics-and-critic.md)
5. [Training roadmap](training.md)
6. [Evaluation rules](evaluation.md)
7. [M4 pilot collection runbook](m4-pilot.md)
8. [M5 data-preparation foundation](m5-data-preparation.md)
9. [Operations](operations.md)
10. [Plan implementation progress](plan-progress.md)

The full long-range design remains in [../plan.md](../plan.md). The repository
implements the contracts, collection/scoring core, M2/M3 metric and critic
foundations, the completed M4 main-pilot matrix, and M5 split/data-card
foundations. M4 repeated-latency calibration is explicitly deferred, and M5's
full dataset has not been collected. There is no learned outcome model,
contextual-bandit policy, or deployment stage yet.
