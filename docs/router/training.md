# Router Training Roadmap

The implemented foundation stops after creating a complete, scored,
budget-aware training dataset. This is intentional: a policy cannot be trained
reliably until quality, latency, compute, failure, and missing-metric behavior
are auditable.

## Current sequence

1. Collect all three outcomes for each question and repeat.
2. Score deterministic metrics while retaining unavailable semantic/critic
   signals as `null`.
3. Build a complete full-information matrix with budget feasibility and oracle
   action labels.
4. Audit action distribution, failures, latency variance, and split leakage.
5. Train outcome predictors and an oracle-imitation policy.
6. Train the constrained full-information contextual bandit.

## Why this order matters

PPO is not the correct first implementation. The collector observes all three
actions offline, so policy training should directly optimize expected outcome
over the complete action vector. Before model fitting, manually inspect a
100-question smoke run and a 1,000–2,000-question pilot.

## Next training milestone

The next code increment will add a compact multi-task model with heads for
quality, latency quantiles, compute, and failure probability. It must be
validated against simple per-action means before a constrained policy is
promoted. Do not use test data for reward tuning, critic prompt changes, or
budget selection.
