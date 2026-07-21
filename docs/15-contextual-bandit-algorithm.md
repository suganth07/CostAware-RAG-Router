# Contextual-Bandit Algorithm Used by the RL Router

## Why this is a contextual bandit first

The full proposal eventually includes sequential actions: run a route, inspect
the answer, accept, escalate, switch, or abstain. That is a multi-step
reinforcement-learning problem.

The implemented first stage has one decision:

```text
question features → choose Hybrid / GraphRAG / Optimized GraphRAG → observe reward
```

This is a **contextual bandit**. It is still reinforcement learning because an
agent chooses an action from a state/context to maximize reward, but there is
no second state transition yet. Starting here prevents an unnecessarily complex
RL system from hiding basic retrieval or metric mistakes.

## Policy representation

The policy is `LinearSoftmaxBandit` in `src/rl_router/policy.py`.

For a feature vector `x` and route `a`, it computes a score:

```text
z_a = w_a · x
```

Then softmax converts the three scores to action probabilities:

```text
pi(a | x) = exp(z_a) / sum over all routes exp(z_route)
```

The selected route in evaluation is the route with the highest probability.

### Why linear softmax?

- It is easy to inspect and explain in a viva.
- It needs only NumPy, which is already a project dependency.
- It is a strong baseline for a small number of interpretable features.
- Its output is a probability distribution, not an opaque hard rule.

It is not claimed to be a deep RL policy. A neural policy is only justified
after this baseline fails on sufficiently large and diverse outcome data.

## Full-feedback reward learning

Usual online bandits only see the reward of the action they selected. Here
offline outcome collection runs all routes, so the reward vector is observed:

```text
r(question) = [reward_hybrid, reward_graph, reward_optimized]
```

The expected policy reward is:

```text
J = average over questions of sum_a pi(a | x) × r_a
```

Training performs gradient ascent on `J`. The route-gradient contribution is:

```text
pi(a | x) × (r_a - expected_reward) × x
```

Intuition:

- a route whose reward is above the policy's current expected reward gets a
  positive update;
- a route below expected reward gets a negative update;
- actions with uncertain/low probability still receive a measured update
  because the outcome table provides their counterfactual reward.

L2 regularization keeps weights from becoming extreme on a small dataset.

## Feature normalization and the bias correction

Question features use different scales: character count may be 100 while an
entity count may be 1. Before training, each non-bias feature is standardized:

```text
x_normalized = (x - training_mean) / training_standard_deviation
```

The training mean and scale are saved inside the model JSON, so future route
predictions use the same transformation.

The `bias` feature is intentionally not centered. Centering a column of ones
would turn it into zeros, preventing the policy from learning a default route.
This is checked by the router smoke test.

## Train/validation split

The trainer uses a deterministic question-order split:

- first 80%: training;
- final 20%: validation by default.

The report includes:

- `route_accuracy`: agreement with cheapest-acceptable-route oracle;
- `mean_selected_reward`: reward of the learned policy's chosen route;
- `mean_oracle_reward`: best reward available from the three measured routes;
- `mean_reward_regret`: the gap between selected and oracle reward.

Route accuracy alone is not the main success metric. If two routes are both
acceptable but one is only slightly more expensive, small route-label
disagreement is less serious than high quality loss or high reward regret.

## Current limitations

- The current policy uses only question text features, not BM25/dense probe
  scores or entity-link confidence.
- The input ordering split is reproducible but should eventually be replaced by
  document-disjoint/entity-disjoint validation where the dataset supports it.
- The bandit chooses one initial route; it does not yet escalate after a weak
  answer.
- A quality proxy based on retrieval metrics is not a replacement for
  human-reviewed faithfulness evaluation.

## Safe extension to sequential RL

Add sequential RL only after the bandit report is stable:

1. Add a post-retrieval critic that uses evidence coverage, retrieval-score
   concentration, answer/context agreement, and latency spent.
2. Add actions `accept`, `abstain`, and `escalate_to_untried_route`.
3. Mask already-tried routes and stop after two or three retrieval attempts.
4. Train/evaluate against a fixed deterministic escalation baseline.
5. Keep gold answer correctness offline only; the live critic must not depend
   on unavailable ground truth.
