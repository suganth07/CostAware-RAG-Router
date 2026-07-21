"""A dependency-light full-feedback contextual-bandit policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(slots=True)
class LinearSoftmaxBandit:
    """Linear softmax policy trained with observed rewards for every action."""

    actions: tuple[str, ...]
    feature_names: tuple[str, ...]
    means: list[float]
    scales: list[float]
    weights: list[list[float]]

    def probabilities(self, features: dict[str, float]) -> dict[str, float]:
        vector = _standardize(features, self.feature_names, self.means, self.scales)
        logits = np.asarray(self.weights, dtype=np.float64) @ vector
        logits -= logits.max(initial=0.0)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return {action: float(probability) for action, probability in zip(self.actions, probabilities)}

    def choose(self, features: dict[str, float]) -> str:
        probabilities = self.probabilities(features)
        return max(self.actions, key=lambda action: (probabilities[action], action))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LinearSoftmaxBandit":
        return cls(
            actions=tuple(str(action) for action in value["actions"]),
            feature_names=tuple(str(name) for name in value["feature_names"]),
            means=[float(item) for item in value["means"]],
            scales=[float(item) for item in value["scales"]],
            weights=[[float(item) for item in row] for row in value["weights"]],
        )


def train_bandit(
    examples: Iterable[dict[str, Any]],
    *,
    actions: tuple[str, ...],
    feature_names: tuple[str, ...],
    epochs: int = 400,
    learning_rate: float = 0.08,
    l2: float = 0.0005,
    seed: int = 7,
) -> LinearSoftmaxBandit:
    """Fit a policy by maximizing expected counterfactual reward.

    Every example contains the reward measured for every available route.  This
    is full-feedback offline contextual-bandit training, not an invalid attempt
    to infer unobserved rewards from the selected action alone.
    """
    rows = list(examples)
    if not rows:
        raise ValueError("No complete route-outcome examples were provided")
    matrix = np.asarray(
        [[float(row["features"].get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    rewards = np.asarray(
        [[float(row["rewards"][action]) for action in actions] for row in rows],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-9] = 1.0
    # The explicit intercept must survive standardization.  Centering a column
    # of ones turns it into zeros and leaves the policy unable to learn a
    # route preference when questions share similar features.
    for index, name in enumerate(feature_names):
        if name == "bias":
            means[index] = 0.0
            scales[index] = 1.0
    values = (matrix - means) / scales
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=(len(actions), len(feature_names)))

    for _ in range(max(1, int(epochs))):
        logits = values @ weights.T
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected_reward = (probabilities * rewards).sum(axis=1, keepdims=True)
        # Gradient ascent on E_pi[r], with L2 regularization for stability.
        advantage = probabilities * (rewards - expected_reward)
        gradient = advantage.T @ values / len(rows) - l2 * weights
        weights += learning_rate * gradient

    return LinearSoftmaxBandit(
        actions=actions,
        feature_names=feature_names,
        means=means.tolist(),
        scales=scales.tolist(),
        weights=weights.tolist(),
    )


def _standardize(
    features: dict[str, float],
    feature_names: tuple[str, ...],
    means: list[float],
    scales: list[float],
) -> np.ndarray:
    values = np.asarray([float(features.get(name, 0.0)) for name in feature_names], dtype=np.float64)
    return (values - np.asarray(means)) / np.asarray(scales)
