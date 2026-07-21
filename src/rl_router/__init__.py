"""Cost-aware route selection utilities for the RAG experiment suite.

The package deliberately contains no R-GCN or query-projector dependency.  It
uses measured outcomes from the Hybrid RAG, GraphRAG, and Optimized GraphRAG
pipelines to train a lightweight contextual-bandit routing policy.
"""

from src.rl_router.features import feature_names, query_features
from src.rl_router.outcomes import (
    CostWeights,
    RewardConfig,
    cheapest_acceptable_route,
    route_cost,
    route_quality,
    route_reward,
)
from src.rl_router.policy import LinearSoftmaxBandit, train_bandit

__all__ = [
    "CostWeights",
    "LinearSoftmaxBandit",
    "RewardConfig",
    "cheapest_acceptable_route",
    "feature_names",
    "query_features",
    "route_cost",
    "route_quality",
    "route_reward",
    "train_bandit",
]
