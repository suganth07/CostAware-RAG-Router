from src.rl_router import (
    RewardConfig,
    cheapest_acceptable_route,
    feature_names,
    query_features,
    train_bandit,
)


def test_cheapest_acceptable_route_prioritizes_quality_gate_then_cost():
    chosen = cheapest_acceptable_route(
        [
            {"route": "hybrid_rag", "quality_score": 0.90, "normalized_cost": 0.80},
            {"route": "graph_rag", "quality_score": 0.20, "normalized_cost": 0.05},
            {"route": "optimized_grag", "quality_score": 0.72, "normalized_cost": 0.30},
        ],
        config=RewardConfig(acceptable_quality=0.65),
    )
    assert chosen["route"] == "optimized_grag"


def test_bandit_learns_deterministic_reward_preference():
    features = query_features("Which films share a director with [Film A]?")
    examples = [
        {
            "features": features,
            "rewards": {
                "hybrid_rag": -0.5,
                "graph_rag": 0.2,
                "optimized_grag": 1.4,
            },
        }
        for _ in range(20)
    ]
    policy = train_bandit(
        examples,
        actions=("hybrid_rag", "graph_rag", "optimized_grag"),
        feature_names=feature_names(),
        epochs=150,
        learning_rate=0.1,
    )
    assert policy.choose(features) == "optimized_grag"
