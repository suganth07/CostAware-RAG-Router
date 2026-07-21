"""Train a three-action contextual-bandit router from measured route outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl_router import RewardConfig, cheapest_acceptable_route, feature_names, route_reward, train_bandit


ACTIONS = ("hybrid_rag", "graph_rag", "optimized_grag")


def main() -> None:
    args = parse_args()
    outcomes = _read_jsonl(Path(args.outcomes))
    reward_config = RewardConfig(acceptable_quality=args.acceptable_quality)
    examples = _complete_examples(outcomes, reward_config=reward_config)
    split = max(1, int(len(examples) * (1.0 - args.validation_fraction)))
    train_examples, validation_examples = examples[:split], examples[split:]
    policy = train_bandit(
        train_examples,
        actions=ACTIONS,
        feature_names=feature_names(),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )
    report = {
        "examples": len(examples),
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "actions": ACTIONS,
        "reward_config": reward_config.to_dict(),
        "train": _evaluate(policy, train_examples),
        "validation": _evaluate(policy, validation_examples),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"policy": policy.to_dict(), "report": report}, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(output), "report": report}, indent=2))


def _complete_examples(outcomes: list[dict], *, reward_config: RewardConfig) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for outcome in outcomes:
        grouped[str(outcome["sample_id"])].append(outcome)
    examples = []
    for sample_id in sorted(grouped):
        rows = grouped[sample_id]
        by_route = {str(row["route"]): row for row in rows}
        if set(by_route) != set(ACTIONS):
            continue
        for row in rows:
            row["reward"] = route_reward(
                float(row["quality_score"]),
                float(row["normalized_cost"]),
                config=reward_config,
            )
        oracle = cheapest_acceptable_route(rows, config=reward_config)
        examples.append(
            {
                "sample_id": sample_id,
                "features": dict(rows[0]["features"]),
                "rewards": {route: float(by_route[route]["reward"]) for route in ACTIONS},
                "oracle_route": str(oracle["route"]),
            }
        )
    if not examples:
        raise RuntimeError("No questions have a complete Hybrid/Graph/Optimized outcome set")
    return examples


def _evaluate(policy, examples: list[dict]) -> dict:
    if not examples:
        return {"route_accuracy": None, "mean_selected_reward": None, "mean_oracle_reward": None}
    selected_rewards = []
    oracle_rewards = []
    correct = 0
    for example in examples:
        selected = policy.choose(example["features"])
        correct += int(selected == example["oracle_route"])
        selected_rewards.append(example["rewards"][selected])
        oracle_rewards.append(max(example["rewards"].values()))
    return {
        "route_accuracy": correct / len(examples),
        "mean_selected_reward": sum(selected_rewards) / len(selected_rewards),
        "mean_oracle_reward": sum(oracle_rewards) / len(oracle_rewards),
        "mean_reward_regret": (sum(oracle_rewards) - sum(selected_rewards)) / len(examples),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cost-aware full-feedback contextual-bandit router.")
    parser.add_argument("--outcomes", default="artifacts/rl_router/outcomes.jsonl")
    parser.add_argument("--output", default="artifacts/rl_router/cost_aware_bandit.json")
    parser.add_argument("--acceptable-quality", type=float, default=0.65)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


if __name__ == "__main__":
    main()
