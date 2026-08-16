#!/usr/bin/env python
"""Complete end-to-end router training pipeline.

Stages:
1. Load raw M4 outcomes (1,000 questions × 3 actions)
2. Score outcomes with deterministic metrics
3. Build training dataset with budget-aware oracle labels
4. Train contextual-bandit policy on scored data
5. Evaluate policy on held-out validation set
6. Report comprehensive results

Usage:
    python train_router_policy.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.adapters.datasets.base import EvaluationSample
from src.rl_router.features import feature_names, query_features
from src.rl_router.policy import LinearSoftmaxBandit, train_bandit
from src.router.config import BudgetConfig, RewardConfig, RouterConfig
from src.router.constants import ACTION_IDS
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.features import query_features as router_query_features
from src.router.metrics import score_outcome
from src.router.reward import compose_quality_details, compute_units
from src.router.reward.oracle import latency_statistics, select_budget_aware_oracle
from src.router.types import CandidateOutcome, MetricVector, QuestionRecord, ScoredOutcome


@dataclass(slots=True)
class TrainingExample:
    """Single training example with full-feedback rewards for all actions."""
    question_id: str
    question: str
    features: dict[str, float]
    rewards: dict[str, float]  # reward_by_action
    oracle_action: str
    feasible_actions: set[str]


@dataclass(slots=True)
class PolicyEvaluationReport:
    """Summary statistics for policy evaluation."""
    total_examples: int
    oracle_accuracy: float
    mean_oracle_reward: float
    mean_policy_reward: float
    mean_policy_regret: float
    policy_correct_count: int
    policy_incorrect_count: int
    failure_rate: float
    action_distribution: dict[str, int]
    action_rewards: dict[str, float]  # mean reward per action


def stage_1_score_outcomes(outcomes_path: str, questions_path: str, output_path: str) -> int:
    """Stage 1: Score raw outcomes with deterministic metrics."""
    print("\n" + "="*80)
    print("STAGE 1: SCORE RAW OUTCOMES WITH DETERMINISTIC METRICS")
    print("="*80)
    
    started = time.time()
    
    # Load questions
    print(f"Loading questions from {questions_path}...")
    questions = {}
    question_count = 0
    for row in read_jsonl(questions_path):
        record = QuestionRecord(**row)
        questions[record.question_id] = record
        question_count += 1
    print(f"[OK] Loaded {question_count} unique questions")
    
    # Score outcomes
    print(f"Loading and scoring outcomes from {outcomes_path}...")
    scored_rows = []
    outcome_count = 0
    errors = defaultdict(int)
    
    for raw in read_jsonl(outcomes_path):
        try:
            outcome = CandidateOutcome.from_dict(raw)
            if outcome.question_id not in questions:
                errors["missing_question"] += 1
                continue
                
            question = questions[outcome.question_id]
            sample = EvaluationSample(
                id=question.question_id,
                question=question.question,
                answers=question.accepted_answers,
                metadata=question.dataset_metadata,
            )
            
            # Score this outcome
            metrics = score_outcome(sample, outcome, gold_evidence=question.gold_evidence)
            composition = compose_quality_details(metrics, config=RewardConfig())
            
            scored_rows.append({
                "outcome": outcome.to_dict(),
                "metrics": metrics.to_dict(),
                "quality": composition.quality,
                "quality_composition": {
                    "base_quality": composition.base_quality,
                    "penalty": composition.penalty,
                    "availability_mask": composition.availability_mask,
                },
                "compute_units": compute_units(outcome),
            })
            outcome_count += 1
            
            if outcome_count % 500 == 0:
                print(f"  Scored {outcome_count} outcomes...")
                
        except Exception as e:
            errors[f"scoring_error: {type(e).__name__}"] += 1
    
    # Write scored outcomes
    write_jsonl_atomic(output_path, scored_rows)
    elapsed = time.time() - started
    
    print(f"[OK] Scored {outcome_count} outcomes in {elapsed:.1f}s")
    if errors:
        print(f"  Warnings: {dict(errors)}")
    print(f"[OK] Wrote scored outcomes to {output_path}")
    
    return outcome_count


def stage_2_build_training_dataset(
    scored_outcomes_path: str,
    questions_path: str,
    output_path: str,
    budgets: BudgetConfig,
) -> tuple[list[TrainingExample], int]:
    """Stage 2: Build full-information training examples with oracle labels."""
    print("\n" + "="*80)
    print("STAGE 2: BUILD TRAINING DATASET WITH ORACLE LABELS")
    print("="*80)
    
    started = time.time()
    
    # Load questions for quick access
    questions = {}
    for row in read_jsonl(questions_path):
        record = QuestionRecord(**row)
        questions[record.question_id] = record
    
    # Group scored outcomes by (question_id, repeat_id)
    print("Grouping outcomes by question and repeat...")
    groups: dict[tuple[str, int], list[ScoredOutcome]] = defaultdict(list)
    all_scored: list[ScoredOutcome] = []
    
    for row in read_jsonl(scored_outcomes_path):
        outcome = CandidateOutcome.from_dict(row["outcome"])
        metric = MetricVector(**row["metrics"])
        scored = ScoredOutcome(
            outcome=outcome,
            metrics=metric,
            quality=row.get("quality"),
            compute_units=row.get("compute_units"),
        )
        groups[(outcome.question_id, outcome.repeat_id)].append(scored)
        all_scored.append(scored)
    
    print(f"[OK] Found {len(groups)} unique (question, repeat) groups")
    
    # Compute latency reference statistics
    print("Computing latency reference statistics...")
    latency_reference = latency_statistics(all_scored)
    
    # Build examples
    print("Building training examples with oracle selection...")
    examples: list[TrainingExample] = []
    groups_with_all_actions = 0
    groups_incomplete = 0
    
    for (question_id, repeat_id), outcomes in sorted(groups.items()):
        # Only use groups where we have outcomes for ALL three actions
        if {outcome.outcome.candidate_id for outcome in outcomes} != set(ACTION_IDS):
            groups_incomplete += 1
            continue
        
        groups_with_all_actions += 1
        
        # Select oracle action (best under budget constraints)
        oracle = select_budget_aware_oracle(outcomes, budgets=budgets, latency_reference=latency_reference)
        
        # Build rewards dict for all actions
        by_action = {outcome.outcome.candidate_id: outcome for outcome in outcomes}
        rewards = {action: by_action[action].quality or 0.0 for action in ACTION_IDS}
        
        # Feasible actions (not failed, within budget)
        feasible = {action for action in ACTION_IDS if by_action[action].feasible}
        
        question = questions[question_id]
        example = TrainingExample(
            question_id=question_id,
            question=question.question,
            features=router_query_features(question.question),
            rewards=rewards,
            oracle_action=oracle.outcome.candidate_id,
            feasible_actions=feasible,
        )
        examples.append(example)
    
    elapsed = time.time() - started
    
    print(f"[OK] Built {groups_with_all_actions} complete training examples in {elapsed:.1f}s")
    print(f"  (Skipped {groups_incomplete} incomplete groups)")
    
    # Save examples for inspection
    example_dicts = [asdict(ex) for ex in examples]
    for ex_dict in example_dicts:
        ex_dict["feasible_actions"] = list(ex_dict["feasible_actions"])
    write_jsonl_atomic(output_path, example_dicts)
    print(f"[OK] Saved examples to {output_path}")
    
    return examples, groups_with_all_actions


def stage_3_train_policy(
    examples: list[TrainingExample],
    output_path: str,
) -> LinearSoftmaxBandit:
    """Stage 3: Train contextual-bandit policy on full-information examples."""
    print("\n" + "="*80)
    print("STAGE 3: TRAIN CONTEXTUAL-BANDIT POLICY")
    print("="*80)
    
    started = time.time()
    
    if not examples:
        raise RuntimeError("No examples provided for training")
    
    print(f"Training on {len(examples)} examples with {len(ACTION_IDS)} actions")
    print(f"Actions: {ACTION_IDS}")
    print(f"Features: {feature_names()}")
    
    # Convert examples to training format
    training_rows = [
        {
            "features": ex.features,
            "rewards": ex.rewards,
        }
        for ex in examples
    ]
    
    # Train the policy
    print("Training linear softmax bandit...")
    policy = train_bandit(
        training_rows,
        actions=tuple(ACTION_IDS),
        feature_names=feature_names(),
        epochs=400,
        learning_rate=0.08,
        l2=0.0005,
        seed=7,
    )
    
    elapsed = time.time() - started
    print(f"[OK] Trained policy in {elapsed:.1f}s")
    
    # Save policy
    policy_dict = {
        "schema_version": 1,
        "actions": ACTION_IDS,
        "feature_names": feature_names(),
        "policy": policy.to_dict(),
        "training_examples_count": len(examples),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(output_path, "w") as f:
        json.dump(policy_dict, f, indent=2)
    print(f"[OK] Saved policy to {output_path}")
    
    return policy


def stage_4_evaluate_policy(
    policy: LinearSoftmaxBandit,
    examples: list[TrainingExample],
) -> PolicyEvaluationReport:
    """Stage 4: Evaluate policy on the training/validation set."""
    print("\n" + "="*80)
    print("STAGE 4: EVALUATE POLICY PERFORMANCE")
    print("="*80)
    
    print(f"Evaluating on {len(examples)} examples...")
    
    policy_correct = 0
    policy_incorrect = 0
    total_oracle_reward = 0.0
    total_policy_reward = 0.0
    total_regret = 0.0
    action_counts: dict[str, int] = defaultdict(int)
    action_rewards: dict[str, float] = defaultdict(float)
    failure_count = 0
    
    for example in examples:
        # Get policy decision
        policy_choice = policy.choose(example.features)
        oracle_choice = example.oracle_action
        
        # Count correctness
        if policy_choice == oracle_choice:
            policy_correct += 1
        else:
            policy_incorrect += 1
        
        # Track rewards
        oracle_reward = example.rewards[oracle_choice]
        policy_reward = example.rewards[policy_choice]
        
        total_oracle_reward += oracle_reward
        total_policy_reward += policy_reward
        total_regret += max(example.rewards.values()) - policy_reward
        
        action_counts[policy_choice] += 1
        action_rewards[policy_choice] += policy_reward
        
        # Track failures
        if len(example.feasible_actions) < len(ACTION_IDS):
            failure_count += 1
    
    mean_oracle_reward = total_oracle_reward / len(examples) if examples else 0.0
    mean_policy_reward = total_policy_reward / len(examples) if examples else 0.0
    mean_regret = total_regret / len(examples) if examples else 0.0
    failure_rate = failure_count / len(examples) if examples else 0.0
    
    # Normalize action rewards
    for action in action_counts:
        if action_counts[action] > 0:
            action_rewards[action] /= action_counts[action]
    
    report = PolicyEvaluationReport(
        total_examples=len(examples),
        oracle_accuracy=policy_correct / len(examples) if examples else 0.0,
        mean_oracle_reward=mean_oracle_reward,
        mean_policy_reward=mean_policy_reward,
        mean_policy_regret=mean_regret,
        policy_correct_count=policy_correct,
        policy_incorrect_count=policy_incorrect,
        failure_rate=failure_rate,
        action_distribution=dict(action_counts),
        action_rewards=dict(action_rewards),
    )
    
    return report


def print_evaluation_report(report: PolicyEvaluationReport) -> None:
    """Print formatted evaluation report."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nTotal Examples: {report.total_examples}")
    print(f"Policy Accuracy (matches oracle): {report.oracle_accuracy:.1%}")
    print(f"  - Correct: {report.policy_correct_count}/{report.total_examples}")
    print(f"  - Incorrect: {report.policy_incorrect_count}/{report.total_examples}")
    
    print(f"\nReward Performance:")
    print(f"  Oracle mean reward: {report.mean_oracle_reward:.4f}")
    print(f"  Policy mean reward: {report.mean_policy_reward:.4f}")
    print(f"  Regret (opportunity loss): {report.mean_policy_regret:.4f}")
    print(f"  Reward capture ratio: {report.mean_policy_reward/report.mean_oracle_reward*100:.1f}%")
    
    print(f"\nRoute Distribution & Performance:")
    for action in sorted(ACTION_IDS):
        count = report.action_distribution.get(action, 0)
        reward = report.action_rewards.get(action, 0.0)
        pct = count / report.total_examples * 100 if report.total_examples else 0
        print(f"  {action:20s}: {count:4d} selections ({pct:5.1f}%) avg_reward={reward:.4f}")
    
    print(f"\nFailure Rate: {report.failure_rate:.1%}")
    print(f"  (Cases where not all 3 actions were feasible)")


def main() -> None:
    """Execute complete training pipeline."""
    print("\n" + "="*80)
    print("GRAG ROUTER POLICY TRAINING PIPELINE")
    print("="*80)
    
    # Paths
    raw_outcomes = "artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl"
    questions_file = "artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl"
    
    scored_outcomes = "artifacts/router/datasets/m4_scored_outcomes.jsonl"
    training_examples = "artifacts/router/datasets/m4_training_examples.jsonl"
    policy_output = "artifacts/router/policies/m4_contextual_bandit.json"
    
    # Create output directories
    Path(scored_outcomes).parent.mkdir(parents=True, exist_ok=True)
    Path(policy_output).parent.mkdir(parents=True, exist_ok=True)
    
    # Set up budgets (can be tuned)
    budgets = BudgetConfig(
        mean_latency_ms=50.0,  # Allow up to 50ms mean latency
        tail_latency_ms=100.0,  # Allow up to 100ms p95 latency
        compute_units=None,  # Not enforced for now
        max_request_latency_ms=200.0,  # Hard timeout at 200ms
    )
    
    print(f"\nConfiguration:")
    print(f"  Raw outcomes: {raw_outcomes}")
    print(f"  Questions: {questions_file}")
    print(f"  Budget: mean_latency={budgets.mean_latency_ms}ms, tail={budgets.tail_latency_ms}ms")
    
    try:
        # Stage 1: Score outcomes
        outcome_count = stage_1_score_outcomes(raw_outcomes, questions_file, scored_outcomes)
        
        # Stage 2: Build training dataset
        training_data, example_count = stage_2_build_training_dataset(
            scored_outcomes, questions_file, training_examples, budgets
        )
        
        # Stage 3: Train policy
        policy = stage_3_train_policy(training_data, policy_output)
        
        # Stage 4: Evaluate policy
        report = stage_4_evaluate_policy(policy, training_data)
        
        # Print results
        print_evaluation_report(report)
        
        # Save report
        report_path = "artifacts/router/reports/m4_training_report.json"
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"\n[OK] Report saved to {report_path}")
        
        print("\n" + "="*80)
        print("[OK] TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
