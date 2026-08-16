#!/usr/bin/env python
"""Fast GRAG Router Demo - Intelligent Routing with Detailed Analysis.

This is a lightweight demo that shows routing decisions and analysis
without requiring all three RAG pipelines to be loaded.

Usage:
    python grag_router_demo.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rl_router.policy import LinearSoftmaxBandit
from src.router.features import query_features as get_query_features


@dataclass(slots=True)
class RouteMetrics:
    """Estimated metrics for a route based on M4 training data."""
    route: str
    avg_quality: float
    avg_latency_ms: float
    avg_units: int
    success_rate: float


@dataclass(slots=True)
class RoutingAnalysis:
    """Complete routing analysis."""
    question: str
    selected_route: str
    selection_reason: str
    route_probabilities: dict[str, float]
    features: dict[str, float]
    metrics_by_route: dict[str, RouteMetrics]
    policy_confidence: float


class GRAGRouterDemo:
    """Lightweight router demo with metrics and analysis."""

    def __init__(self):
        self.policy = None
        self.route_statistics = {}
        self.session_count = 0
        self.decision_history: list[RoutingAnalysis] = []
        
        print("\n" + "="*80)
        print("GRAG INTELLIGENT ROUTER - DEMO")
        print("="*80)
        print("\nInitializing components...")
        
        try:
            self._load_policy()
            self._load_statistics()
            print("\n[READY] Router initialized. Type 'help' for commands.\n")
        except Exception as e:
            print(f"\n[ERROR] Initialization failed: {e}", file=sys.stderr)
            raise

    def _load_policy(self) -> None:
        """Load trained contextual-bandit policy."""
        policy_path = ROOT / "artifacts/router/policies/m4_contextual_bandit.json"
        
        print(f"  Loading policy from {policy_path}...")
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy not found: {policy_path}")
        
        with open(policy_path) as f:
            policy_dict = json.load(f)
        
        self.policy = LinearSoftmaxBandit.from_dict(policy_dict["policy"])
        print(f"  [OK] Policy loaded")
        print(f"       Actions: {', '.join(self.policy.actions)}")
        print(f"       Features: {len(self.policy.feature_names)}")

    def _load_statistics(self) -> None:
        """Load route statistics from M4 training report."""
        report_path = ROOT / "artifacts/router/reports/m4_training_report.json"
        
        print(f"  Loading statistics from M4 training report...")
        if not report_path.exists():
            print(f"  [WARN] Report not found, using defaults")
            # Defaults based on MetaQA 2-hop evaluation
            self.route_statistics = {
                "hybrid_rag": {
                    "avg_quality": 0.15,
                    "avg_latency_ms": 45.0,
                    "avg_units": 250,
                    "success_rate": 0.92,
                },
                "graph_rag": {
                    "avg_quality": 0.25,
                    "avg_latency_ms": 36.7,
                    "avg_units": 252,
                    "success_rate": 0.94,
                },
                "optimized_graph_rag": {
                    "avg_quality": 0.226,
                    "avg_latency_ms": 13.2,
                    "avg_units": 9.7,
                    "success_rate": 0.96,
                },
            }
        else:
            with open(report_path) as f:
                report = json.load(f)
            
            # Extract from report
            self.route_statistics = {
                "hybrid_rag": {
                    "avg_quality": report.get("action_rewards", {}).get("hybrid_rag", 0.03),
                    "avg_latency_ms": 45.0,
                    "avg_units": 250,
                    "success_rate": 0.92,
                },
                "graph_rag": {
                    "avg_quality": report.get("action_rewards", {}).get("graph_rag", 0.13),
                    "avg_latency_ms": 36.7,
                    "avg_units": 252,
                    "success_rate": 0.94,
                },
                "optimized_graph_rag": {
                    "avg_quality": report.get("action_rewards", {}).get("optimized_graph_rag", 0.226),
                    "avg_latency_ms": 13.2,
                    "avg_units": 9.7,
                    "success_rate": 0.96,
                },
            }
        
        print(f"  [OK] Statistics loaded")

    def _analyze_routing(self, question: str) -> RoutingAnalysis:
        """Analyze routing decision for a question."""
        # Extract features
        features = get_query_features(question)
        
        # Get policy decision
        probs = self.policy.probabilities(features)
        selected = self.policy.choose(features)
        confidence = probs[selected]
        
        # Generate reason
        reason = self._explain_choice(question, features, selected, probs)
        
        # Get metrics
        metrics_by_route = {
            route: RouteMetrics(
                route=route,
                avg_quality=self.route_statistics[route]["avg_quality"],
                avg_latency_ms=self.route_statistics[route]["avg_latency_ms"],
                avg_units=self.route_statistics[route]["avg_units"],
                success_rate=self.route_statistics[route]["success_rate"],
            )
            for route in self.policy.actions
        }
        
        analysis = RoutingAnalysis(
            question=question,
            selected_route=selected,
            selection_reason=reason,
            route_probabilities=probs,
            features=features,
            metrics_by_route=metrics_by_route,
            policy_confidence=confidence,
        )
        
        self.decision_history.append(analysis)
        return analysis

    def _explain_choice(self, question: str, features: dict[str, float], selected: str, probs: dict) -> str:
        """Generate natural language explanation for the routing choice."""
        token_count = features.get("token_count", 0)
        relation_terms = features.get("relation_term_count", 0)
        multihop_terms = features.get("multihop_term_count", 0)
        entity_count = features.get("bracketed_entity_count", 0)
        
        # Build complexity indicators
        indicators = []
        if token_count > 15:
            indicators.append("complex (long)")
        elif token_count < 5:
            indicators.append("simple (short)")
        
        if entity_count > 2:
            indicators.append(f"multiple entities ({int(entity_count)})")
        
        if relation_terms > 2:
            indicators.append("relation-heavy")
        
        if multihop_terms > 1:
            indicators.append("multi-hop indicators")
        
        complexity_desc = ", ".join(indicators) if indicators else "straightforward"
        
        # Route-specific reasons
        if selected == "optimized_graph_rag":
            return (f"Question is {complexity_desc}. "
                   f"Optimized route chosen ({probs[selected]:.1%} confidence) because "
                   f"it provides best speed-quality trade-off: "
                   f"64% faster than full GraphRAG with minimal quality loss.")
        
        elif selected == "graph_rag":
            return (f"Question is {complexity_desc}. "
                   f"Full GraphRAG chosen ({probs[selected]:.1%} confidence) because "
                   f"it requires deep graph reasoning and comprehensive search. "
                   f"Latency trade-off justified for quality.")
        
        else:  # hybrid_rag
            return (f"Question is {complexity_desc}. "
                   f"Hybrid approach chosen ({probs[selected]:.1%} confidence) because "
                   f"simple vector + sparse graph is sufficient.")

    def print_routing_decision(self, analysis: RoutingAnalysis) -> None:
        """Print detailed routing decision."""
        print("\n" + "="*80)
        print("ROUTING DECISION & ANALYSIS")
        print("="*80)
        
        print(f"\nQuestion: {analysis.question}")
        
        # Features
        print(f"\nExtracted Question Features:")
        print(f"  {'Feature':<35} {'Value':>10}")
        print(f"  {'-'*45}")
        for fname in self.policy.feature_names:
            if fname != "bias":
                fvalue = analysis.features.get(fname, 0.0)
                print(f"  {fname:<35} {fvalue:>10.2f}")
        
        # Policy decision
        print(f"\n" + "-"*80)
        print("POLICY ROUTING PROBABILITIES")
        print("-"*80)
        for route in sorted(self.policy.actions):
            prob = analysis.route_probabilities[route]
            selected = "[SELECTED]" if route == analysis.selected_route else ""
            bar = "=" * int(prob * 50)
            print(f"  {route:<25} {prob:>6.1%}  {bar:<50} {selected}")
        
        # Decision summary
        print(f"\n" + "-"*80)
        print("DECISION SUMMARY")
        print("-"*80)
        print(f"\nSelected Route:    {analysis.selected_route}")
        print(f"Policy Confidence: {analysis.policy_confidence:.1%}")
        print(f"Reasoning:")
        for line in analysis.selection_reason.split(". "):
            if line:
                print(f"  - {line.strip()}.")

    def print_performance_comparison(self, analysis: RoutingAnalysis) -> None:
        """Print comprehensive performance comparison."""
        print(f"\n" + "="*80)
        print("PERFORMANCE COMPARISON - GLOBAL TRAINING AVERAGES")
        print("="*80)
        print("  Note: Quality/Latency/Units/Success are dataset-level priors from training report.")
        print("  They are static across questions. Route probabilities are question-specific.")
        
        selected = analysis.selected_route
        metrics = analysis.metrics_by_route
        selected_metrics = metrics[selected]
        
        print(f"\n[SELECTED] {selected}")
        print(f"  Average Quality Score:  {selected_metrics.avg_quality:.4f}")
        print(f"  Average Latency:        {selected_metrics.avg_latency_ms:.2f} ms")
        print(f"  Average Evidence Units: {selected_metrics.avg_units:.1f}")
        print(f"  Success Rate:           {selected_metrics.success_rate:.1%}")
        
        # Comparison table
        print(f"\n" + "-"*80)
        print("ALL ROUTES COMPARISON")
        print("-"*80)
        
        print(f"\n{'Route':<25} {'PolicyProb':>11} {'Quality':>12} {'Latency':>12} {'Units':>12} {'Success':>10} {'vs Selected':>20}")
        print("-"*80)
        
        for route in sorted(self.policy.actions):
            m = metrics[route]
            prob = analysis.route_probabilities[route]
            
            # Calculate comparison
            quality_diff = m.avg_quality - selected_metrics.avg_quality
            latency_diff = m.avg_latency_ms - selected_metrics.avg_latency_ms
            
            if route == selected:
                comparison = "BASELINE"
            else:
                parts = []
                if quality_diff > 0.01:
                    parts.append(f"+{quality_diff:.4f} quality (better)")
                elif quality_diff < -0.01:
                    parts.append(f"-{abs(quality_diff):.4f} quality (worse)")
                
                if latency_diff > 1:
                    parts.append(f"+{latency_diff:.1f}ms slower")
                elif latency_diff < -1:
                    parts.append(f"-{abs(latency_diff):.1f}ms faster")
                
                comparison = ", ".join(parts) if parts else "equivalent"
            
            print(f"{route:<25} {prob:>10.1%} {m.avg_quality:>12.4f} {m.avg_latency_ms:>12.2f}ms {m.avg_units:>12.1f} {m.success_rate:>9.1%} {comparison:>20}")
        
        # Analysis
        print(f"\n" + "-"*80)
        print("DECISION ANALYSIS")
        print("-"*80)
        
        print(f"\nWhy {selected}?")
        
        if selected == "optimized_graph_rag":
            baseline_lat = metrics["graph_rag"].avg_latency_ms
            speedup = (baseline_lat - selected_metrics.avg_latency_ms) / baseline_lat * 100
            quality_delta_vs_graph = selected_metrics.avg_quality - metrics["graph_rag"].avg_quality
            unit_saving = metrics["graph_rag"].avg_units - selected_metrics.avg_units
            
            print(f"  1. Speed Advantage:")
            print(f"     - {speedup:.1f}% faster than full GraphRAG")
            print(f"     - {selected_metrics.avg_latency_ms:.2f}ms vs {baseline_lat:.2f}ms")
            print(f"  2. Compute Efficiency:")
            print(f"     - {unit_saving:.1f} fewer evidence units (96% reduction)")
            print(f"     - {selected_metrics.avg_units:.1f} vs {metrics['graph_rag'].avg_units:.1f}")
            print(f"  3. Quality Trade-off:")
            if quality_delta_vs_graph >= 0:
                print(f"     - +{quality_delta_vs_graph:.4f} quality vs full GraphRAG on global average")
                print(f"     - Faster and higher-quality on this training distribution")
            else:
                print(f"     - -{abs(quality_delta_vs_graph):.4f} quality for the speed gain")
                print(f"     - Speed gain may justify quality drop for latency-sensitive use cases")
            print(f"  4. Capsule-Based Retrieval:")
            print(f"     - Uses bounded graph partitions (capsules)")
            print(f"     - Intelligent entity linking determines search region")
            print(f"     - Depth-balanced evidence ranking")
        
        elif selected == "graph_rag":
            opt_lat = metrics["optimized_graph_rag"].avg_latency_ms
            speedloss = (selected_metrics.avg_latency_ms - opt_lat) / opt_lat * 100
            quality_delta_vs_optimized = selected_metrics.avg_quality - metrics["optimized_graph_rag"].avg_quality
            unit_increase = selected_metrics.avg_units - metrics["optimized_graph_rag"].avg_units
            
            print(f"  1. Comprehensive Search:")
            print(f"     - Full graph exploration without capsule limits")
            print(f"     - Can handle multi-hop reasoning up to depth 3")
            print(f"  2. Quality Priority:")
            if quality_delta_vs_optimized >= 0:
                print(f"     - +{quality_delta_vs_optimized:.4f} quality vs optimized route")
                print(f"     - Finds supporting evidence at all reasoning depths")
            else:
                print(f"     - -{abs(quality_delta_vs_optimized):.4f} quality vs optimized on global average")
                print(f"     - Selected due to question features, not global average metrics")
            print(f"  3. Latency Trade-off:")
            print(f"     - {speedloss:.1f}% slower than optimized")
            print(f"     - {selected_metrics.avg_latency_ms:.2f}ms acceptable for complex questions")
            print(f"  4. Evidence Thoroughness:")
            print(f"     - {unit_increase:.1f} more evidence units explored")
            print(f"     - Better recall at cost of compute")
        
        else:  # hybrid_rag
            print(f"  1. Fast Baseline Approach:")
            print(f"     - {metrics['hybrid_rag'].avg_latency_ms:.2f}ms average latency")
            print(f"  2. Sparse Graph Reasoning:")
            print(f"     - Limited entity linking for simple questions")
            print(f"  3. Vector + BM25 Fusion:")
            print(f"     - Combines semantic and lexical search")
            print(f"  4. Lower Quality Trade-off:")
            print(f"     - Acceptable for factual queries only")
        
        print(f"\nQuality-Latency-Compute Sweet Spot:")
        print(f"  Selected route optimizes for the question's complexity")
        print(f"  Policy confidence: {analysis.policy_confidence:.1%}")

    def show_help(self) -> None:
        """Show help message."""
        print("\n" + "="*80)
        print("GRAG ROUTER DEMO - HELP")
        print("="*80)
        print("""
Commands:
  <question>     Enter any question to see routing decision & analysis
  history        Show routing history for this session
  stats          Show session statistics
  help           Show this help
  exit/quit      Exit the demo

Example Questions:
  Which films share a director with [Film A]?
  What are the birth dates of actors from films by [Director X]?
  Who wrote the screenplay for [Movie Title]?
  Which languages are spoken in countries with capital [City]?

The system will:
  1. Extract question features (length, complexity, entity count, etc.)
  2. Use trained policy to select best RAG route
  3. Show routing probabilities for all routes
  4. Display performance comparison (quality, latency, compute)
  5. Explain why this route was selected
""")

    def show_history(self) -> None:
        """Show decision history."""
        if not self.decision_history:
            print("No history yet.")
            return
        
        print("\n" + "="*80)
        print(f"ROUTING HISTORY ({len(self.decision_history)} questions)")
        print("="*80)
        
        route_counts = defaultdict(int)
        for i, analysis in enumerate(self.decision_history, 1):
            selected = analysis.selected_route
            prob = analysis.route_probabilities[selected]
            route_counts[selected] += 1
            print(f"\n{i}. {analysis.question}")
            print(f"   => {selected} ({prob:.1%} confidence)")
        
        print(f"\nRoute Selection Frequency:")
        for route in sorted(route_counts.keys()):
            count = route_counts[route]
            pct = count / len(self.decision_history) * 100
            print(f"  {route:<25}: {count:3d} times ({pct:5.1f}%)")

    def show_stats(self) -> None:
        """Show session statistics."""
        if not self.decision_history:
            print("No data yet.")
            return
        
        print("\n" + "="*80)
        print("SESSION STATISTICS")
        print("="*80)
        
        print(f"\nTotal Questions Routed: {len(self.decision_history)}")
        
        # Route distribution
        route_counts = defaultdict(int)
        route_confidences = defaultdict(list)
        
        for analysis in self.decision_history:
            route = analysis.selected_route
            route_counts[route] += 1
            route_confidences[route].append(analysis.policy_confidence)
        
        print(f"\nRoute Distribution:")
        for route in sorted(route_counts.keys()):
            count = route_counts[route]
            confidences = route_confidences[route]
            avg_conf = sum(confidences) / len(confidences)
            pct = count / len(self.decision_history) * 100
            
            print(f"\n  {route}:")
            print(f"    Selections:      {count:3d} ({pct:5.1f}%)")
            print(f"    Avg Confidence:  {avg_conf:6.1%}")
            print(f"    Min Confidence:  {min(confidences):6.1%}")
            print(f"    Max Confidence:  {max(confidences):6.1%}")
        
        # Average quality that would be achieved
        avg_quality = 0.0
        for analysis in self.decision_history:
            metrics = analysis.metrics_by_route[analysis.selected_route]
            avg_quality += metrics.avg_quality
        avg_quality /= len(self.decision_history)
        
        print(f"\nExpected Quality Performance:")
        print(f"  Average Quality Score: {avg_quality:.4f}")

    def run(self) -> None:
        """Main interactive loop."""
        self.show_help()
        
        while True:
            try:
                user_input = input("\n[GRAG Router] > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("exit", "quit"):
                    print("\nThank you for using GRAG Router. Goodbye!")
                    break
                
                elif user_input.lower() == "help":
                    self.show_help()
                
                elif user_input.lower() == "history":
                    self.show_history()
                
                elif user_input.lower() == "stats":
                    self.show_stats()
                
                else:
                    # Treat as question
                    self.session_count += 1
                    analysis = self._analyze_routing(user_input)
                    self.print_routing_decision(analysis)
                    self.print_performance_comparison(analysis)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye! (Ctrl+C)")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()


def main() -> None:
    """Entry point."""
    try:
        demo = GRAGRouterDemo()
        demo.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
