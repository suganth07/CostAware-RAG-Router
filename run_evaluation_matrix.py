#!/usr/bin/env python
"""Comprehensive Evaluation Matrix for GRAG System.

This script runs a complete evaluation across all RAG routes and generates
detailed performance metrics comparing:
- Hybrid RAG
- GraphRAG
- Optimized GraphRAG
- Router-Selected Route

Outputs:
- Detailed metrics CSV
- Summary statistics JSON
- Comparison matrix
- Performance visualization data
- Visual evaluation matrix images
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.adapters.datasets import get_dataset_adapter
from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
)
from src.adapters.llm import OllamaLLM, OllamaLLMConfig
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import NumpyVectorStore
from src.evaluation.core import (
    exact_match,
    token_f1,
    context_answer_recall,
)
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig
from src.optimized_grag.models import FeatureStructuralEncoder
from src.rag.bm25 import BM25RAG
from src.rag.hybrid import HybridRAG
from src.rag.normal import NormalRAG
from src.rag.types import RAGConfig
from src.rl_router.policy import LinearSoftmaxBandit
from src.router.features import query_features as get_query_features

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    import numpy as np  # numpy is required anyway
    print("[WARN] Matplotlib not available. Install with: pip install matplotlib")
    print("       Visualization features will be disabled.")


@dataclass(slots=True)
class RouteEvaluation:
    """Evaluation result for a single route on a single question."""
    route: str
    question_id: str
    question: str
    answer: str
    ground_truth: list[str]
    latency_ms: float
    exact_match: float | None
    token_f1: float | None
    context_recall: float | None
    retrieved_units: int
    error: str | None = None


@dataclass(slots=True)
class EvaluationMatrix:
    """Complete evaluation matrix with all metrics."""
    timestamp: str
    dataset: str
    num_questions: int
    route_evaluations: list[RouteEvaluation]
    router_decisions: list[dict[str, Any]]
    summary_metrics: dict[str, dict[str, float]]
    comparison_matrix: dict[str, Any]


class GRAGEvaluator:
    """Comprehensive evaluator for GRAG system."""

    def __init__(
        self,
        dataset_name: str = "metaqa",
        hop: str = "2-hop",
        num_questions: int = 50,
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.hop = hop
        self.num_questions = num_questions
        self.verbose = verbose
        self.pipelines = {}
        self.policy = None

        if self.verbose:
            print("\n" + "="*80)
            print("GRAG COMPREHENSIVE EVALUATION MATRIX")
            print("="*80)
            print(f"\nDataset: {dataset_name} ({hop})")
            print(f"Questions: {num_questions}")
            print("\nInitializing components...")

    def initialize(self) -> None:
        """Initialize all pipelines and router policy."""
        # Load router policy
        self._load_policy()

        # Initialize adapters
        self._initialize_adapters()

        # Load dataset
        self._load_dataset()

        # Initialize pipelines
        self._initialize_pipelines()

        if self.verbose:
            print("\n[READY] All components initialized successfully")

    def _load_policy(self) -> None:
        """Load trained router policy."""
        policy_path = ROOT / "artifacts/router/policies/m4_contextual_bandit.json"

        if self.verbose:
            print(f"\n  Loading router policy...")

        if not policy_path.exists():
            print(f"  [WARN] Policy not found at {policy_path}")
            print(f"         Router evaluation will be skipped")
            self.policy = None
            return

        with open(policy_path) as f:
            policy_dict = json.load(f)

        self.policy = LinearSoftmaxBandit.from_dict(policy_dict["policy"])

        if self.verbose:
            print(f"  [OK] Policy loaded")
            print(f"       Actions: {', '.join(self.policy.actions)}")

    def _initialize_adapters(self) -> None:
        """Initialize embedding and LLM adapters."""
        if self.verbose:
            print(f"\n  Initializing adapters...")

        ollama_base_url = "http://localhost:11434"

        self.embedding = CachingEmbeddingAdapter(
            OllamaEmbeddingAdapter(
                OllamaEmbeddingConfig(
                    model="nomic-embed-text",
                    base_url=ollama_base_url,
                    keep_alive="30m",
                    timeout_seconds=60,
                    batch_size=8,
                    options={"num_gpu": 999},
                )
            ),
            cache_path=ROOT / "artifacts/cache/embeddings.sqlite3",
            namespace="nomic-embed-text",
            model_name="nomic-embed-text",
        )

        self.llm = OllamaLLM(
            OllamaLLMConfig(
                model="qwen3:4b-instruct",
                base_url=ollama_base_url,
                temperature=0.0,
                num_predict=160,
                keep_alive="30m",
                default_options={"num_gpu": 999},
                timeout_seconds=60,
            )
        )

        if self.verbose:
            print(f"  [OK] Adapters initialized")

    def _load_dataset(self) -> None:
        """Load evaluation dataset."""
        if self.verbose:
            print(f"\n  Loading dataset...")

        adapter = get_dataset_adapter(
            self.dataset_name,
            hop=self.hop,
            variant="vanilla",
            split="test"
        )
        self.bundle = adapter.load()
        self.samples = self.bundle.eval_samples[:self.num_questions]

        if self.verbose:
            print(f"  [OK] Loaded {len(self.samples)} evaluation samples")
            print(f"       Documents: {len(self.bundle.documents)}")
            print(f"       Graph nodes: {len(self.bundle.graph_nodes)}")
            print(f"       Graph edges: {len(self.bundle.graph_edges)}")

    def _initialize_pipelines(self) -> None:
        """Initialize all RAG pipelines."""
        if self.verbose:
            print(f"\n  Initializing pipelines...")

        graph_store = InMemoryGraphStore()
        vector_store = NumpyVectorStore(
            cache_directory=ROOT / "artifacts/cache/vector_indexes"
        )

        # Normal RAG for Hybrid
        normal_rag = NormalRAG(
            vector_store=vector_store,
            embedding_adapter=self.embedding,
            llm=self.llm,
            config=RAGConfig(chunk_size=900, chunk_overlap=120, top_k=12),
        )

        # BM25 RAG for Hybrid
        bm25_rag = BM25RAG(
            llm=self.llm,
            config=RAGConfig(top_k=12),
        )

        # Hybrid RAG
        self.pipelines["hybrid_rag"] = HybridRAG(
            normal_rag=normal_rag,
            bm25_rag=bm25_rag,
            llm=self.llm,
            config=RAGConfig(chunk_size=900, chunk_overlap=120, top_k=12),
        )

        # GraphRAG
        self.pipelines["graph_rag"] = GraphRAG(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_adapter=self.embedding,
            llm=self.llm,
            config=GraphRAGConfig(
                chunk_size=900,
                chunk_overlap=120,
                top_k=12,
                vector_candidate_k=64,
                vector_seed_k=12,
                graph_depth=3,
                graph_limit=1200,
            ),
        )

        # Optimized GraphRAG
        self.pipelines["optimized_graph_rag"] = OptimizedGraphRAG(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_adapter=self.embedding,
            llm=self.llm,
            config=OptimizedGraphRAGConfig(
                chunk_size=900,
                chunk_overlap=120,
                max_region_nodes=1000,
                max_capsule_nodes=250,
                max_capsule_edges=900,
                top_k_capsules=12,
                max_merged_nodes=2400,
                max_candidate_paths=11,
                max_path_length=3,
                final_paths_to_llm=11,
            ),
            structural_encoder=FeatureStructuralEncoder(),
        )

        # Index dataset in all pipelines
        if self.verbose:
            print(f"\n  Indexing dataset in all pipelines...")

        for route, pipeline in self.pipelines.items():
            if self.verbose:
                print(f"    Indexing {route}...")

            if route in {"graph_rag", "optimized_graph_rag"}:
                pipeline.index_documents(
                    self.bundle.documents,
                    extra_graph_nodes=self.bundle.graph_nodes,
                    extra_graph_edges=self.bundle.graph_edges,
                )
            else:
                pipeline.index_documents(self.bundle.documents)

        if self.verbose:
            print(f"  [OK] All pipelines indexed")

    def evaluate(self) -> EvaluationMatrix:
        """Run comprehensive evaluation across all routes."""
        if self.verbose:
            print("\n" + "="*80)
            print("RUNNING EVALUATION")
            print("="*80)

        route_evaluations = []
        router_decisions = []

        for idx, sample in enumerate(self.samples, 1):
            if self.verbose:
                print(f"\n[{idx}/{len(self.samples)}] {sample.question[:60]}...")

            # Get router decision if policy is available
            router_decision = None
            selected_route = None
            if self.policy:
                features = get_query_features(sample.question)
                probs = self.policy.probabilities(features)
                selected_route = self.policy.choose(features)
                router_decision = {
                    "question_id": sample.id,
                    "question": sample.question,
                    "features": features,
                    "probabilities": probs,
                    "selected_route": selected_route,
                }
                router_decisions.append(router_decision)
                if self.verbose:
                    print(f"  Router selected: {selected_route} ({probs[selected_route]:.1%})")

            # Evaluate all routes
            for route_name, pipeline in self.pipelines.items():
                if self.verbose:
                    print(f"  Running {route_name}...", end=" ", flush=True)

                try:
                    started = time.perf_counter()
                    output = pipeline.answer(sample.question)
                    latency_ms = (time.perf_counter() - started) * 1000

                    answer_text = output.answer if hasattr(output, "answer") else str(output)
                    context_text = output.context if hasattr(output, "context") else ""

                    # Count retrieved units
                    if hasattr(output, "context") and hasattr(output.context, "source_snippets"):
                        retrieved_units = len(output.context.source_snippets)
                    elif hasattr(output, "evidence_units"):
                        retrieved_units = len(output.evidence_units)
                    else:
                        retrieved_units = 0

                    # Calculate metrics
                    em = exact_match(answer_text, sample.answers)
                    tf1 = token_f1(answer_text, sample.answers)
                    ctx_recall = context_answer_recall(str(context_text), sample.answers)

                    evaluation = RouteEvaluation(
                        route=route_name,
                        question_id=sample.id,
                        question=sample.question,
                        answer=answer_text,
                        ground_truth=list(sample.answers),
                        latency_ms=latency_ms,
                        exact_match=em,
                        token_f1=tf1,
                        context_recall=ctx_recall,
                        retrieved_units=retrieved_units,
                    )

                    route_evaluations.append(evaluation)

                    if self.verbose:
                        print(f"{latency_ms:.0f}ms, EM={em or 0:.2f}, F1={tf1 or 0:.2f}")

                except Exception as e:
                    if self.verbose:
                        print(f"ERROR: {str(e)[:50]}")

                    route_evaluations.append(
                        RouteEvaluation(
                            route=route_name,
                            question_id=sample.id,
                            question=sample.question,
                            answer="",
                            ground_truth=list(sample.answers),
                            latency_ms=-1,
                            exact_match=None,
                            token_f1=None,
                            context_recall=None,
                            retrieved_units=0,
                            error=str(e),
                        )
                    )

        # Compute summary metrics
        summary_metrics = self._compute_summary_metrics(route_evaluations)

        # Create comparison matrix
        comparison_matrix = self._create_comparison_matrix(
            route_evaluations, router_decisions, summary_metrics
        )

        return EvaluationMatrix(
            timestamp=datetime.now().isoformat(),
            dataset=f"{self.dataset_name}_{self.hop}",
            num_questions=len(self.samples),
            route_evaluations=route_evaluations,
            router_decisions=router_decisions,
            summary_metrics=summary_metrics,
            comparison_matrix=comparison_matrix,
        )

    def _compute_summary_metrics(
        self, evaluations: list[RouteEvaluation]
    ) -> dict[str, dict[str, float]]:
        """Compute summary metrics for each route."""
        by_route = defaultdict(list)
        for eval in evaluations:
            if eval.error is None:
                by_route[eval.route].append(eval)

        summary = {}
        for route, evals in by_route.items():
            valid_em = [e.exact_match for e in evals if e.exact_match is not None]
            valid_f1 = [e.token_f1 for e in evals if e.token_f1 is not None]
            valid_cr = [e.context_recall for e in evals if e.context_recall is not None]
            latencies = [e.latency_ms for e in evals]
            units = [e.retrieved_units for e in evals]

            summary[route] = {
                "num_questions": len(evals),
                "exact_match_mean": sum(valid_em) / len(valid_em) if valid_em else 0.0,
                "token_f1_mean": sum(valid_f1) / len(valid_f1) if valid_f1 else 0.0,
                "context_recall_mean": sum(valid_cr) / len(valid_cr) if valid_cr else 0.0,
                "latency_mean_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "latency_p50_ms": sorted(latencies)[len(latencies)//2] if latencies else 0.0,
                "latency_p95_ms": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0.0,
                "retrieved_units_mean": sum(units) / len(units) if units else 0.0,
            }

        return summary

    def _create_comparison_matrix(
        self,
        evaluations: list[RouteEvaluation],
        router_decisions: list[dict],
        summary: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Create detailed comparison matrix."""
        # Router accuracy (if available)
        router_accuracy = None
        if router_decisions:
            # Compare router selection to best-performing route per question
            by_question = defaultdict(list)
            for eval in evaluations:
                if eval.error is None:
                    by_question[eval.question_id].append(eval)

            correct_selections = 0
            total_selections = 0

            for decision in router_decisions:
                qid = decision["question_id"]
                selected = decision["selected_route"]

                # Find best route for this question (by F1)
                question_evals = by_question[qid]
                if question_evals:
                    best_route = max(
                        question_evals,
                        key=lambda e: (e.token_f1 or 0.0, -(e.latency_ms or float('inf')))
                    ).route

                    if selected == best_route:
                        correct_selections += 1
                    total_selections += 1

            router_accuracy = correct_selections / total_selections if total_selections > 0 else 0.0

        # Speed comparisons
        if "optimized_graph_rag" in summary and "graph_rag" in summary:
            speedup = summary["graph_rag"]["latency_mean_ms"] / summary["optimized_graph_rag"]["latency_mean_ms"]
        else:
            speedup = None

        # Quality comparisons
        quality_comparison = {}
        for route, metrics in summary.items():
            quality_comparison[route] = {
                "exact_match": metrics["exact_match_mean"],
                "token_f1": metrics["token_f1_mean"],
                "context_recall": metrics["context_recall_mean"],
            }

        return {
            "router_accuracy": router_accuracy,
            "optimized_vs_normal_speedup": speedup,
            "quality_comparison": quality_comparison,
            "latency_comparison": {
                route: metrics["latency_mean_ms"]
                for route, metrics in summary.items()
            },
            "efficiency_comparison": {
                route: metrics["retrieved_units_mean"]
                for route, metrics in summary.items()
            },
        }

    def save_results(self, matrix: EvaluationMatrix, output_dir: str | Path) -> dict[str, Path]:
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = output_path / f"evaluation_matrix_{timestamp}.json"
        summary_file = output_path / f"summary_metrics_{timestamp}.json"
        comparison_file = output_path / f"comparison_matrix_{timestamp}.json"
        csv_file = output_path / f"detailed_results_{timestamp}.csv"

        # JSON files
        results_file.write_text(
            json.dumps(asdict(matrix), indent=2, default=str),
            encoding="utf-8"
        )

        summary_file.write_text(
            json.dumps(matrix.summary_metrics, indent=2),
            encoding="utf-8"
        )

        comparison_file.write_text(
            json.dumps(matrix.comparison_matrix, indent=2),
            encoding="utf-8"
        )

        # CSV file
        import csv
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "route", "question_id", "question", "answer", "ground_truth",
                "latency_ms", "exact_match", "token_f1", "context_recall",
                "retrieved_units", "error"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for eval in matrix.route_evaluations:
                writer.writerow({
                    "route": eval.route,
                    "question_id": eval.question_id,
                    "question": eval.question,
                    "answer": eval.answer,
                    "ground_truth": str(eval.ground_truth),
                    "latency_ms": eval.latency_ms,
                    "exact_match": eval.exact_match,
                    "token_f1": eval.token_f1,
                    "context_recall": eval.context_recall,
                    "retrieved_units": eval.retrieved_units,
                    "error": eval.error or "",
                })

        paths = {
            "results": results_file,
            "summary": summary_file,
            "comparison": comparison_file,
            "csv": csv_file,
        }

        # Generate visualizations if available
        if VISUALIZATION_AVAILABLE:
            try:
                viz_paths = self._generate_visualizations(matrix, output_path, timestamp)
                paths.update(viz_paths)
            except Exception as e:
                print(f"[WARN] Failed to generate visualizations: {e}")

        return paths

    def _generate_visualizations(
        self, matrix: EvaluationMatrix, output_path: Path, timestamp: str
    ) -> dict[str, Path]:
        """Generate visual evaluation matrices."""
        viz_files = {}

        # 1. Performance comparison heatmap
        heatmap_file = output_path / f"performance_heatmap_{timestamp}.png"
        self._create_performance_heatmap(matrix, heatmap_file)
        viz_files["performance_heatmap"] = heatmap_file

        # 2. Quality vs Latency scatter
        scatter_file = output_path / f"quality_vs_latency_{timestamp}.png"
        self._create_quality_latency_scatter(matrix, scatter_file)
        viz_files["quality_latency_scatter"] = scatter_file

        # 3. Detailed metrics comparison
        comparison_file = output_path / f"metrics_comparison_{timestamp}.png"
        self._create_metrics_comparison(matrix, comparison_file)
        viz_files["metrics_comparison"] = comparison_file

        # 4. Router decision analysis (if available)
        if matrix.router_decisions:
            router_file = output_path / f"router_analysis_{timestamp}.png"
            self._create_router_analysis(matrix, router_file)
            viz_files["router_analysis"] = router_file

        return viz_files

    def _create_performance_heatmap(self, matrix: EvaluationMatrix, output_file: Path) -> None:
        """Create performance heatmap showing all metrics."""
        routes = sorted(matrix.summary_metrics.keys())
        metrics_labels = ["Exact Match", "Token F1", "Context Recall", "Latency (norm)", "Efficiency (norm)"]

        # Prepare data matrix
        data = []
        for route in routes:
            metrics = matrix.summary_metrics[route]
            row = [
                metrics["exact_match_mean"] * 100,
                metrics["token_f1_mean"] * 100,
                metrics["context_recall_mean"] * 100,
                0,  # Will normalize latency
                0,  # Will normalize units
            ]
            data.append(row)

        # Normalize latency and units (inverse for better = greener)
        latencies = [matrix.summary_metrics[r]["latency_mean_ms"] for r in routes]
        units = [matrix.summary_metrics[r]["retrieved_units_mean"] for r in routes]

        max_latency = max(latencies) if latencies else 1
        max_units = max(units) if units else 1

        for idx, route in enumerate(routes):
            # Invert so lower latency = higher score (green)
            data[idx][3] = (1 - latencies[idx] / max_latency) * 100
            # Invert so fewer units = higher score (green)
            data[idx][4] = (1 - units[idx] / max_units) * 100

        data_array = np.array(data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(metrics_labels)))
        ax.set_yticks(np.arange(len(routes)))
        ax.set_xticklabels(metrics_labels, rotation=45, ha='right')
        ax.set_yticklabels([r.replace('_', ' ').title() for r in routes])

        # Add text annotations
        for i in range(len(routes)):
            for j in range(len(metrics_labels)):
                text = ax.text(j, i, f'{data_array[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10, fontweight='bold')

        ax.set_title(f"GRAG Performance Heatmap\n{matrix.dataset} - {matrix.num_questions} questions",
                    fontsize=14, fontweight='bold', pad=20)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_quality_latency_scatter(self, matrix: EvaluationMatrix, output_file: Path) -> None:
        """Create quality vs latency scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        routes = list(matrix.summary_metrics.keys())
        colors = {'hybrid_rag': '#3498db', 'graph_rag': '#e74c3c', 'optimized_graph_rag': '#2ecc71'}
        markers = {'hybrid_rag': 'o', 'graph_rag': 's', 'optimized_graph_rag': '^'}

        for route in routes:
            metrics = matrix.summary_metrics[route]
            x = metrics["latency_mean_ms"]
            y = metrics["token_f1_mean"] * 100
            size = metrics["retrieved_units_mean"] * 3

            ax.scatter(x, y, s=size, c=colors.get(route, '#95a5a6'),
                      marker=markers.get(route, 'o'), alpha=0.7, edgecolors='black', linewidth=2,
                      label=route.replace('_', ' ').title())

            # Add route label
            ax.annotate(route.replace('_', '\n').title(), (x, y),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor=colors.get(route, '#95a5a6'), alpha=0.3))

        ax.set_xlabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Token F1 Score (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Quality vs Latency Trade-off\n{matrix.dataset} - {matrix.num_questions} questions',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=10)

        # Add ideal region annotation
        ax.axhline(y=ax.get_ylim()[1]*0.8, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=ax.get_xlim()[1]*0.3, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.text(ax.get_xlim()[0] + 20, ax.get_ylim()[1]*0.82,
               'Ideal Region\n(High Quality, Low Latency)',
               fontsize=9, style='italic', color='green', alpha=0.7)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_metrics_comparison(self, matrix: EvaluationMatrix, output_file: Path) -> None:
        """Create detailed metrics comparison bar chart."""
        routes = sorted(matrix.summary_metrics.keys())
        metrics_to_plot = [
            ("exact_match_mean", "Exact Match", "%"),
            ("token_f1_mean", "Token F1", "%"),
            ("context_recall_mean", "Context Recall", "%"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

        # Plot quality metrics
        for idx, (metric_key, metric_label, unit) in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [matrix.summary_metrics[r][metric_key] * 100 for r in routes]
            bars = ax.bar(range(len(routes)), values, color=colors[:len(routes)], alpha=0.8, edgecolor='black', linewidth=1.5)

            ax.set_xticks(range(len(routes)))
            ax.set_xticklabels([r.replace('_', '\n').title() for r in routes], fontsize=9)
            ax.set_ylabel(f'{metric_label} ({unit})', fontsize=11, fontweight='bold')
            ax.set_title(metric_label, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Plot latency comparison
        ax = axes[3]
        latencies = [matrix.summary_metrics[r]["latency_mean_ms"] for r in routes]
        bars = ax.bar(range(len(routes)), latencies, color=colors[:len(routes)], alpha=0.8, edgecolor='black', linewidth=1.5)

        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels([r.replace('_', '\n').title() for r in routes], fontsize=9)
        ax.set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
        ax.set_title('Average Latency', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Add value labels on bars
        for bar, val in zip(bars, latencies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{val:.0f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

        fig.suptitle(f'Detailed Metrics Comparison\n{matrix.dataset} - {matrix.num_questions} questions',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_router_analysis(self, matrix: EvaluationMatrix, output_file: Path) -> None:
        """Create router decision analysis visualization."""
        if not matrix.router_decisions:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Route selection distribution
        ax = axes[0, 0]
        route_counts = defaultdict(int)
        for decision in matrix.router_decisions:
            route_counts[decision["selected_route"]] += 1

        routes = list(route_counts.keys())
        counts = [route_counts[r] for r in routes]
        colors_map = {'hybrid_rag': '#3498db', 'graph_rag': '#e74c3c', 'optimized_graph_rag': '#2ecc71'}
        colors = [colors_map.get(r, '#95a5a6') for r in routes]

        wedges, texts, autotexts = ax.pie(counts, labels=[r.replace('_', '\n').title() for r in routes],
                                           autopct='%1.1f%%', colors=colors, startangle=90,
                                           textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_title('Router Selection Distribution', fontsize=12, fontweight='bold')

        # 2. Confidence scores
        ax = axes[0, 1]
        confidences = []
        selected_routes = []
        for decision in matrix.router_decisions:
            selected = decision["selected_route"]
            prob = decision["probabilities"][selected]
            confidences.append(prob * 100)
            selected_routes.append(selected)

        route_confidences = defaultdict(list)
        for route, conf in zip(selected_routes, confidences):
            route_confidences[route].append(conf)

        bp_data = [route_confidences[r] for r in routes]
        bp = ax.boxplot(bp_data, labels=[r.replace('_', '\n').title() for r in routes],
                       patch_artist=True, showmeans=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Confidence (%)', fontsize=11, fontweight='bold')
        ax.set_title('Router Confidence Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 3. Router accuracy
        ax = axes[1, 0]
        accuracy = matrix.comparison_matrix.get("router_accuracy", 0) * 100

        # Create accuracy gauge
        ax.barh([0], [accuracy], height=0.5, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2)
        ax.barh([0], [100-accuracy], left=[accuracy], height=0.5, color='#e74c3c', alpha=0.3, edgecolor='black', linewidth=2)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_yticks([])
        ax.set_title('Router Accuracy', fontsize=12, fontweight='bold')
        ax.text(accuracy/2, 0, f'{accuracy:.1f}%', ha='center', va='center',
               fontsize=14, fontweight='bold', color='white')
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')

        # 4. Performance gain
        ax = axes[1, 1]
        speedup = matrix.comparison_matrix.get("optimized_vs_normal_speedup", 0)

        if speedup > 0:
            categories = ['Optimized\nvs\nNormal']
            gains = [speedup]

            bars = ax.bar(categories, gains, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=2, width=0.4)
            ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (1.0x)')
            ax.set_ylabel('Speedup Factor', fontsize=11, fontweight='bold')
            ax.set_title('Performance Speedup', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.legend()

            # Add value label
            for bar, val in zip(bars, gains):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{val:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

        fig.suptitle(f'Router Analysis\n{matrix.dataset} - {matrix.num_questions} questions',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def print_summary(self, matrix: EvaluationMatrix) -> None:
        """Print summary of evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)

        print(f"\nDataset: {matrix.dataset}")
        print(f"Questions Evaluated: {matrix.num_questions}")
        print(f"Timestamp: {matrix.timestamp}")

        print("\n" + "-"*80)
        print("ROUTE PERFORMANCE METRICS")
        print("-"*80)

        print(f"\n{'Route':<25} {'EM':>8} {'F1':>8} {'C-Recall':>10} {'Latency':>12} {'Units':>8}")
        print("-"*80)

        for route, metrics in sorted(matrix.summary_metrics.items()):
            print(
                f"{route:<25} "
                f"{metrics['exact_match_mean']:>7.1%} "
                f"{metrics['token_f1_mean']:>7.1%} "
                f"{metrics['context_recall_mean']:>9.1%} "
                f"{metrics['latency_mean_ms']:>10.0f} ms "
                f"{metrics['retrieved_units_mean']:>8.0f}"
            )

        print("\n" + "-"*80)
        print("COMPARISON ANALYSIS")
        print("-"*80)

        comp = matrix.comparison_matrix

        if comp.get("router_accuracy") is not None:
            print(f"\nRouter Accuracy: {comp['router_accuracy']:.1%}")
            print("  (Percentage of times router selected best-performing route)")

        if comp.get("optimized_vs_normal_speedup"):
            speedup = comp["optimized_vs_normal_speedup"]
            print(f"\nOptimized vs Normal GraphRAG Speedup: {speedup:.2f}x")
            print(f"  ({((speedup-1)*100):.0f}% faster)")

        print("\nLatency Ranking:")
        for idx, (route, latency) in enumerate(
            sorted(comp["latency_comparison"].items(), key=lambda x: x[1]), 1
        ):
            print(f"  {idx}. {route:<25} {latency:>10.0f} ms")

        print("\nQuality Ranking (by Token F1):")
        quality = {
            route: metrics["token_f1"]
            for route, metrics in comp["quality_comparison"].items()
        }
        for idx, (route, f1) in enumerate(
            sorted(quality.items(), key=lambda x: x[1], reverse=True), 1
        ):
            print(f"  {idx}. {route:<25} F1={f1:>6.1%}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive GRAG evaluation matrix"
    )
    parser.add_argument(
        "--dataset",
        default="metaqa",
        help="Dataset name (default: metaqa)"
    )
    parser.add_argument(
        "--hop",
        default="2-hop",
        choices=["1-hop", "2-hop", "3-hop"],
        help="Hop level for MetaQA (default: 2-hop)"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50)"
    )
    parser.add_argument(
        "--output",
        default="results/evaluation_matrix",
        help="Output directory for results (default: results/evaluation_matrix)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    try:
        evaluator = GRAGEvaluator(
            dataset_name=args.dataset,
            hop=args.hop,
            num_questions=args.questions,
            verbose=not args.quiet,
        )

        evaluator.initialize()
        matrix = evaluator.evaluate()
        paths = evaluator.save_results(matrix, args.output)
        evaluator.print_summary(matrix)

        print("\n" + "="*80)
        print("RESULTS SAVED")
        print("="*80)
        for name, path in paths.items():
            print(f"  {name:15s}: {path}")

        print("\n[SUCCESS] Evaluation complete!")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Evaluation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
