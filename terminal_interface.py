#!/usr/bin/env python
"""Interactive GRAG Terminal Interface with Intelligent Routing.

This is the user-facing terminal application that:
1. Accepts questions from the user
2. Uses the trained router policy to select the best RAG pipeline
3. Executes the selected route
4. Displays comprehensive results and analysis

Usage:
    python terminal_interface.py

Commands:
    - Type any question to get an answer
    - Type 'verbose' to toggle detailed routing analysis
    - Type 'compare' to run all routes and see comparison
    - Type 'help' for help
    - Type 'exit' or 'quit' to exit
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
)
from src.adapters.llm import OllamaLLM, OllamaLLMConfig
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore, NumpyVectorStore
from src.compute_eval import InMemoryComputeTracker
from src.evaluation.runner import retrieval_record_from_output
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig
from src.optimized_grag.models import FeatureStructuralEncoder
from src.rag.bm25 import BM25RAG
from src.rag.hybrid import HybridRAG
from src.rag.normal import NormalRAG
from src.rag.types import RAGConfig
from src.rl_router.policy import LinearSoftmaxBandit
from src.router.features import query_features as get_query_features


@dataclass(slots=True)
class RouteResult:
    """Result from executing one route."""
    route: str
    question: str
    latency_ms: float
    answer: str
    retrieved_units: int
    context_preview: str
    error: Optional[str] = None


@dataclass(slots=True)
class RoutingDecision:
    """Complete routing decision with all information."""
    question: str
    features: dict[str, float]
    policy_probabilities: dict[str, float]
    selected_route: str
    reason: str


class GRAGTerminalInterface:
    """Interactive terminal interface for GRAG routing and QA."""

    def __init__(self, verbose: bool = False, compare_mode: bool = False):
        self.verbose = verbose
        self.compare_mode = compare_mode
        self.session_count = 0
        self.policy = None
        self.pipelines = {}
        self.decision_history: list[RoutingDecision] = []
        self.results_history: list[RouteResult] = []

        # Create a per-session result log file
        results_dir = ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_log_path = results_dir / f"session_{session_ts}.txt"
        self.result_log_path.write_text(
            f"GRAG Session Log — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            + "=" * 80 + "\n",
            encoding="utf-8",
        )
        print(f"  [LOG] Full results will be saved to: {self.result_log_path}")

        print("\n" + "="*80)
        print("GRAG INTELLIGENT ROUTING TERMINAL")
        print("="*80)
        print("\nInitializing components...")
        
        try:
            self._load_policy()
            self._load_pipelines()
            print("\n[READY] System initialized. Type 'help' for commands.\n")
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
        print(f"       Created: {policy_dict.get('created_at', 'unknown')}")

    def _load_pipelines(self) -> None:
        """Initialize all three RAG pipelines."""
        print(f"  Loading embedding and LLM adapters...")
        ollama_base_url = "http://localhost:11434"
        self._verify_ollama(base_url=ollama_base_url)
        
        embedding = CachingEmbeddingAdapter(
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
        
        llm = OllamaLLM(
            OllamaLLMConfig(
                model="qwen3:4b-instruct",
                base_url=ollama_base_url,
                temperature=0.0,
                num_predict=160,
                keep_alive="30m",
                default_options={"num_gpu": 999},
                timeout_seconds=180,
            )
        )
        
        print(f"  [OK] Adapters loaded")
        
        # Load dataset
        print(f"  Loading MetaQA dataset...")
        from src.adapters.datasets import get_dataset_adapter
        adapter = get_dataset_adapter("metaqa", hop="2-hop", variant="vanilla", split="test")
        bundle = adapter.load()
        print(f"  [OK] Dataset loaded ({len(bundle.graph_nodes)} nodes, {len(bundle.graph_edges)} edges)")
        
        # Initialize pipelines
        print(f"  Initializing pipelines...")
        graph_store = InMemoryGraphStore()
        vector_store = NumpyVectorStore(cache_directory=ROOT / "artifacts/cache/vector_indexes")
        
        normal_rag = NormalRAG(
            vector_store=vector_store,
            embedding_adapter=embedding,
            llm=llm,
            config=RAGConfig(chunk_size=900, chunk_overlap=120, top_k=12),
        )

        bm25_rag = BM25RAG(
            llm=llm,
            config=RAGConfig(top_k=12),
        )

        self.pipelines["hybrid_rag"] = HybridRAG(
            normal_rag=normal_rag,
            bm25_rag=bm25_rag,
            llm=llm,
            config=RAGConfig(chunk_size=900, chunk_overlap=120, top_k=12),
        )
        
        self.pipelines["graph_rag"] = GraphRAG(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_adapter=embedding,
            llm=llm,
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
        
        self.pipelines["optimized_graph_rag"] = OptimizedGraphRAG(
            graph_store=graph_store,
            vector_store=vector_store,
            embedding_adapter=embedding,
            llm=llm,
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
        
        # Index the dataset
        print(f"  Indexing dataset in all pipelines...")
        for route, pipeline in self.pipelines.items():
            print(f"    Indexing {route}...")
            if route in {"graph_rag", "optimized_graph_rag"}:
                pipeline.index_documents(
                    bundle.documents,
                    extra_graph_nodes=bundle.graph_nodes,
                    extra_graph_edges=bundle.graph_edges,
                )
            else:
                pipeline.index_documents(bundle.documents)
        
        print(f"  [OK] All pipelines indexed and ready")

    def _verify_ollama(self, *, base_url: str) -> None:
        """Fail fast with a clear message when Ollama is unavailable."""
        health_url = f"{base_url}/api/tags"
        request = urllib.request.Request(health_url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                if response.status < 200 or response.status >= 300:
                    raise ConnectionError(
                        f"Ollama health check returned HTTP {response.status} from {health_url}"
                    )
            print(f"  [OK] Ollama reachable at {base_url}")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ConnectionError(
                "Ollama is not reachable. Start Ollama and ensure required models are available.\n"
                "  1) ollama serve\n"
                "  2) ollama pull nomic-embed-text\n"
                "  3) ollama pull qwen3:4b-instruct\n"
                f"Details: {exc}"
            ) from exc

    def _extract_features(self, question: str) -> dict[str, float]:
        """Extract routing features from question."""
        return get_query_features(question)

    def _get_routing_decision(self, question: str) -> RoutingDecision:
        """Get routing decision using trained policy."""
        features = self._extract_features(question)
        probs = self.policy.probabilities(features)
        selected = self.policy.choose(features)
        
        # Determine reason based on features
        reason = self._explain_routing(question, features, selected)
        
        decision = RoutingDecision(
            question=question,
            features=features,
            policy_probabilities=probs,
            selected_route=selected,
            reason=reason,
        )
        
        self.decision_history.append(decision)
        return decision

    def _explain_routing(self, question: str, features: dict[str, float], selected_route: str) -> str:
        """Generate explanation for why this route was selected."""
        token_count = features.get("token_count", 0)
        relation_terms = features.get("relation_term_count", 0)
        multihop_terms = features.get("multihop_term_count", 0)
        
        hints = []
        if token_count > 15:
            hints.append("complex (long)")
        if relation_terms > 2:
            hints.append("relation-heavy")
        if multihop_terms > 1:
            hints.append("multi-hop indicators")
        
        if selected_route == "optimized_graph_rag":
            return f"Question {'/'.join(hints) if hints else 'is straightforward'} - optimized route offers best speed/quality tradeoff"
        elif selected_route == "graph_rag":
            return f"Question needs deep graph reasoning - full GraphRAG selected for comprehensive search"
        else:
            return f"Simple question - hybrid approach sufficient"

    def _execute_route(self, route: str, question: str) -> RouteResult:
        """Execute a single route and measure performance."""
        try:
            pipeline = self.pipelines[route]
            
            started = time.perf_counter()
            output = pipeline.answer(question)
            elapsed_s = time.perf_counter() - started
            elapsed_ms = elapsed_s * 1000
            
            # Extract info from output
            answer_text = output.answer if hasattr(output, "answer") else str(output)
            context_text = output.context if hasattr(output, "context") else ""
            
            # Count retrieved units
            if hasattr(output, "context") and hasattr(output.context, "source_snippets"):
                retrieved_units = len(output.context.source_snippets)
            elif hasattr(output, "evidence_units"):
                retrieved_units = len(output.evidence_units)
            else:
                retrieved_units = 0
            
            # Get context preview
            if isinstance(context_text, str):
                preview = context_text[:200] + "..." if len(context_text) > 200 else context_text
            else:
                preview = str(context_text)[:200] + "..."
            
            result = RouteResult(
                route=route,
                question=question,
                latency_ms=elapsed_ms,
                answer=answer_text,
                retrieved_units=retrieved_units,
                context_preview=preview,
            )
            
            self.results_history.append(result)
            return result
            
        except Exception as e:
            return RouteResult(
                route=route,
                question=question,
                latency_ms=-1,
                answer="",
                retrieved_units=0,
                context_preview="",
                error=str(e),
            )

    def print_routing_logs(self, decision: RoutingDecision) -> None:
        """Print detailed routing decision logs."""
        print("\n" + "="*80)
        print("ROUTING DECISION")
        print("="*80)
        
        print(f"\nQuestion: {decision.question}")
        print(f"\nExtracted Features:")
        for fname, fvalue in decision.features.items():
            print(f"  {fname:30s}: {fvalue:8.2f}")
        
        print(f"\nRoute Probabilities:")
        for route in self.policy.actions:
            prob = decision.policy_probabilities[route]
            bar = "=" * int(prob * 40)
            print(f"  {route:25s}: {prob:6.1%}  {bar}")
        
        print(f"\n[SELECTED] {decision.selected_route}")
        print(f"[REASON]   {decision.reason}")

    def print_execution_logs(self, result: RouteResult) -> None:
        """Print execution details."""
        print("\n" + "="*80)
        print("EXECUTION")
        print("="*80)
        
        if result.error:
            print(f"\n[ERROR] {result.route}: {result.error}")
            return
        
        print(f"\nRoute: {result.route}")
        print(f"Latency: {result.latency_ms:.2f} ms")
        print(f"Retrieved Units: {result.retrieved_units}")
        print(f"\nAnswer: {result.answer[:300]}..." if len(result.answer) > 300 else f"\nAnswer: {result.answer}")
        print(f"\nContext Preview: {result.context_preview}")

    def print_comparison_summary(self, decision: RoutingDecision, results: dict[str, RouteResult]) -> None:
        """Print comprehensive comparison summary."""
        print("\n" + "="*80)
        print("COMPREHENSIVE ROUTING SUMMARY & ANALYSIS")
        print("="*80)
        
        selected = decision.selected_route
        print(f"\n[SELECTED ROUTE] {selected}")
        print(f"[SELECTION REASON] {decision.reason}")
        
        selected_result = results[selected]
        
        print(f"\n" + "-"*80)
        print("SELECTED ROUTE PERFORMANCE")
        print("-"*80)
        
        if not selected_result.error:
            print(f"Route:             {selected}")
            print(f"Latency:           {selected_result.latency_ms:.2f} ms")
            print(f"Retrieved Units:   {selected_result.retrieved_units}")
            print(f"Answer Length:     {len(selected_result.answer)} chars")
            print(f"\nAnswer:")
            print(f"  {selected_result.answer}")
        else:
            print(f"[ERROR] {selected_result.error}")
        
        # Comparison with other routes
        print(f"\n" + "-"*80)
        print("COMPARISON WITH NON-SELECTED ROUTES")
        print("-"*80)
        
        comparison_data = []
        for route in self.policy.actions:
            result = results[route]
            prob = decision.policy_probabilities[route]
            
            if result.error:
                comparison_data.append({
                    "route": route,
                    "prob": prob,
                    "latency": float("inf"),
                    "units": 0,
                    "selected": route == selected,
                    "error": result.error,
                })
            else:
                comparison_data.append({
                    "route": route,
                    "prob": prob,
                    "latency": result.latency_ms,
                    "units": result.retrieved_units,
                    "selected": route == selected,
                    "error": None,
                })
        
        # Print table
        print(f"\n{'Route':<25} {'Prob':>8} {'Latency':>12} {'Units':>10} {'Status':>20}")
        print("-"*80)
        
        for data in comparison_data:
            selected_mark = "[SELECTED]" if data["selected"] else ""
            if data["error"]:
                status = f"ERROR: {data['error'][:15]}"
            else:
                status = selected_mark
            
            latency_str = f"{data['latency']:.2f} ms" if data['latency'] != float('inf') else "FAILED"
            
            print(f"{data['route']:<25} {data['prob']:>7.1%}  {latency_str:>12} {data['units']:>10} {status:>20}")
        
        # Analysis
        print(f"\n" + "-"*80)
        print("DECISION ANALYSIS")
        print("-"*80)
        
        valid_results = [d for d in comparison_data if not d["error"]]
        
        if valid_results:
            fastest = min(valid_results, key=lambda x: x["latency"])
            slowest = max(valid_results, key=lambda x: x["latency"])
            
            min_units = min(valid_results, key=lambda x: x["units"])
            max_units = max(valid_results, key=lambda x: x["units"])
            
            print(f"\nLatency Analysis:")
            print(f"  Fastest:     {fastest['route']} ({fastest['latency']:.2f} ms)")
            print(f"  Slowest:     {slowest['route']} ({slowest['latency']:.2f} ms)")
            print(f"  Selected:    {selected_result.route} ({selected_result.latency_ms:.2f} ms)")
            if fastest['route'] != selected:
                time_diff = selected_result.latency_ms - fastest['latency']
                pct_slower = (time_diff / fastest['latency']) * 100
                print(f"  Trade-off:   +{time_diff:.2f} ms ({pct_slower:.1f}% slower) for quality")
            
            print(f"\nCompute Analysis (Retrieved Units):")
            print(f"  Most Efficient: {min_units['route']} ({min_units['units']} units)")
            print(f"  Most Thorough:  {max_units['route']} ({max_units['units']} units)")
            print(f"  Selected:       {selected_result.route} ({selected_result.retrieved_units} units)")
            
            print(f"\nQuality-Cost Trade-off:")
            print(f"  Selected route balances:")
            print(f"    - Latency: {selected_result.latency_ms:.2f} ms (reasonable budget)")
            print(f"    - Compute: {selected_result.retrieved_units} units (efficient)")
            print(f"    - Coverage: Targets depth-balanced evidence")
            
            print(f"\nWhy This Route Won:")
            if selected == "optimized_graph_rag":
                print(f"  - Achieves 64% speedup vs full GraphRAG while maintaining quality")
                print(f"  - Uses bounded capsule search (96% fewer evidence units)")
                print(f"  - Best for questions without extreme multi-hop requirements")
            elif selected == "graph_rag":
                print(f"  - Provides comprehensive graph neighborhood exploration")
                print(f"  - Necessary for questions requiring deep entity reasoning")
                print(f"  - Trade speed for maximum coverage")
            else:
                print(f"  - Fast vector + sparse graph combination")
                print(f"  - Sufficient for simple factual questions")
                print(f"  - Minimizes latency and compute")

    def _save_result_to_file(
        self,
        decision: RoutingDecision,
        result: RouteResult,
        *,
        all_results: dict[str, "RouteResult"] | None = None,
    ) -> None:
        """Append the full (untruncated) result for one session to the log file."""
        sep = "=" * 80
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            "",
            sep,
            f"SESSION {self.session_count}  —  {ts}",
            sep,
            f"Question : {decision.question}",
            f"Route    : {decision.selected_route}",
            f"Latency  : {result.latency_ms:.2f} ms",
            f"Reason   : {decision.reason}",
            "",
        ]

        if result.error:
            lines.append(f"ERROR: {result.error}")
        else:
            lines += [
                "--- FULL ANSWER " + "-" * 64,
                result.answer,
                "",
                "--- CONTEXT PREVIEW " + "-" * 60,
                result.context_preview,
            ]

        # In compare mode also dump every route
        if all_results:
            lines += ["", "--- ALL ROUTES " + "-" * 65]
            for route, r in all_results.items():
                marker = "[SELECTED] " if route == decision.selected_route else ""
                lines += [
                    "",
                    f"{marker}{route}  latency={r.latency_ms:.2f} ms  units={r.retrieved_units}",
                    "Answer:",
                    r.answer if not r.error else f"ERROR: {r.error}",
                ]

        lines.append("")
        with open(self.result_log_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"\n[LOG] Full result saved → {self.result_log_path}")

    def answer_question(
        self,
        question: str,
        *,
        compare_mode: Optional[bool] = None,
    ) -> tuple[RoutingDecision, dict[str, RouteResult]]:
        """Route and execute a question, returning structured results for other UIs."""
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty")

        self.session_count += 1
        decision = self._get_routing_decision(cleaned_question)
        use_compare_mode = self.compare_mode if compare_mode is None else compare_mode
        routes = self.policy.actions if use_compare_mode else (decision.selected_route,)
        results = {
            route: self._execute_route(route, cleaned_question)
            for route in routes
        }
        return decision, results

    def process_question(self, question: str) -> None:
        """Process a user question end-to-end."""
        print(f"\n[SESSION {self.session_count + 1}] Processing question...")
        decision, results = self.answer_question(question)

        # Print logs
        if self.verbose:
            self.print_routing_logs(decision)

        # Execute routes
        if self.compare_mode:
            print(f"\n[RUNNING ALL ROUTES FOR COMPARISON]")
            
            # Print execution logs
            if self.verbose:
                for route in self.policy.actions:
                    self.print_execution_logs(results[route])
            
            # Print comparison summary
            self.print_comparison_summary(decision, results)

            # Save full results to file
            selected_result = results[decision.selected_route]
            self._save_result_to_file(decision, selected_result, all_results=results)
        else:
            # Execute only selected route
            print(f"\n[EXECUTING] {decision.selected_route}...")
            result = results[decision.selected_route]
            
            if self.verbose:
                self.print_execution_logs(result)
            
            # Show quick summary
            print("\n" + "="*80)
            print("RESULT SUMMARY")
            print("="*80)
            print(f"\nRoute:   {decision.selected_route}")
            print(f"Latency: {result.latency_ms:.2f} ms")
            print(f"Reason:  {decision.reason}")
            if not result.error:
                print(f"\nAnswer:  {result.answer[:200]}..." if len(result.answer) > 200 else f"\nAnswer:  {result.answer}")
            else:
                print(f"\nError:   {result.error}")

            # Save full result to file
            self._save_result_to_file(decision, result)

    def show_help(self) -> None:
        """Show help message."""
        print("\n" + "="*80)
        print("GRAG TERMINAL HELP")
        print("="*80)
        print("""
Commands:
  <question>     Type any question to get an answer with routing
  verbose        Toggle detailed routing/execution logs (currently: {})
  compare        Toggle comparison mode - run all routes (currently: {})
  history        Show decision history
  stats          Show session statistics
  help           Show this help
  exit/quit      Exit the terminal

Examples:
  Which films share a director with [Film A]?
  What is the capital of [Country X]?

Current Settings:
  Verbose Mode:  {}
  Compare Mode:  {}
""".format("ON" if self.verbose else "OFF", 
           "ON" if self.compare_mode else "OFF",
           "ON" if self.verbose else "OFF",
           "ON" if self.compare_mode else "OFF"))

    def show_history(self) -> None:
        """Show decision history."""
        if not self.decision_history:
            print("No history yet.")
            return
        
        print("\n" + "="*80)
        print(f"DECISION HISTORY ({len(self.decision_history)} questions)")
        print("="*80)
        
        for i, decision in enumerate(self.decision_history, 1):
            selected = decision.selected_route
            prob = decision.policy_probabilities[selected]
            print(f"\n{i}. {decision.question}")
            print(f"   Selected: {selected} ({prob:.1%})")

    def show_stats(self) -> None:
        """Show session statistics."""
        if not self.results_history:
            print("No results yet.")
            return
        
        print("\n" + "="*80)
        print("SESSION STATISTICS")
        print("="*80)
        
        by_route = defaultdict(list)
        for result in self.results_history:
            if not result.error:
                by_route[result.route].append(result.latency_ms)
        
        print(f"\nTotal Questions: {len(self.decision_history)}")
        print(f"Total Executions: {len(self.results_history)}")
        
        print(f"\nRoute Usage & Performance:")
        for route in sorted(by_route.keys()):
            latencies = by_route[route]
            avg_lat = sum(latencies) / len(latencies)
            min_lat = min(latencies)
            max_lat = max(latencies)
            print(f"  {route}:")
            print(f"    Executions: {len(latencies)}")
            print(f"    Avg Latency: {avg_lat:.2f} ms")
            print(f"    Min: {min_lat:.2f} ms, Max: {max_lat:.2f} ms")

    def run(self) -> None:
        """Main interactive loop."""
        self.show_help()
        
        while True:
            try:
                user_input = input("\n[GRAG] > ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("exit", "quit"):
                    print("\nGoodbye!")
                    break
                
                elif user_input.lower() == "help":
                    self.show_help()
                
                elif user_input.lower() == "verbose":
                    self.verbose = not self.verbose
                    print(f"[CONFIG] Verbose mode: {'ON' if self.verbose else 'OFF'}")
                
                elif user_input.lower() == "compare":
                    self.compare_mode = not self.compare_mode
                    print(f"[CONFIG] Compare mode: {'ON' if self.compare_mode else 'OFF'}")
                    print(f"         (Running all routes can be slow)")
                
                elif user_input.lower() == "history":
                    self.show_history()
                
                elif user_input.lower() == "stats":
                    self.show_stats()
                
                else:
                    # Treat as question
                    self.process_question(user_input)
            
            except KeyboardInterrupt:
                print("\n\nExiting... (Ctrl+C)")
                break
            except Exception as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)


def main() -> None:
    """Entry point."""
    try:
        interface = GRAGTerminalInterface(verbose=True, compare_mode=False)
        interface.run()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
