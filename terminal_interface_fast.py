#!/usr/bin/env python
"""Fast-start GRAG terminal interface.

This variant boots much faster than the full interface by indexing only a
bounded subset of documents and graph artifacts.

Usage:
    python terminal_interface_fast.py
    python terminal_interface_fast.py --docs 2500 --nodes 7000 --edges 20000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from src.adapters.embedding import (
    CachingEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    OllamaEmbeddingConfig,
)
from src.adapters.llm import OllamaLLM, OllamaLLMConfig
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import NumpyVectorStore
from src.g_rag import GraphRAG, GraphRAGConfig
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig
from src.optimized_grag.models import FeatureStructuralEncoder
from src.rag.bm25 import BM25RAG
from src.rag.hybrid import HybridRAG
from src.rag.normal import NormalRAG
from src.rag.types import RAGConfig

from terminal_interface import GRAGTerminalInterface


ROOT = Path(__file__).resolve().parent


class FastStartGRAGTerminalInterface(GRAGTerminalInterface):
    """Drop-in terminal interface with faster indexing for quick validation."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        compare_mode: bool = False,
        max_docs: int = 2000,
        max_graph_nodes: int = 5000,
        max_graph_edges: int = 15000,
        seed_query: str | None = None,
    ) -> None:
        self.max_docs = max(1, int(max_docs))
        self.max_graph_nodes = max(0, int(max_graph_nodes))
        self.max_graph_edges = max(0, int(max_graph_edges))
        self.seed_query = (seed_query or "").strip()
        super().__init__(verbose=verbose, compare_mode=compare_mode)

    def _query_terms(self, text: str) -> list[str]:
        terms = re.findall(r"[A-Za-z0-9_]+", text.lower())
        # Ignore short stop-like tokens and keep order.
        seen: set[str] = set()
        filtered: list[str] = []
        for term in terms:
            if len(term) < 3:
                continue
            if term in {"the", "with", "from", "that", "which", "what", "who", "where", "when", "and"}:
                continue
            if term not in seen:
                seen.add(term)
                filtered.append(term)
        return filtered

    def _doc_text(self, doc: Any) -> str:
        if isinstance(doc, dict):
            return str(doc.get("text", "")).lower()
        return str(getattr(doc, "text", "")).lower()

    def _node_text(self, node: Any) -> str:
        node_id = str(getattr(node, "id", ""))
        labels = " ".join(getattr(node, "labels", ()) or ())
        props = getattr(node, "properties", {}) or {}
        props_text = " ".join(str(v) for v in props.values())
        return f"{node_id} {labels} {props_text}".lower()

    def _select_fast_subset(
        self,
        *,
        documents: list[Any],
        graph_nodes: list[Any],
        graph_edges: list[Any],
    ) -> tuple[list[Any], list[Any], list[Any], int, int]:
        """Select a bounded subset, optionally biased toward the seed query."""
        terms = self._query_terms(self.seed_query)
        if not terms:
            return (
                documents[: self.max_docs],
                graph_nodes[: self.max_graph_nodes],
                graph_edges[: self.max_graph_edges],
                0,
                0,
            )

        # Score all documents by count of matched query terms.
        scored_docs: list[tuple[int, int]] = []
        doc_matches = 0
        for idx, doc in enumerate(documents):
            text = self._doc_text(doc)
            score = sum(1 for term in terms if term in text)
            if score > 0:
                doc_matches += 1
                scored_docs.append((score, idx))
        scored_docs.sort(key=lambda item: (-item[0], item[1]))

        selected_doc_indices = [idx for _, idx in scored_docs[: self.max_docs]]
        if len(selected_doc_indices) < self.max_docs:
            selected_doc_index_set = set(selected_doc_indices)
            for idx in range(len(documents)):
                if idx not in selected_doc_index_set:
                    selected_doc_indices.append(idx)
                    selected_doc_index_set.add(idx)
                    if len(selected_doc_indices) >= self.max_docs:
                        break
        selected_docs = [documents[idx] for idx in selected_doc_indices]

        # Select query-relevant nodes.
        matched_nodes: list[Any] = []
        selected_node_ids: set[str] = set()
        for node in graph_nodes:
            text = self._node_text(node)
            if any(term in text for term in terms):
                matched_nodes.append(node)
                selected_node_ids.add(str(getattr(node, "id", "")))
                if len(matched_nodes) >= self.max_graph_nodes:
                    break
        if len(matched_nodes) < self.max_graph_nodes:
            for node in graph_nodes:
                node_id = str(getattr(node, "id", ""))
                if node_id in selected_node_ids:
                    continue
                matched_nodes.append(node)
                selected_node_ids.add(node_id)
                if len(matched_nodes) >= self.max_graph_nodes:
                    break

        # Prefer edges attached to selected nodes; then fill.
        selected_edges: list[Any] = []
        for edge in graph_edges:
            src = str(getattr(edge, "source_id", ""))
            dst = str(getattr(edge, "target_id", ""))
            if src in selected_node_ids or dst in selected_node_ids:
                selected_edges.append(edge)
                if len(selected_edges) >= self.max_graph_edges:
                    break
        if len(selected_edges) < self.max_graph_edges:
            selected_edge_ids = {str(getattr(edge, "id", "")) for edge in selected_edges}
            for edge in graph_edges:
                edge_id = str(getattr(edge, "id", ""))
                if edge_id in selected_edge_ids:
                    continue
                selected_edges.append(edge)
                selected_edge_ids.add(edge_id)
                if len(selected_edges) >= self.max_graph_edges:
                    break

        node_matches = sum(1 for node in graph_nodes if any(term in self._node_text(node) for term in terms))
        return selected_docs, matched_nodes, selected_edges, doc_matches, node_matches

    def _load_pipelines(self) -> None:
        """Initialize all pipelines and index only bounded data for fast startup."""
        print("  Loading embedding and LLM adapters...")
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

        print("  [OK] Adapters loaded")

        print("  Loading MetaQA dataset...")
        from src.adapters.datasets import get_dataset_adapter

        adapter = get_dataset_adapter("metaqa", hop="2-hop", variant="vanilla", split="test")
        bundle = adapter.load()

        total_docs = len(bundle.documents)
        total_nodes = len(bundle.graph_nodes)
        total_edges = len(bundle.graph_edges)

        docs, graph_nodes, graph_edges, doc_matches, node_matches = self._select_fast_subset(
            documents=list(bundle.documents),
            graph_nodes=list(bundle.graph_nodes),
            graph_edges=list(bundle.graph_edges),
        )

        print(
            f"  [OK] Dataset loaded (full: docs={total_docs}, nodes={total_nodes}, edges={total_edges})"
        )
        print(
            "  [FAST MODE] Using subset "
            f"docs={len(docs)}/{total_docs}, "
            f"nodes={len(graph_nodes)}/{total_nodes}, "
            f"edges={len(graph_edges)}/{total_edges}"
        )
        if self.seed_query:
            print(f"  [FAST MODE] Seed query: {self.seed_query}")
            print(f"  [FAST MODE] Query lexical matches in full set: docs={doc_matches}, nodes={node_matches}")
            if doc_matches == 0 and node_matches == 0:
                print("  [WARN] No lexical match for seed query in dataset; answer quality may be poor.")

        print("  Initializing pipelines...")
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

        print("  Indexing subset in all pipelines...")
        for route, pipeline in self.pipelines.items():
            print(f"    Indexing {route}...")
            if route in {"graph_rag", "optimized_graph_rag"}:
                pipeline.index_documents(
                    docs,
                    extra_graph_nodes=graph_nodes,
                    extra_graph_edges=graph_edges,
                )
            else:
                pipeline.index_documents(docs)

        print("  [OK] Fast-mode pipelines indexed and ready")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-start GRAG terminal interface")
    parser.add_argument("--docs", type=int, default=2000, help="Max documents to index")
    parser.add_argument("--nodes", type=int, default=5000, help="Max graph nodes to index")
    parser.add_argument("--edges", type=int, default=15000, help="Max graph edges to index")
    parser.add_argument("--quiet", action="store_true", help="Start with verbose mode disabled")
    parser.add_argument("--compare", action="store_true", help="Start with compare mode enabled")
    parser.add_argument(
        "--ask",
        action="append",
        default=[],
        help="Run one question non-interactively (can be repeated)",
    )
    parser.add_argument(
        "--seed-query",
        type=str,
        default="",
        help="Bias fast subset selection toward this query text",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_query = args.seed_query.strip() or (args.ask[0].strip() if args.ask else "")
    interface = FastStartGRAGTerminalInterface(
        verbose=not args.quiet,
        compare_mode=args.compare,
        max_docs=args.docs,
        max_graph_nodes=args.nodes,
        max_graph_edges=args.edges,
        seed_query=seed_query,
    )
    if args.ask:
        for question in args.ask:
            if question.strip():
                interface.process_question(question.strip())
        return
    interface.run()


if __name__ == "__main__":
    main()
