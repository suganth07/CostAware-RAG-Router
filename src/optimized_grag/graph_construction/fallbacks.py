from __future__ import annotations

import re
from collections.abc import Iterable

from src.adapters.storage.graph_store import GraphEdge, GraphNode
from src.helpers import Document
from src.optimized_grag.graph import LocalGraph


def build_document_chain_graph(
    documents: Iterable[Document],
    *,
    dataset_name: str = "documents",
) -> LocalGraph:
    """Build a minimal document graph when entity extraction is too sparse."""

    graph = LocalGraph()
    previous_id: str | None = None
    for index, document in enumerate(documents):
        node_id = f"doc_{index}_{slug_id(document.id, f'document_{index}')}"
        graph.add_node(
            GraphNode(
                id=node_id,
                labels=("Document",),
                properties={
                    "dataset": dataset_name,
                    "document_id": document.id,
                    "name": document.metadata.get("title") or document.id,
                },
            )
        )
        if previous_id is not None:
            graph.add_edge(
                GraphEdge(
                    id=f"document_chain:{previous_id}:{node_id}",
                    source_id=previous_id,
                    target_id=node_id,
                    type="NEXT_DOCUMENT",
                    properties={"dataset": dataset_name},
                )
            )
        previous_id = node_id
    return graph


def slug_id(value: object, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return cleaned[:80] or fallback
