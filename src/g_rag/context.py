from __future__ import annotations

from src.adapters.storage.graph_store import GraphNeighborhood
from src.adapters.storage.vector_store import VectorSearchResult


class ContextBuilder:
    def __init__(self, *, max_chars: int = 12_000) -> None:
        self.max_chars = max_chars

    def build(
        self,
        *,
        vector_results: list[VectorSearchResult],
        graph_neighborhood: GraphNeighborhood,
        graph_edge_scores: dict[str, float] | None = None,
        graph_edge_depths: dict[str, int] | None = None,
    ) -> str:
        lines: list[str] = []
        used_chars = 0

        def append(line: str) -> bool:
            nonlocal used_chars
            addition = len(line) + (1 if lines else 0)
            if used_chars + addition > self.max_chars:
                return False
            lines.append(line)
            used_chars += addition
            return True

        if vector_results:
            append("[Ranked source evidence]")
            for index, result in enumerate(vector_results, start=1):
                doc = result.document
                source = doc.metadata.get("document_id") or doc.metadata.get("source") or doc.id
                if not append(
                    f"{index}. score={result.score:.4f} source={source} {doc.text}"
                ):
                    break

        if graph_neighborhood.edges:
            if lines:
                append("")
            append("[Ranked graph evidence]")
            nodes = {node.id: node for node in graph_neighborhood.nodes}
            edge_scores = graph_edge_scores or {}
            edge_depths = graph_edge_depths or {}
            for index, edge in enumerate(graph_neighborhood.edges, start=1):
                source = nodes.get(edge.source_id)
                target = nodes.get(edge.target_id)
                source_name = (
                    source.properties.get("name") if source else edge.source_id
                )
                target_name = (
                    target.properties.get("name") if target else edge.target_id
                )
                score = edge_scores.get(edge.id, 0.0)
                depth = edge_depths.get(edge.id)
                depth_text = f" depth={depth}" if depth is not None else ""
                if not append(
                    f"{index}. score={score:.4f}{depth_text} "
                    f"{source_name} -[{edge.type}]-> {target_name}"
                ):
                    break

        return "\n".join(lines)
