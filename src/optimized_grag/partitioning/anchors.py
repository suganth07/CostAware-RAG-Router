from __future__ import annotations

from dataclasses import dataclass

from src.optimized_grag.partitioning.hubs import HubProfile
from src.optimized_grag.partitioning.types import Region
from src.optimized_grag.graph import LocalGraph


@dataclass(slots=True)
class AnchorSelectorConfig:
    anchors_per_region: int = 5


class AnchorSelector:
    def __init__(self, config: AnchorSelectorConfig | None = None) -> None:
        self.config = config or AnchorSelectorConfig()

    def select(self, graph: LocalGraph, region: Region, hubs: HubProfile) -> list[str]:
        scored = []
        for node_id in region.node_ids:
            if hubs.is_hub(node_id) and len(region.node_ids) > self.config.anchors_per_region:
                continue
            degree = graph.degree(node_id)
            doc_frequency = _document_frequency(graph.nodes[node_id].properties)
            bridge_score = _external_neighbor_count(graph, node_id, region.node_ids)
            score = 0.55 * degree + 0.25 * doc_frequency + 0.20 * bridge_score
            scored.append((score, node_id))

        if not scored:
            scored = [(graph.degree(node_id), node_id) for node_id in region.node_ids]

        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[: self.config.anchors_per_region]]


def _document_frequency(properties: dict) -> float:
    if "source_document_id" in properties or "document_id" in properties:
        return 1.0
    if "source_chunk_id" in properties:
        return 0.5
    return 0.0


def _external_neighbor_count(graph: LocalGraph, node_id: str, region_nodes: set[str]) -> int:
    return sum(1 for next_id, _ in graph.neighbors(node_id) if next_id not in region_nodes)
