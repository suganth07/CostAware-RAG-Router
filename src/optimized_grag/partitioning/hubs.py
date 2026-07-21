from __future__ import annotations

from dataclasses import dataclass

from src.optimized_grag.graph import LocalGraph, percentile


@dataclass(slots=True)
class HubDetectorConfig:
    percentile_threshold: float = 95.0
    min_degree: int = 3


@dataclass(slots=True)
class HubProfile:
    hub_node_ids: set[str]
    threshold: float
    degrees: dict[str, int]

    def is_hub(self, node_id: str) -> bool:
        return node_id in self.hub_node_ids


class HubDetector:
    def __init__(self, config: HubDetectorConfig | None = None) -> None:
        self.config = config or HubDetectorConfig()

    def detect(self, graph: LocalGraph) -> HubProfile:
        degrees = {node_id: graph.degree(node_id) for node_id in graph.nodes}
        threshold = max(
            float(self.config.min_degree),
            percentile(list(degrees.values()), self.config.percentile_threshold),
        )
        hubs = {node_id for node_id, degree in degrees.items() if degree >= threshold}
        return HubProfile(hub_node_ids=hubs, threshold=threshold, degrees=degrees)
