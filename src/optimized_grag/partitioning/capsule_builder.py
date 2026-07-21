from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from src.optimized_grag.partitioning.hubs import HubProfile
from src.optimized_grag.partitioning.types import GraphCapsule, Region
from src.optimized_grag.graph import LocalGraph


@dataclass(slots=True)
class CapsuleBuilderConfig:
    max_hops: int = 2
    max_capsule_nodes: int = 300
    max_capsule_edges: int = 1000
    max_hub_neighbors: int = 20
    max_overlap_ratio: float = 0.15
    max_capsules_per_region: int = 5


class CapsuleBuilder:
    def __init__(self, config: CapsuleBuilderConfig | None = None) -> None:
        self.config = config or CapsuleBuilderConfig()

    def build_for_region(
        self,
        graph: LocalGraph,
        region: Region,
        anchors: list[str],
        hubs: HubProfile,
    ) -> list[GraphCapsule]:
        capsules = []
        for anchor_id in anchors[: self.config.max_capsules_per_region]:
            capsule = self.build(graph, region, anchor_id, hubs)
            if capsule is not None:
                capsules.append(capsule)
        return capsules

    def build(
        self,
        graph: LocalGraph,
        region: Region,
        anchor_id: str,
        hubs: HubProfile,
    ) -> GraphCapsule | None:
        if anchor_id not in graph.nodes:
            return None

        node_ids = {anchor_id}
        edge_ids: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(anchor_id, 0)])
        seen_depth = {anchor_id: 0}

        while queue and len(node_ids) < self.config.max_capsule_nodes:
            node_id, depth = queue.popleft()
            if depth >= self.config.max_hops:
                continue

            neighbors = self._ranked_neighbors(graph, node_id, region.node_ids, hubs)
            if hubs.is_hub(node_id):
                neighbors = neighbors[: self.config.max_hub_neighbors]

            for score, next_id, edge_id in neighbors:
                if len(node_ids) >= self.config.max_capsule_nodes:
                    break
                edge_ids.add(edge_id)
                if len(edge_ids) >= self.config.max_capsule_edges:
                    break
                if next_id not in node_ids:
                    node_ids.add(next_id)
                    if next_id not in seen_depth or seen_depth[next_id] > depth + 1:
                        seen_depth[next_id] = depth + 1
                        queue.append((next_id, depth + 1))

        boundary_node_ids = self._boundary_overlap(graph, node_ids, region.node_ids, hubs)
        node_ids.update(boundary_node_ids)
        # Inspect only adjacency lists for capsule nodes. Scanning every graph
        # edge once per capsule made full MetaQA construction effectively
        # O(capsules * graph_edges).
        for node_id in sorted(node_ids):
            for edge_id in graph.edge_ids(node_id):
                edge = graph.edges[edge_id]
                if edge.source_id in node_ids and edge.target_id in node_ids:
                    edge_ids.add(edge_id)
        if len(edge_ids) > self.config.max_capsule_edges:
            edge_ids = set(sorted(edge_ids)[: self.config.max_capsule_edges])

        nodes = [graph.nodes[node_id] for node_id in sorted(node_ids) if node_id in graph.nodes]
        edges = [graph.edges[edge_id] for edge_id in sorted(edge_ids) if edge_id in graph.edges]
        if not nodes:
            return None

        return GraphCapsule(
            id=f"capsule:{region.id}:{anchor_id}",
            anchor_id=anchor_id,
            region_id=region.id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            nodes=nodes,
            edges=edges,
            level=region.level,
            boundary_node_ids=boundary_node_ids,
            metadata={"partition_method": "budgeted_ego"},
        )

    def _ranked_neighbors(
        self,
        graph: LocalGraph,
        node_id: str,
        region_nodes: set[str],
        hubs: HubProfile,
    ) -> list[tuple[float, str, str]]:
        ranked = []
        for next_id, edge in graph.neighbors(node_id):
            if next_id not in region_nodes:
                continue
            edge_frequency = float(edge.properties.get("frequency") or 1.0)
            source_count = float(edge.properties.get("source_count") or 1.0)
            degree = hubs.degrees.get(next_id, graph.degree(next_id))
            hub_penalty = 0.25 if hubs.is_hub(next_id) else 1.0
            score = hub_penalty * (0.40 * edge_frequency + 0.30 * source_count + 0.30 / max(degree, 1))
            ranked.append((score, next_id, edge.id))
        ranked.sort(reverse=True)
        return ranked

    def _boundary_overlap(
        self,
        graph: LocalGraph,
        node_ids: set[str],
        region_nodes: set[str],
        hubs: HubProfile,
    ) -> set[str]:
        overlap_budget = max(1, int(len(node_ids) * self.config.max_overlap_ratio))
        candidates: dict[str, float] = {}
        for node_id in sorted(node_ids):
            for next_id, _ in graph.neighbors(node_id):
                if next_id in node_ids or next_id not in region_nodes:
                    continue
                score = graph.degree(next_id)
                if hubs.is_hub(next_id):
                    score *= 0.5
                candidates[next_id] = max(candidates.get(next_id, 0.0), score)
        ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
        return {node_id for node_id, _ in ranked[:overlap_budget]}
