from __future__ import annotations

from dataclasses import dataclass, replace

from src.optimized_grag.partitioning.hubs import HubProfile
from src.optimized_grag.partitioning.types import CapsuleStats, GraphCapsule
from src.optimized_grag.graph import LocalGraph, graph_density, largest_component_nodes


@dataclass(slots=True)
class CapsuleValidatorConfig:
    max_capsule_nodes: int = 300
    max_capsule_edges: int = 1000
    max_hub_ratio: float = 0.20
    max_overlap_ratio: float = 0.15
    min_connected_ratio: float = 0.85


class CapsuleValidator:
    def __init__(self, config: CapsuleValidatorConfig | None = None) -> None:
        self.config = config or CapsuleValidatorConfig()

    def repair(self, graph: LocalGraph, capsule: GraphCapsule, hubs: HubProfile) -> GraphCapsule | None:
        node_ids = set(capsule.node_ids)
        edge_ids = set(capsule.edge_ids)

        if len(node_ids) > self.config.max_capsule_nodes:
            ordered = sorted(node_ids, key=lambda node_id: (-graph.degree(node_id), node_id))
            keep = set(ordered[: self.config.max_capsule_nodes])
            if capsule.anchor_id in node_ids:
                keep.add(capsule.anchor_id)
            node_ids = keep

        edge_ids = {
            edge_id
            for edge_id in edge_ids
            if edge_id in graph.edges
            and graph.edges[edge_id].source_id in node_ids
            and graph.edges[edge_id].target_id in node_ids
        }
        if len(edge_ids) > self.config.max_capsule_edges:
            edge_ids = set(sorted(edge_ids)[: self.config.max_capsule_edges])

        largest = largest_component_nodes(graph, node_ids)
        connected_ratio = len(largest) / max(len(node_ids), 1)
        if largest and connected_ratio < self.config.min_connected_ratio:
            node_ids = largest
            edge_ids = {
                edge_id
                for edge_id in edge_ids
                if graph.edges[edge_id].source_id in node_ids and graph.edges[edge_id].target_id in node_ids
            }

        if not node_ids:
            return None

        hub_count = sum(1 for node_id in node_ids if hubs.is_hub(node_id))
        boundary_node_ids = set(capsule.boundary_node_ids) & node_ids
        stats = CapsuleStats(
            node_count=len(node_ids),
            edge_count=len(edge_ids),
            hub_count=hub_count,
            density=graph_density(len(node_ids), len(edge_ids)),
            connected_component_ratio=connected_ratio,
            overlap_ratio=len(boundary_node_ids) / max(len(node_ids), 1),
        )
        if stats.hub_count / max(stats.node_count, 1) > self.config.max_hub_ratio:
            non_hubs = {node_id for node_id in node_ids if not hubs.is_hub(node_id)}
            if non_hubs:
                node_ids = non_hubs | ({capsule.anchor_id} if capsule.anchor_id in node_ids else set())

        nodes = [graph.nodes[node_id] for node_id in sorted(node_ids) if node_id in graph.nodes]
        edges = [graph.edges[edge_id] for edge_id in sorted(edge_ids) if edge_id in graph.edges]
        if not nodes:
            return None

        return replace(
            capsule,
            node_ids=node_ids,
            edge_ids=edge_ids,
            nodes=nodes,
            edges=edges,
            boundary_node_ids=boundary_node_ids,
            relation_types={edge.type for edge in edges},
            stats=stats,
        )
