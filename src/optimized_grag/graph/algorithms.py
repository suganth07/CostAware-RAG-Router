from __future__ import annotations

from collections import deque

from src.optimized_grag.graph.local import LocalGraph


def connected_components(graph: LocalGraph, node_ids: set[str] | None = None) -> list[set[str]]:
    allowed = set(node_ids or graph.nodes.keys())
    seen: set[str] = set()
    components: list[set[str]] = []

    for start_id in sorted(allowed):
        if start_id in seen:
            continue
        component: set[str] = set()
        queue = deque([start_id])
        seen.add(start_id)
        while queue:
            node_id = queue.popleft()
            component.add(node_id)
            for next_id, _ in graph.neighbors(node_id):
                if next_id in allowed and next_id not in seen:
                    seen.add(next_id)
                    queue.append(next_id)
        components.append(component)

    components.sort(key=lambda component: (-len(component), min(component, default="")))
    return components


def largest_component_nodes(graph: LocalGraph, node_ids: set[str]) -> set[str]:
    components = connected_components(graph, node_ids)
    return components[0] if components else set()


def graph_density(node_count: int, edge_count: int) -> float:
    if node_count <= 1:
        return 0.0
    possible_edges = node_count * (node_count - 1)
    return edge_count / possible_edges


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction
