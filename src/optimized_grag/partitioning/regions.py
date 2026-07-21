from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from src.optimized_grag.partitioning.types import Region
from src.optimized_grag.graph import LocalGraph, connected_components


@dataclass(slots=True)
class RegionPartitionerConfig:
    max_region_nodes: int = 1000
    label_propagation_rounds: int = 6
    max_balance_ratio: float = 0.85


class RegionPartitioner:
    """Dependency-free Leiden/METIS-style approximation for bounded regions."""

    def __init__(self, config: RegionPartitionerConfig | None = None) -> None:
        self.config = config or RegionPartitionerConfig()

    def partition(self, graph: LocalGraph) -> list[Region]:
        components = connected_components(graph)
        regions: list[Region] = []
        for index, component in enumerate(components):
            root = Region(id=f"region:{index}", node_ids=component, level=0)
            regions.extend(self._recursive_split(graph, root))
        return regions

    def _recursive_split(self, graph: LocalGraph, region: Region) -> list[Region]:
        if len(region.node_ids) <= self.config.max_region_nodes:
            return [region]

        groups = self._label_propagation_split(graph, region.node_ids)
        if not self._split_is_good(groups, len(region.node_ids)):
            groups = self._balanced_split(graph, region.node_ids)

        result: list[Region] = []
        for index, group in enumerate(groups):
            child = Region(
                id=f"{region.id}.{index}",
                node_ids=group,
                level=region.level + 1,
                parent_id=region.id,
            )
            result.extend(self._recursive_split(graph, child))
        return result

    def _label_propagation_split(self, graph: LocalGraph, node_ids: set[str]) -> list[set[str]]:
        labels = {node_id: node_id for node_id in sorted(node_ids)}
        ordered_nodes = sorted(node_ids, key=lambda node_id: (-graph.degree(node_id), node_id))
        for _ in range(self.config.label_propagation_rounds):
            changed = False
            for node_id in ordered_nodes:
                neighbor_labels = Counter(
                    labels[next_id]
                    for next_id, _ in graph.neighbors(node_id)
                    if next_id in node_ids
                )
                if not neighbor_labels:
                    continue
                best_count = max(neighbor_labels.values())
                best_label = min(
                    label for label, count in neighbor_labels.items() if count == best_count
                )
                if labels[node_id] != best_label:
                    labels[node_id] = best_label
                    changed = True
            if not changed:
                break

        groups_by_label: dict[str, set[str]] = defaultdict(set)
        for node_id, label in sorted(labels.items()):
            groups_by_label[label].add(node_id)
        return [groups_by_label[label] for label in sorted(groups_by_label) if groups_by_label[label]]

    def _split_is_good(self, groups: list[set[str]], total_nodes: int) -> bool:
        if len(groups) <= 1:
            return False
        largest = max(len(group) for group in groups)
        return largest <= total_nodes * self.config.max_balance_ratio

    def _balanced_split(self, graph: LocalGraph, node_ids: set[str]) -> list[set[str]]:
        ordered = sorted(node_ids, key=lambda node_id: (-graph.degree(node_id), node_id))
        group_count = max(2, (len(ordered) + self.config.max_region_nodes - 1) // self.config.max_region_nodes)
        groups = [set() for _ in range(group_count)]
        for index, node_id in enumerate(ordered):
            groups[index % group_count].add(node_id)
        return [group for group in groups if group]
