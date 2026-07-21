from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from src.optimized_grag.partitioning.types import GraphCapsule


@dataclass(slots=True)
class CapsuleSupergraph:
    adjacency: dict[str, dict[str, float]] = field(default_factory=dict)

    def neighbors(self, capsule_id: str) -> dict[str, float]:
        return self.adjacency.get(capsule_id, {})


def build_capsule_supergraph(capsules: list[GraphCapsule]) -> CapsuleSupergraph:
    """Build only real capsule links via inverted membership indexes.

    This preserves the old overlap score while avoiding an O(capsules^2) scan,
    which became prohibitive after enforcing complete node coverage.
    """
    capsules_by_node: dict[str, list[str]] = defaultdict(list)
    capsules_by_source: dict[str, list[str]] = defaultdict(list)
    for capsule in capsules:
        for node_id in capsule.node_ids:
            capsules_by_node[node_id].append(capsule.id)
        for source_id in capsule.source_document_ids:
            capsules_by_source[source_id].append(capsule.id)

    pair_scores: dict[tuple[str, str], float] = defaultdict(float)
    for capsule_ids in capsules_by_node.values():
        for left_id, right_id in combinations(sorted(set(capsule_ids)), 2):
            pair_scores[(left_id, right_id)] += 1.0
    for capsule_ids in capsules_by_source.values():
        for left_id, right_id in combinations(sorted(set(capsule_ids)), 2):
            pair_scores[(left_id, right_id)] += 0.5

    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for (left_id, right_id), score in sorted(pair_scores.items()):
        adjacency[left_id][right_id] = score
        adjacency[right_id][left_id] = score
    return CapsuleSupergraph(adjacency=dict(adjacency))
