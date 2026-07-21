from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass

from src.adapters.storage.graph_store import GraphEdge, GraphNeighborhood, GraphNode
from src.adapters.storage.vector_store import VectorSearchResult


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}
_CANONICAL = {
    "act": "actor",
    "acted": "actor",
    "actor": "actor",
    "actors": "actor",
    "appear": "actor",
    "appeared": "actor",
    "star": "actor",
    "starred": "actor",
    "stars": "actor",
    "direct": "director",
    "directed": "director",
    "director": "director",
    "directors": "director",
    "genre": "genre",
    "genres": "genre",
    "type": "genre",
    "types": "genre",
    "language": "language",
    "languages": "language",
    "release": "year",
    "released": "year",
    "year": "year",
    "years": "year",
    "screenwriter": "writer",
    "wrote": "writer",
    "written": "writer",
    "writer": "writer",
    "writers": "writer",
    "tag": "tag",
    "tags": "tag",
    "rating": "rating",
    "ratings": "rating",
    "vote": "votes",
    "votes": "votes",
    "film": "movie",
    "films": "movie",
    "movie": "movie",
    "movies": "movie",
}


@dataclass(slots=True)
class RankedGraphEvidence:
    neighborhood: GraphNeighborhood
    edge_scores: dict[str, float]
    edge_depths: dict[str, int]


class ProductionEvidenceRanker:
    """Fuse dense, lexical, entity, relation, and graph-depth evidence."""

    def rank_vectors(
        self,
        *,
        query: str,
        results: list[VectorSearchResult],
        limit: int,
    ) -> list[VectorSearchResult]:
        query_terms = _terms(query)
        rescored: list[VectorSearchResult] = []
        for result in results:
            document_terms = _terms(result.document.text)
            lexical = _cosine_overlap(query_terms, document_terms)
            dense = max(0.0, min(1.0, (float(result.score) + 1.0) / 2.0))
            entity_exact = float(
                any(
                    _terms(str(value)) <= query_terms
                    for key, value in result.document.metadata.items()
                    if key in {"subject", "object", "title"} and value
                )
            )
            score = 0.72 * dense + 0.23 * lexical + 0.05 * entity_exact
            rescored.append(
                VectorSearchResult(document=result.document, score=score)
            )
        rescored.sort(key=lambda value: (-value.score, value.document.id))
        return rescored[: max(0, int(limit))]

    def rank_graph(
        self,
        *,
        query: str,
        neighborhood: GraphNeighborhood,
        seed_node_ids: list[str],
        vector_results: list[VectorSearchResult],
        max_depth: int,
        limit: int,
    ) -> RankedGraphEvidence:
        if limit <= 0 or not neighborhood.edges:
            return RankedGraphEvidence(
                neighborhood=GraphNeighborhood(
                    seed_ids=neighborhood.seed_ids,
                    nodes=list(neighborhood.nodes),
                    edges=[],
                ),
                edge_scores={},
                edge_depths={},
            )
        nodes = {node.id: node for node in neighborhood.nodes}
        query_terms = _terms(query)
        query_relations = _canonical_terms(query)
        seed_ids = set(seed_node_ids)
        vector_support = _vector_entity_support(vector_results)
        edge_depths = _edge_depths(
            neighborhood.edges,
            seed_ids,
            max_depth=max_depth,
        )

        # Collapse duplicate facts before ranking. Production graph stores can
        # contain repeated provenance edges for the same semantic triple.
        best_by_fact: dict[tuple[str, str, str], tuple[GraphEdge, float, int]] = {}
        for order, edge in enumerate(neighborhood.edges):
            source = nodes.get(edge.source_id, GraphNode(id=edge.source_id))
            target = nodes.get(edge.target_id, GraphNode(id=edge.target_id))
            relation_terms = _canonical_terms(
                f"{edge.type} {edge.properties.get('relation') or ''}"
            )
            endpoint_terms = _terms(
                f"{_node_name(source)} {_node_name(target)}"
            )
            relation_score = _cosine_overlap(query_relations, relation_terms)
            endpoint_score = _cosine_overlap(query_terms, endpoint_terms)
            seed_score = float(
                edge.source_id in seed_ids or edge.target_id in seed_ids
            )
            support_score = max(
                vector_support.get(edge.source_id, 0.0),
                vector_support.get(edge.target_id, 0.0),
            )
            depth = edge_depths.get(edge.id, max_depth + 1)
            depth_score = 1.0 / max(depth, 1)
            stable_order_score = 1.0 / (order + 1)
            score = (
                0.38 * relation_score
                + 0.20 * endpoint_score
                + 0.20 * seed_score
                + 0.12 * support_score
                + 0.07 * depth_score
                + 0.03 * stable_order_score
            )
            fact = (edge.source_id, edge.type, edge.target_id)
            current = best_by_fact.get(fact)
            if current is None or (score, edge.id) > (current[1], current[0].id):
                best_by_fact[fact] = (edge, score, depth)

        ranked = sorted(
            best_by_fact.values(),
            key=lambda value: (-value[1], value[2], value[0].id),
        )
        selected = _select_depth_balanced(
            ranked,
            limit=min(limit, len(ranked)),
            max_depth=max_depth,
        )
        selected_edges = [value[0] for value in selected]
        selected_node_ids = set(seed_ids)
        for edge in selected_edges:
            selected_node_ids.add(edge.source_id)
            selected_node_ids.add(edge.target_id)
        selected_nodes = [
            node for node in neighborhood.nodes if node.id in selected_node_ids
        ]
        return RankedGraphEvidence(
            neighborhood=GraphNeighborhood(
                seed_ids=neighborhood.seed_ids,
                nodes=selected_nodes,
                edges=selected_edges,
            ),
            edge_scores={value[0].id: value[1] for value in selected},
            edge_depths={value[0].id: value[2] for value in selected},
        )


def _select_depth_balanced(
    ranked: list[tuple[GraphEdge, float, int]],
    *,
    limit: int,
    max_depth: int,
) -> list[tuple[GraphEdge, float, int]]:
    if limit <= 0:
        return []
    depth_count = max(1, int(max_depth))
    quota, remainder = divmod(limit, depth_count)
    buckets: dict[int, list[tuple[GraphEdge, float, int]]] = {}
    for depth in range(1, depth_count + 1):
        depth_quota = quota + (1 if depth <= remainder else 0)
        buckets[depth] = [value for value in ranked if value[2] == depth][
            :depth_quota
        ]
    # Interleave depths so an LLM with a short evidence window sees candidate
    # endpoints from every allowed reasoning depth near the top.
    selected: list[tuple[GraphEdge, float, int]] = []
    selected_ids: set[str] = set()
    bucket_position = 0
    while len(selected) < limit:
        added = False
        for depth in range(1, depth_count + 1):
            bucket = buckets[depth]
            if bucket_position >= len(bucket):
                continue
            value = bucket[bucket_position]
            if value[0].id not in selected_ids:
                selected.append(value)
                selected_ids.add(value[0].id)
                added = True
        if not added:
            break
        bucket_position += 1
    for value in ranked:
        if len(selected) >= limit:
            break
        if value[0].id in selected_ids:
            continue
        selected.append(value)
        selected_ids.add(value[0].id)
    return selected


def _edge_depths(
    edges: list[GraphEdge],
    seed_ids: set[str],
    *,
    max_depth: int,
) -> dict[str, int]:
    adjacency: dict[str, list[tuple[str, GraphEdge]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_id, []).append((edge.target_id, edge))
        adjacency.setdefault(edge.target_id, []).append((edge.source_id, edge))
    node_depth = {seed_id: 0 for seed_id in seed_ids}
    queue = deque(seed_ids)
    result: dict[str, int] = {}
    while queue:
        node_id = queue.popleft()
        depth = node_depth[node_id]
        if depth >= max_depth:
            continue
        for next_id, edge in adjacency.get(node_id, []):
            edge_depth = depth + 1
            previous = result.get(edge.id)
            if previous is None or edge_depth < previous:
                result[edge.id] = edge_depth
            if next_id not in node_depth:
                node_depth[next_id] = edge_depth
                queue.append(next_id)
    return result


def _vector_entity_support(
    results: list[VectorSearchResult],
) -> dict[str, float]:
    support: dict[str, float] = {}
    for rank, result in enumerate(results, start=1):
        weight = 1.0 / rank
        for entity_id in result.document.metadata.get("entity_ids") or []:
            key = str(entity_id)
            support[key] = max(support.get(key, 0.0), weight)
    return support


def _node_name(node: GraphNode) -> str:
    return str(node.properties.get("name") or node.id)


def _terms(value: str) -> set[str]:
    return {
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(value))
        if token not in _STOPWORDS
    }


def _canonical_terms(value: str) -> set[str]:
    return {_CANONICAL.get(token, token) for token in _terms(value)}


def _cosine_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))
