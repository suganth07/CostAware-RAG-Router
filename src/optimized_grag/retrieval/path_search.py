from __future__ import annotations

import gc
from collections import deque

from src.adapters.storage.graph_store import GraphEdge, GraphNode
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.partitioning.types import GraphCapsule
from src.optimized_grag.retrieval.types import EvidencePath


class BoundedLocalPathSearcher:
    def __init__(
        self,
        *,
        max_merged_nodes: int = 1200,
        max_candidate_paths: int = 200,
        max_path_length: int = 3,
        final_paths: int = 12,
        max_seed_neighbors: int = 100,
        seed_expansion_depth: int = 1,
        max_seed_expansion_nodes: int = 1200,
        direct_seed_path_search: bool = True,
    ) -> None:
        self.max_merged_nodes = max_merged_nodes
        self.max_candidate_paths = max_candidate_paths
        self.max_path_length = max_path_length
        self.final_paths = final_paths
        self.max_seed_neighbors = max_seed_neighbors
        self.seed_expansion_depth = max(0, int(seed_expansion_depth))
        self.max_seed_expansion_nodes = max(1, int(max_seed_expansion_nodes))
        self.direct_seed_path_search = bool(direct_seed_path_search)
        self.last_search_mode = "uninitialized"
        self._node_terms_cache: dict[tuple[str, str, str], frozenset[str]] = {}
        self._edge_terms_cache: dict[tuple[str, str], frozenset[str]] = {}

    def prepare_graph(self, graph: LocalGraph) -> None:
        """Precompute immutable lexical features outside query latency."""
        for node in graph.nodes.values():
            self._node_terms(node)
        for edge in graph.edges.values():
            self._edge_terms(edge)
        # The feature dictionaries are long-lived and acyclic. Promote them
        # before serving queries so a generation-2 collection cannot create a
        # one-off tail-latency spike midway through an evaluation/request batch.
        gc.collect()

    def search(
        self,
        *,
        capsules: list[GraphCapsule],
        seed_node_ids: list[str],
        query_terms: set[str],
        seed_graph: LocalGraph | None = None,
    ) -> list[EvidencePath]:
        relation_query_terms = {_relation_stem(token) for token in query_terms}
        ranked_neighbors_cache: dict[str, list[tuple[str, GraphEdge]]] = {}
        exact_seeds = [
            node_id
            for node_id in dict.fromkeys(seed_node_ids)
            if seed_graph is not None and node_id in seed_graph.nodes
        ]
        if self.direct_seed_path_search and seed_graph is not None and exact_seeds:
            # Query-linked nodes are exact identifiers, so walking the indexed
            # graph directly is both more complete and substantially cheaper
            # than copying capsules, materializing a large seed neighborhood,
            # and then walking that copy. Capsules still route source evidence
            # and remain the fallback when entity linking yields no exact seed.
            graph = seed_graph
            seeds = exact_seeds
            self.last_search_mode = "direct_exact_seed"
        else:
            graph = self._merge_capsules(capsules)
            self.last_search_mode = "capsule_merge_fallback"
        if seed_graph is not None and self.last_search_mode != "direct_exact_seed":
            self._add_seed_neighborhoods(
                graph=graph,
                source=seed_graph,
                seed_node_ids=seed_node_ids,
                query_terms=query_terms,
                relation_query_terms=relation_query_terms,
                ranked_neighbors_cache=ranked_neighbors_cache,
            )
        if not graph.nodes:
            return []

        seeds = [node_id for node_id in seed_node_ids if node_id in graph.nodes]
        if not seeds:
            seeds = [capsule.anchor_id for capsule in capsules if capsule.anchor_id in graph.nodes]
        if not seeds:
            seeds = list(graph.nodes.keys())[:3]

        seeds = list(dict.fromkeys(seeds))
        if len(seeds) > self.max_candidate_paths:
            seeds = seeds[: self.max_candidate_paths]

        paths: list[EvidencePath] = []
        base_budget, remainder = divmod(self.max_candidate_paths, max(len(seeds), 1))
        for index, seed_id in enumerate(seeds):
            seed_budget = base_budget + (1 if index < remainder else 0)
            paths.extend(
                self._walk_from_seed(
                    graph,
                    seed_id,
                    query_terms,
                    relation_query_terms,
                    max_paths=seed_budget,
                    ranked_neighbors_cache=ranked_neighbors_cache,
                    relation_only_neighbors=(
                        self.last_search_mode == "direct_exact_seed"
                    ),
                )
            )

        paths.sort(key=lambda path: path.score, reverse=True)
        return _select_paths_with_seed_coverage(
            paths,
            seeds,
            self.final_paths,
            max_depth=self.max_path_length,
        )

    def _merge_capsules(self, capsules: list[GraphCapsule]) -> LocalGraph:
        graph = LocalGraph()
        ordered_nodes: list[GraphNode] = []
        ordered_edges: list[GraphEdge] = []
        seen_nodes: set[str] = set()
        for capsule in capsules:
            for node in capsule.nodes:
                if node.id not in seen_nodes and len(seen_nodes) < self.max_merged_nodes:
                    ordered_nodes.append(node)
                    seen_nodes.add(node.id)
            for edge in capsule.edges:
                if edge.source_id in seen_nodes and edge.target_id in seen_nodes:
                    ordered_edges.append(edge)
        graph.add_nodes(ordered_nodes)
        graph.add_edges(ordered_edges)
        return graph

    def _add_seed_neighborhoods(
        self,
        *,
        graph: LocalGraph,
        source: LocalGraph,
        seed_node_ids: list[str],
        query_terms: set[str],
        relation_query_terms: set[str],
        ranked_neighbors_cache: dict[str, list[tuple[str, GraphEdge]]],
    ) -> None:
        """Add a bounded, relation-aware neighborhood around every linked seed."""
        unique_seeds = list(dict.fromkeys(seed_node_ids))
        if not unique_seeds or self.seed_expansion_depth <= 0:
            return
        base_budget, remainder = divmod(
            self.max_seed_expansion_nodes,
            len(unique_seeds),
        )
        for seed_index, seed_id in enumerate(unique_seeds):
            seed = source.nodes.get(seed_id)
            if seed is None:
                continue
            graph.add_node(seed)
            seed_budget = base_budget + (1 if seed_index < remainder else 0)
            discovered = 1
            seen = {seed_id}
            queue: deque[tuple[str, int]] = deque([(seed_id, 0)])
            while queue and discovered < seed_budget:
                node_id, depth = queue.popleft()
                if depth >= self.seed_expansion_depth:
                    continue
                neighbors = self._ranked_neighbors(
                    source,
                    node_id,
                    query_terms,
                    relation_query_terms,
                    ranked_neighbors_cache=ranked_neighbors_cache,
                )
                for next_id, edge in neighbors:
                    if discovered >= seed_budget:
                        break
                    graph.add_node(source.nodes[next_id])
                    graph.add_edge(edge)
                    if next_id not in seen:
                        seen.add(next_id)
                        discovered += 1
                        queue.append((next_id, depth + 1))

    def _walk_from_seed(
        self,
        graph: LocalGraph,
        seed_id: str,
        query_terms: set[str],
        relation_query_terms: set[str],
        *,
        max_paths: int,
        ranked_neighbors_cache: dict[str, list[tuple[str, GraphEdge]]],
        relation_only_neighbors: bool = False,
    ) -> list[EvidencePath]:
        if max_paths <= 0 or self.max_path_length <= 0:
            return []
        paths: list[EvidencePath] = []
        seed_overlap = self._node_query_overlap(graph.nodes[seed_id], query_terms)
        frontier: list[
            tuple[str, list[str], list[str], float, float]
        ] = [(seed_id, [seed_id], [], seed_overlap, 0.0)]
        base_budget, remainder = divmod(max_paths, self.max_path_length)

        for depth in range(1, self.max_path_length + 1):
            depth_budget = base_budget + (1 if depth <= remainder else 0)
            expanded: list[
                tuple[float, tuple[str, ...], str, list[str], list[str], float, float]
            ] = []
            signatures: set[tuple[str, ...]] = set()
            for node_id, node_path, edge_path, node_overlap, relation_overlap in frontier:
                ranked_neighbors = self._ranked_neighbors(
                    graph,
                    node_id,
                    query_terms,
                    relation_query_terms,
                    ranked_neighbors_cache=ranked_neighbors_cache,
                    relation_only=relation_only_neighbors,
                )
                for next_id, edge in ranked_neighbors:
                    if next_id in node_path:
                        continue
                    next_nodes = [*node_path, next_id]
                    next_edges = [*edge_path, edge.id]
                    signature = tuple(next_edges)
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    next_node_overlap = node_overlap + self._node_query_overlap(
                        graph.nodes[next_id], query_terms
                    )
                    next_relation_overlap = relation_overlap + self._edge_query_overlap(
                        edge,
                        query_terms,
                        relation_query_terms=relation_query_terms,
                    )
                    score = (
                        0.35 * next_node_overlap
                        + 0.50 * next_relation_overlap
                        + 0.15 * min(1.0, depth / max(self.max_path_length, 1))
                    )
                    expanded.append(
                        (
                            score,
                            signature,
                            next_id,
                            next_nodes,
                            next_edges,
                            next_node_overlap,
                            next_relation_overlap,
                        )
                    )
            if not expanded:
                break
            expanded.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )
            beam_width = max(depth_budget, min(max_paths, 256))
            retained = expanded[:beam_width]
            paths.extend(
                self._to_evidence_path(
                    graph,
                    item[3],
                    item[4],
                    query_terms,
                    seed_id=seed_id,
                    score_override=item[0],
                )
                for item in retained[:depth_budget]
            )
            frontier = [
                (item[2], item[3], item[4], item[5], item[6])
                for item in retained
            ]

        return paths[:max_paths]

    def _to_evidence_path(
        self,
        graph: LocalGraph,
        node_path: list[str],
        edge_path: list[str],
        query_terms: set[str],
        *,
        seed_id: str,
        score_override: float | None = None,
    ) -> EvidencePath:
        nodes = [graph.nodes[node_id] for node_id in node_path if node_id in graph.nodes]
        edges = [graph.edges[edge_id] for edge_id in edge_path if edge_id in graph.edges]
        overlap = sum(self._node_query_overlap(node, query_terms) for node in nodes)
        relation_overlap = sum(
            self._edge_query_overlap(edge, query_terms) for edge in edges
        )
        depth_fit = min(1.0, len(edges) / max(self.max_path_length, 1))
        score = score_override if score_override is not None else (
            0.35 * overlap + 0.50 * relation_overlap + 0.15 * depth_fit
        )
        return EvidencePath(
            nodes=nodes,
            edges=edges,
            score=score,
            text=_path_text(nodes, edges),
            metadata={"path_length": len(edges), "seed_node_id": seed_id},
        )

    def _node_query_overlap(self, node: GraphNode, query_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        haystack = self._node_terms(node)
        return len(haystack & query_terms) / max(len(query_terms), 1)

    def _node_terms(self, node: GraphNode) -> frozenset[str]:
        name = str(node.properties.get("name") or "")
        text = str(node.properties.get("text") or "")
        cache_key = (node.id, name, text[:1000])
        haystack = self._node_terms_cache.get(cache_key)
        if haystack is None:
            values = {node.id.lower()}
            if name:
                values.update(name.lower().split())
            if text:
                values.update(text.lower().split()[:120])
            haystack = frozenset(values)
            self._node_terms_cache[cache_key] = haystack
        return haystack

    def _edge_query_overlap(
        self,
        edge: GraphEdge,
        query_terms: set[str],
        *,
        relation_query_terms: set[str] | None = None,
    ) -> float:
        if not query_terms:
            return 0.0
        normalized_query_terms = relation_query_terms or {
            _relation_stem(token) for token in query_terms
        }
        edge_terms = self._edge_terms(edge)
        return len(edge_terms & normalized_query_terms) / max(len(query_terms), 1)

    def _edge_terms(self, edge: GraphEdge) -> frozenset[str]:
        relation = str(edge.properties.get("relation") or "")
        # Relation vocabulary is tiny compared with edge count (MetaQA has
        # 134k edges but only a handful of relation labels). Cache the parsed
        # label once per relation instead of once per edge.
        cache_key = (str(edge.type), relation)
        edge_terms = self._edge_terms_cache.get(cache_key)
        if edge_terms is None:
            edge_terms = frozenset(
                _relation_stem(token.lower())
                for value in (edge.type, relation)
                if value
                for token in str(value).replace("_", " ").split()
            )
            self._edge_terms_cache[cache_key] = edge_terms
        return edge_terms

    def _ranked_neighbors(
        self,
        graph: LocalGraph,
        node_id: str,
        query_terms: set[str],
        relation_query_terms: set[str],
        *,
        ranked_neighbors_cache: dict[str, list[tuple[str, GraphEdge]]] | None = None,
        relation_only: bool = False,
    ) -> list[tuple[str, GraphEdge]]:
        if ranked_neighbors_cache is not None:
            cached = ranked_neighbors_cache.get(node_id)
            if cached is not None:
                return cached
        neighbors = graph.neighbors(node_id)
        # Apply the hard fan-out bound before computing node/query overlap and
        # sorting. Relation-matching edges are the useful pool for multi-hop QA;
        # deterministic non-matches only fill unused capacity. This prevents a
        # hub with tens of thousands of edges from dominating one query.
        if len(neighbors) > self.max_seed_neighbors:
            relation_matches: list[tuple[str, GraphEdge]] = []
            fallback: list[tuple[str, GraphEdge]] = []
            for item in neighbors:
                if self._edge_query_overlap(
                    item[1],
                    query_terms,
                    relation_query_terms=relation_query_terms,
                ) > 0.0:
                    if len(relation_matches) < self.max_seed_neighbors:
                        relation_matches.append(item)
                elif len(fallback) < self.max_seed_neighbors:
                    fallback.append(item)
            neighbors = [
                *relation_matches,
                *fallback[: max(0, self.max_seed_neighbors - len(relation_matches))],
            ]
        if relation_only:
            ranked = sorted(
                neighbors,
                key=lambda item: (
                    -self._edge_query_overlap(
                        item[1],
                        query_terms,
                        relation_query_terms=relation_query_terms,
                    ),
                    item[0],
                    item[1].id,
                ),
            )[: self.max_seed_neighbors]
        else:
            ranked = sorted(
                neighbors,
                key=lambda item: (
                    -self._edge_query_overlap(
                        item[1],
                        query_terms,
                        relation_query_terms=relation_query_terms,
                    ),
                    -self._node_query_overlap(graph.nodes[item[0]], query_terms),
                    item[0],
                    item[1].id,
                ),
            )[: self.max_seed_neighbors]
        if ranked_neighbors_cache is not None:
            ranked_neighbors_cache[node_id] = ranked
        return ranked


def _path_text(nodes: list[GraphNode], edges: list[GraphEdge]) -> str:
    if not nodes:
        return ""
    parts = [str(nodes[0].properties.get("name") or nodes[0].id)]
    for edge, node in zip(edges, nodes[1:]):
        parts.append(f"-[{edge.type}]->")
        parts.append(str(node.properties.get("name") or node.id))
    return " ".join(parts)


def _relation_stem(token: str) -> str:
    """Tiny deterministic stemmer for relation words used in routing only."""
    value = token.lower()
    for suffix in ("ing", "ors", "ers", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[: -len(suffix)]
    return value


def _select_paths_with_seed_coverage(
    paths: list[EvidencePath],
    seed_node_ids: list[str],
    limit: int,
    *,
    max_depth: int = 1,
) -> list[EvidencePath]:
    if limit <= 0:
        return []
    selected: list[EvidencePath] = []
    signatures: set[tuple[str, ...]] = set()
    for seed_id in seed_node_ids:
        match = next(
            (
                path
                for path in paths
                if path.metadata.get("seed_node_id") == seed_id
                and tuple(edge.id for edge in path.edges) not in signatures
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            signatures.add(tuple(edge.id for edge in match.edges))
        if len(selected) >= limit:
            return selected
    # Runtime does not know the gold hop count. Preserve evidence from every
    # searched depth, then let relation/query scores fill the remaining slots.
    depth_count = max(1, max_depth)
    base_depth_quota, depth_remainder = divmod(limit, depth_count)
    for depth in range(1, depth_count + 1):
        depth_quota = base_depth_quota + (1 if depth <= depth_remainder else 0)
        already_selected = sum(len(path.edges) == depth for path in selected)
        for path in paths:
            if already_selected >= depth_quota or len(selected) >= limit:
                break
            if len(path.edges) != depth:
                continue
            signature = tuple(edge.id for edge in path.edges)
            if signature in signatures:
                continue
            selected.append(path)
            signatures.add(signature)
            already_selected += 1
    for path in paths:
        signature = tuple(edge.id for edge in path.edges)
        if signature in signatures:
            continue
        selected.append(path)
        signatures.add(signature)
        if len(selected) >= limit:
            break
    return selected
