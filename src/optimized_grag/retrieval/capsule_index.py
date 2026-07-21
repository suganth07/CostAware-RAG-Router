from __future__ import annotations

import math
import re
from heapq import heappop, heappush
from dataclasses import dataclass
from typing import Any

from src.adapters.embedding import EmbeddingAdapter
from src.optimized_grag.models.structural import StructuralEncoder
from src.optimized_grag.partitioning.supergraph import CapsuleSupergraph
from src.optimized_grag.partitioning.types import CapsuleCandidate, GraphCapsule

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class IndexedCapsule:
    capsule: GraphCapsule
    text_vector: list[float]
    structural_vector: list[float]
    entity_tokens: set[str]


class CapsuleIndex:
    """In-memory capsule index for cheap first-stage retrieval and reranking."""

    def __init__(self, *, structural_encoder: StructuralEncoder) -> None:
        self.structural_encoder = structural_encoder
        self._items: dict[str, IndexedCapsule] = {}
        self._ordered_ids: list[str] = []
        self._position_by_id: dict[str, int] = {}
        self._retrieval_matrix: Any = None
        self._entity_positions: dict[str, list[int]] = {}
        self._node_positions: dict[str, list[int]] = {}
        self._region_positions: dict[str, list[int]] = {}
        self._source_scores: Any = None
        self.retrieval_mode = "semantic_text"
        self.supergraph = CapsuleSupergraph()

    def build(
        self,
        *,
        capsules: list[GraphCapsule],
        embedding_adapter: EmbeddingAdapter,
        supergraph: CapsuleSupergraph,
    ) -> None:
        self.supergraph = supergraph
        structural_vectors = [
            self.structural_encoder.encode_capsule(capsule) for capsule in capsules
        ]
        encode_routing_capsule = getattr(
            self.structural_encoder, "encode_routing_capsule", None
        )
        routing_vectors = (
            [encode_routing_capsule(capsule) for capsule in capsules]
            if callable(encode_routing_capsule)
            else structural_vectors
        )
        # Once the runtime capsules are bound to a matching R-GCN + query
        # projector, that learned space is already the intended first-stage
        # retrieval space. Re-embedding every capsule summary is redundant and
        # made a fresh MetaQA index issue hundreds of local-model requests.
        use_trained_space = bool(
            getattr(self.structural_encoder, "trained_space_enabled", False)
        )
        if use_trained_space:
            vectors = routing_vectors
            routing_kind = str(
                getattr(self.structural_encoder, "routing_space_kind", "structural_rgcn")
            )
            self.retrieval_mode = (
                "trained_semantic_projection"
                if routing_kind == "semantic_capsule_text"
                else "trained_structural"
            )
        else:
            summaries = [capsule.summary for capsule in capsules]
            vectors = embedding_adapter.embed_texts(summaries).vectors if summaries else []
            self.retrieval_mode = "semantic_text"

        self._items = {}
        self._ordered_ids = []
        self._position_by_id = {}
        self._entity_positions = {}
        self._node_positions = {}
        self._region_positions = {}
        source_score_values: list[float] = []
        for position, (capsule, vector, structural_vector) in enumerate(
            zip(capsules, vectors, structural_vectors)
        ):
            entity_tokens = _capsule_entity_tokens(capsule)
            self._items[capsule.id] = IndexedCapsule(
                capsule=capsule,
                text_vector=vector,
                structural_vector=structural_vector,
                entity_tokens=entity_tokens,
            )
            self._ordered_ids.append(capsule.id)
            self._position_by_id[capsule.id] = position
            source_score_values.append(
                min(1.0, len(capsule.source_document_ids) / 5.0)
            )
            for token in entity_tokens:
                self._entity_positions.setdefault(token, []).append(position)
            for node_id in capsule.node_ids:
                self._node_positions.setdefault(node_id, []).append(position)
            self._region_positions.setdefault(capsule.region_id, []).append(position)
        self._retrieval_matrix = _normalized_matrix(vectors)
        self._source_scores = _float_array(source_score_values)

    def candidates(
        self,
        *,
        query: str,
        query_vector: list[float],
        query_entities: list[str],
        seed_node_ids: list[str] | None = None,
        limit: int,
    ) -> list[CapsuleCandidate]:
        query_tokens = set(_tokenize(query)) | {_normalize_token(entity) for entity in query_entities}
        exact_seed_ids = set(seed_node_ids or [])
        if not self._ordered_ids or limit <= 0:
            return []

        retrieval_query = (
            self.structural_encoder.project_query(query_vector)
            if self.retrieval_mode.startswith("trained_")
            else query_vector
        )
        text_scores = _matrix_cosine_scores(self._retrieval_matrix, retrieval_query)
        entity_matches = _zero_array(len(self._ordered_ids))
        for token in query_tokens:
            _increment_positions(entity_matches, self._entity_positions.get(token, ()))

        matched_seeds: dict[int, set[str]] = {}
        for seed_id in exact_seed_ids:
            for position in self._node_positions.get(seed_id, ()):
                matched_seeds.setdefault(position, set()).add(seed_id)

        entity_denominator = max(len(query_tokens), 1)
        entity_scores = _divide_array(entity_matches, entity_denominator)
        for position in matched_seeds:
            entity_scores[position] = max(float(entity_scores[position]), 1.0)
        source_scores = self._source_scores
        scores = _weighted_scores(text_scores, entity_scores, source_scores)

        # Sorting 10k--50k scalar scores is cheap. The expensive operation was
        # doing every 768-dimensional cosine in Python; the matrix multiply
        # above performs it in optimized native code and keeps deterministic
        # ID tie-breaking.
        ranked_positions = sorted(
            _top_position_pool(scores, matched_seeds, limit),
            key=lambda position: (
                len(matched_seeds.get(position, ())),
                float(scores[position]),
                self._ordered_ids[position],
            ),
            reverse=True,
        )

        selected_positions: list[int] = []
        selected_position_set: set[int] = set()
        for seed_id in dict.fromkeys(seed_node_ids or []):
            match = next(
                (
                    position
                    for position in ranked_positions
                    if seed_id in matched_seeds.get(position, ())
                    and position not in selected_position_set
                ),
                None,
            )
            if match is not None:
                selected_positions.append(match)
                selected_position_set.add(match)
            if len(selected_positions) >= limit:
                break
        for position in ranked_positions:
            if len(selected_positions) >= limit:
                break
            if position not in selected_position_set:
                selected_positions.append(position)
                selected_position_set.add(position)

        candidates = []
        for position in selected_positions:
            item = self._items[self._ordered_ids[position]]
            matched_seed_ids = sorted(matched_seeds.get(position, ()))
            exact_seed_match = bool(matched_seed_ids)
            candidates.append(
                CapsuleCandidate(
                    capsule=item.capsule,
                    score=float(scores[position]),
                    text_score=float(text_scores[position]),
                    entity_score=float(entity_scores[position]),
                    source_score=float(source_scores[position]),
                    metadata={
                        "stage": "first_stage",
                        "retrieval_mode": self.retrieval_mode,
                        "exact_seed_match": exact_seed_match,
                        "matched_seed_ids": matched_seed_ids,
                    },
                )
            )

        return _take_with_seed_coverage(candidates, seed_node_ids or [], limit)

    def structural_vector(self, capsule_id: str) -> list[float]:
        item = self._items.get(capsule_id)
        return item.structural_vector if item else []

    def exact_seed_candidates(
        self,
        *,
        seed_node_ids: list[str],
        limit: int,
    ) -> list[CapsuleCandidate]:
        """Route exact linked entities without a redundant dense projection.

        Seed-containing capsules are selected first, one per seed, followed by
        their strongest supergraph neighbors. Learned projection remains the
        fallback for queries whose entity linker cannot identify a graph node.
        """
        if limit <= 0:
            return []
        matched_by_position: dict[int, set[str]] = {}
        for seed_id in dict.fromkeys(seed_node_ids):
            for position in self._node_positions.get(seed_id, ()):
                matched_by_position.setdefault(position, set()).add(seed_id)
        if not matched_by_position:
            return []

        def seed_key(position: int) -> tuple[int, int, str]:
            item = self._items[self._ordered_ids[position]]
            return (
                -len(matched_by_position[position]),
                len(item.capsule.node_ids),
                item.capsule.id,
            )

        ordered_seed_positions = sorted(matched_by_position, key=seed_key)
        selected_positions: list[int] = []
        selected_set: set[int] = set()
        for seed_id in dict.fromkeys(seed_node_ids):
            position = next(
                (
                    value
                    for value in ordered_seed_positions
                    if seed_id in matched_by_position[value]
                    and value not in selected_set
                ),
                None,
            )
            if position is not None:
                selected_positions.append(position)
                selected_set.add(position)
            if len(selected_positions) >= limit:
                break
        for position in ordered_seed_positions:
            if len(selected_positions) >= limit:
                break
            if position not in selected_set:
                selected_positions.append(position)
                selected_set.add(position)

        if len(selected_positions) < limit:
            frontier: list[tuple[float, str, int]] = []
            queued: set[int] = set()
            for position in selected_positions:
                capsule_id = self._ordered_ids[position]
                for neighbor_id, weight in self.supergraph.neighbors(capsule_id).items():
                    neighbor_position = self._position_by_id.get(neighbor_id)
                    if (
                        neighbor_position is None
                        or neighbor_position in selected_set
                        or neighbor_position in queued
                    ):
                        continue
                    heappush(frontier, (-float(weight), neighbor_id, neighbor_position))
                    queued.add(neighbor_position)
            # Best-first supergraph expansion provides the same fixed capsule
            # budget as learned routing without scoring the whole vector matrix.
            while frontier and len(selected_positions) < limit:
                _, _, position = heappop(frontier)
                queued.discard(position)
                if position in selected_set:
                    continue
                selected_positions.append(position)
                selected_set.add(position)
                capsule_id = self._ordered_ids[position]
                for neighbor_id, weight in self.supergraph.neighbors(capsule_id).items():
                    neighbor_position = self._position_by_id.get(neighbor_id)
                    if (
                        neighbor_position is None
                        or neighbor_position in selected_set
                        or neighbor_position in queued
                    ):
                        continue
                    heappush(
                        frontier,
                        (-float(weight), neighbor_id, neighbor_position),
                    )
                    queued.add(neighbor_position)

        if len(selected_positions) < limit:
            # A supergraph component can be intentionally disconnected. Fill
            # from the same offline partition region before using a stable
            # global fallback, preserving the configured capsule count without
            # paying for a dense all-capsule projection.
            region_ids = list(
                dict.fromkeys(
                    self._items[self._ordered_ids[position]].capsule.region_id
                    for position in selected_positions
                )
            )
            for region_id in region_ids:
                for position in self._region_positions.get(region_id, ()):
                    if len(selected_positions) >= limit:
                        break
                    if position not in selected_set:
                        selected_positions.append(position)
                        selected_set.add(position)
                if len(selected_positions) >= limit:
                    break
        if len(selected_positions) < limit:
            for position in range(len(self._ordered_ids)):
                if len(selected_positions) >= limit:
                    break
                if position not in selected_set:
                    selected_positions.append(position)
                    selected_set.add(position)

        candidates: list[CapsuleCandidate] = []
        for position in selected_positions:
            item = self._items[self._ordered_ids[position]]
            matched_seed_ids = sorted(matched_by_position.get(position, ()))
            exact = bool(matched_seed_ids)
            candidates.append(
                CapsuleCandidate(
                    capsule=item.capsule,
                    score=1.0 if exact else 0.5,
                    entity_score=1.0 if exact else 0.0,
                    source_score=float(self._source_scores[position]),
                    metadata={
                        "stage": "exact_seed_inverted_index",
                        "exact_seed_match": exact,
                        "matched_seed_ids": matched_seed_ids,
                    },
                )
            )
        return candidates

    def get(self, capsule_id: str) -> GraphCapsule | None:
        item = self._items.get(capsule_id)
        return item.capsule if item else None

    def capsules(self) -> list[GraphCapsule]:
        return [item.capsule for item in self._items.values()]


def _capsule_entity_tokens(capsule: GraphCapsule) -> set[str]:
    tokens = set()
    for node in capsule.nodes:
        tokens.add(_normalize_token(node.id))
        name = node.properties.get("name")
        if name:
            tokens.update(_tokenize(str(name)))
    return {token for token in tokens if token}


def _tokenize(text: str) -> list[str]:
    return [_normalize_token(match.group(0)) for match in _TOKEN_RE.finditer(text)]


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalized_matrix(vectors: list[list[float]]):
    if not vectors:
        return None
    try:
        import numpy as np
    except ImportError:
        return [_l2_normalize(vector) for vector in vectors]
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        return [_l2_normalize(vector) for vector in vectors]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _matrix_cosine_scores(matrix, query_vector: list[float]):
    if matrix is None:
        return []
    try:
        import numpy as np
    except ImportError:
        return [_cosine_similarity(query_vector, vector) for vector in matrix]
    if not isinstance(matrix, np.ndarray):
        return [_cosine_similarity(query_vector, vector) for vector in matrix]
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1 or query.size == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    dimension = min(matrix.shape[1], query.shape[0])
    if dimension <= 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    query = query[:dimension]
    norm = float(np.linalg.norm(query))
    if norm == 0.0:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    return matrix[:, :dimension] @ (query / norm)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def _float_array(values: list[float]):
    try:
        import numpy as np
    except ImportError:
        return list(values)
    return np.asarray(values, dtype=np.float32)


def _zero_array(length: int):
    try:
        import numpy as np
    except ImportError:
        return [0.0] * length
    return np.zeros(length, dtype=np.float32)


def _increment_positions(values, positions) -> None:
    if not positions:
        return
    try:
        import numpy as np
    except ImportError:
        for position in positions:
            values[position] += 1.0
        return
    if isinstance(values, np.ndarray):
        values[np.asarray(positions, dtype=np.int64)] += 1.0
        return
    for position in positions:
        values[position] += 1.0


def _divide_array(values, denominator: int):
    try:
        return values / float(denominator)
    except TypeError:
        return [value / float(denominator) for value in values]


def _top_position_pool(scores, matched_seeds: dict[int, set[str]], limit: int) -> list[int]:
    """Return every seed match plus enough top scores for an exact top-limit sort."""
    length = len(scores)
    if limit >= length:
        return list(range(length))
    try:
        import numpy as np
    except ImportError:
        highest = sorted(range(length), key=lambda position: scores[position], reverse=True)[:limit]
        return list(dict.fromkeys([*matched_seeds, *highest]))
    if not isinstance(scores, np.ndarray):
        highest = sorted(range(length), key=lambda position: scores[position], reverse=True)[:limit]
        return list(dict.fromkeys([*matched_seeds, *highest]))
    count = min(max(limit, 1), length)
    partitioned = np.argpartition(scores, length - count)[length - count :]
    cutoff = float(scores[partitioned].min())
    highest = np.flatnonzero(scores >= cutoff).tolist()
    return list(dict.fromkeys([*matched_seeds, *highest]))


def _weighted_scores(text_scores, entity_scores, source_scores):
    try:
        return 0.65 * text_scores + 0.30 * entity_scores + 0.05 * source_scores
    except TypeError:
        return [
            0.65 * float(text) + 0.30 * float(entity) + 0.05 * float(source)
            for text, entity, source in zip(text_scores, entity_scores, source_scores)
        ]


def _take_with_seed_coverage(
    candidates: list[CapsuleCandidate],
    seed_node_ids: list[str],
    limit: int,
) -> list[CapsuleCandidate]:
    if limit <= 0:
        return []
    selected: list[CapsuleCandidate] = []
    selected_ids: set[str] = set()
    for seed_id in dict.fromkeys(seed_node_ids):
        match = next(
            (
                candidate
                for candidate in candidates
                if seed_id in (candidate.metadata.get("matched_seed_ids") or [])
                and candidate.capsule.id not in selected_ids
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_ids.add(match.capsule.id)
        if len(selected) >= limit:
            return selected
    for candidate in candidates:
        if candidate.capsule.id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.capsule.id)
        if len(selected) >= limit:
            break
    return selected
