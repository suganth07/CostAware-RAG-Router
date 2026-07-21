from __future__ import annotations

from dataclasses import replace

from src.optimized_grag.models.structural import StructuralEncoder, cosine_similarity
from src.optimized_grag.partitioning.types import CapsuleCandidate
from src.optimized_grag.retrieval.capsule_index import CapsuleIndex
from src.optimized_grag.retrieval.scoring import CapsuleScoreWeights


class CapsuleRouter:
    def __init__(
        self,
        *,
        capsule_index: CapsuleIndex,
        structural_encoder: StructuralEncoder,
        weights: CapsuleScoreWeights | None = None,
    ) -> None:
        self.capsule_index = capsule_index
        self.structural_encoder = structural_encoder
        self.weights = weights or CapsuleScoreWeights()

    def route(
        self,
        *,
        query: str,
        query_vector: list[float],
        query_entities: list[str],
        seed_node_ids: list[str] | None = None,
        first_stage_limit: int,
        top_k: int,
    ) -> list[CapsuleCandidate]:
        candidates = self.capsule_index.candidates(
            query=query,
            query_vector=query_vector,
            query_entities=query_entities,
            seed_node_ids=seed_node_ids,
            limit=first_stage_limit,
        )
        project_structural_query = getattr(
            self.structural_encoder, "project_structural_query", None
        )
        query_graph_vector = (
            project_structural_query(query_vector)
            if callable(project_structural_query)
            else self.structural_encoder.project_query(query_vector)
        )
        selected_ids = {candidate.capsule.id for candidate in candidates[:top_k]}
        reranked: list[CapsuleCandidate] = []

        for candidate in candidates:
            structural_score = (
                cosine_similarity(
                    query_graph_vector,
                    self.capsule_index.structural_vector(candidate.capsule.id),
                )
                if query_graph_vector
                else 0.0
            )
            proximity_score = _proximity_score(
                candidate.capsule.id,
                selected_ids,
                self.capsule_index,
            )
            final_score = (
                self.weights.text * candidate.text_score
                + self.weights.entity * candidate.entity_score
                + self.weights.structural * structural_score
                + self.weights.source * candidate.source_score
                + self.weights.proximity * proximity_score
            )
            reranked.append(
                replace(
                    candidate,
                    score=final_score,
                    structural_score=structural_score,
                    proximity_score=proximity_score,
                    metadata={**candidate.metadata, "stage": "structural_rerank"},
                )
            )

        # Exact entity linking is stronger evidence than a learned or semantic
        # score. Keep a capsule containing the query seed ahead of capsules that
        # merely look similar, then use the hybrid score within each group.
        reranked.sort(
            key=lambda candidate: (
                len(candidate.metadata.get("matched_seed_ids") or []),
                candidate.score,
                candidate.capsule.id,
            ),
            reverse=True,
        )
        return _take_with_seed_coverage(reranked, seed_node_ids or [], top_k)


def _proximity_score(capsule_id: str, selected_ids: set[str], index: CapsuleIndex) -> float:
    neighbors = index.supergraph.neighbors(capsule_id)
    if not neighbors:
        return 0.0
    score = sum(weight for neighbor_id, weight in neighbors.items() if neighbor_id in selected_ids)
    total = sum(neighbors.values()) or 1.0
    return min(1.0, score / total)


def _take_with_seed_coverage(
    candidates: list[CapsuleCandidate],
    seed_node_ids: list[str],
    limit: int,
) -> list[CapsuleCandidate]:
    """Reserve one best capsule per linked seed, then fill by hybrid score."""
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
