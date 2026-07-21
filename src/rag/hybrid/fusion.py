from __future__ import annotations

from dataclasses import replace

from src.rag.types import RetrievalResult


def reciprocal_rank_fusion(
    result_sets: list[list[RetrievalResult]],
    *,
    limit: int,
    rank_constant: int = 60,
) -> list[RetrievalResult]:
    fused: dict[str, RetrievalResult] = {}
    scores: dict[str, float] = {}
    score_parts: dict[str, dict[str, float]] = {}
    sources: dict[str, list[str]] = {}

    for results in result_sets:
        for rank, result in enumerate(results, start=1):
            contribution = 1.0 / (rank_constant + rank)
            scores[result.id] = scores.get(result.id, 0.0) + contribution
            score_parts.setdefault(result.id, {})[result.source] = contribution
            sources.setdefault(result.id, [])
            if result.source not in sources[result.id]:
                sources[result.id].append(result.source)
            if result.id not in fused:
                fused[result.id] = result

    merged = []
    for result_id, result in fused.items():
        metadata = {
            **result.metadata,
            "hybrid_sources": sources.get(result_id, []),
            "hybrid_score_parts": score_parts.get(result_id, {}),
        }
        merged.append(
            replace(
                result,
                score=scores[result_id],
                source="hybrid",
                metadata=metadata,
            )
        )

    merged.sort(key=lambda result: result.score, reverse=True)
    return merged[:limit]
