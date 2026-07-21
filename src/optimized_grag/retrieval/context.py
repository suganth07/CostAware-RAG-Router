from __future__ import annotations

import math
import re

from src.optimized_grag.partitioning.types import CapsuleCandidate
from src.optimized_grag.retrieval.types import BuiltContext, EvidencePath, SourceSnippet

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "both", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with",
}


class OptimizedContextBuilder:
    """Build evidence-first context and expose exactly which sources were rendered."""

    def __init__(
        self,
        *,
        max_chars: int = 12_000,
        snippets_per_capsule: int = 3,
        max_source_snippets: int = 24,
    ) -> None:
        self.max_chars = max_chars
        self.snippets_per_capsule = snippets_per_capsule
        self.max_source_snippets = max_source_snippets

    def build(
        self,
        *,
        query: str,
        candidates: list[CapsuleCandidate],
        paths: list[EvidencePath],
    ) -> BuiltContext:
        ranked_snippets = self.rank_source_snippets(
            query=query,
            candidates=candidates,
            paths=paths,
        )
        lines: list[str] = []
        rendered_snippets: list[SourceSnippet] = []
        used_chars = 0

        def append(line: str) -> bool:
            nonlocal used_chars
            addition = len(line) + (1 if lines else 0)
            if used_chars + addition > self.max_chars:
                return False
            lines.append(line)
            used_chars += addition
            return True

        if paths:
            append("[Ranked graph evidence paths]")
            for index, path in enumerate(paths, start=1):
                seed = path.metadata.get("seed_node_id") or "fallback"
                if not append(f"{index}. score={path.score:.4f} seed={seed} {path.text}"):
                    break

        if ranked_snippets:
            if lines:
                append("")
            append("[Query-ranked source evidence]")
            for index, snippet in enumerate(ranked_snippets, start=1):
                provenance = snippet.document_id or snippet.chunk_id or snippet.id
                block = (
                    f"{index}. score={snippet.score:.4f} source={provenance} "
                    f"capsule={snippet.capsule_id}\n{snippet.text}"
                )
                if not append(block):
                    break
                rendered_snippets.append(snippet)

        if candidates:
            if lines:
                append("")
            append("[Selected graph capsule overview]")
            for index, candidate in enumerate(candidates, start=1):
                capsule = candidate.capsule
                matched = candidate.metadata.get("matched_seed_ids") or []
                block = (
                    f"{index}. capsule={capsule.id} score={candidate.score:.4f} "
                    f"matched_seeds={matched} nodes={capsule.stats.node_count} "
                    f"edges={capsule.stats.edge_count}\n{_capsule_overview(capsule)}"
                )
                if not append(block):
                    break

        return BuiltContext(
            text="\n".join(lines),
            source_snippets=rendered_snippets,
        )

    def rank_source_snippets(
        self,
        *,
        query: str,
        candidates: list[CapsuleCandidate],
        paths: list[EvidencePath],
    ) -> list[SourceSnippet]:
        query_terms = _content_terms(query)
        path_node_ids = {node.id for path in paths for node in path.nodes}
        best_by_id: dict[str, SourceSnippet] = {}

        for candidate in candidates:
            capsule = candidate.capsule
            candidate_score = max(0.0, min(1.0, (candidate.score + 1.0) / 2.0))
            for node in capsule.nodes:
                text = node.properties.get("text")
                if not text:
                    continue
                snippet_text = " ".join(str(text).split())
                snippet_terms = _content_terms(snippet_text)
                lexical_score = _cosine_token_overlap(query_terms, snippet_terms)
                path_bonus = 1.0 if node.id in path_node_ids else 0.0
                score = 0.65 * lexical_score + 0.20 * candidate_score + 0.15 * path_bonus
                chunk_id = str(node.properties.get("source_chunk_id") or node.id)
                document_id_value = node.properties.get("source_document_id")
                document_id = str(document_id_value) if document_id_value is not None else None
                snippet = SourceSnippet(
                    id=chunk_id,
                    text=snippet_text,
                    score=score,
                    capsule_id=capsule.id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    metadata={"node_id": node.id},
                )
                existing = best_by_id.get(snippet.id)
                if existing is None or (snippet.score, snippet.capsule_id) > (
                    existing.score,
                    existing.capsule_id,
                ):
                    best_by_id[snippet.id] = snippet

        ranked = sorted(
            best_by_id.values(),
            key=lambda snippet: (-snippet.score, snippet.id, snippet.capsule_id),
        )
        selected: list[SourceSnippet] = []
        per_capsule: dict[str, int] = {}
        for snippet in ranked:
            if per_capsule.get(snippet.capsule_id, 0) >= self.snippets_per_capsule:
                continue
            selected.append(snippet)
            per_capsule[snippet.capsule_id] = per_capsule.get(snippet.capsule_id, 0) + 1
            if len(selected) >= self.max_source_snippets:
                break
        return selected


def _content_terms(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS}


def _cosine_token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _capsule_overview(capsule: object) -> str:
    nodes = getattr(capsule, "nodes", [])
    names = [
        str(node.properties.get("name") or node.id)
        for node in nodes
        if "Chunk" not in node.labels
    ][:20]
    relations = sorted(getattr(capsule, "relation_types", set()))[:12]
    return (
        f"Anchor: {getattr(capsule, 'anchor_id', '')}. "
        f"Entities: {', '.join(names)}. Relations: {', '.join(relations)}."
    )
