"""Evidence-ranking metrics with one-to-one gold matching."""

from __future__ import annotations

import math
from typing import Any, Sequence

from src.evaluation.core import EvidenceUnit, normalize_text


def evidence_ndcg_at_k(gold_evidence: Sequence[dict[str, Any]], units: Sequence[EvidenceUnit], *, k: int = 10) -> float | None:
    """Binary evidence nDCG that does not double-count the same gold fact."""
    if not gold_evidence:
        return None
    if k < 1:
        raise ValueError("k must be positive")
    unmatched = list(gold_evidence)
    dcg = 0.0
    for rank, unit in enumerate(units[:k], start=1):
        match_index = next((index for index, gold in enumerate(unmatched) if _matches(gold, unit)), None)
        if match_index is None:
            continue
        dcg += 1.0 / math.log2(rank + 1)
        unmatched.pop(match_index)
    ideal_count = min(k, len(gold_evidence))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else None


def _matches(gold: dict[str, Any], unit: EvidenceUnit) -> bool:
    identifiers = {str(value).strip().casefold() for value in gold.get("identifiers", []) if value}
    unit_identifiers = {
        str(value).strip().casefold()
        for value in (unit.id, unit.metadata.get("document_id"), unit.metadata.get("url"), unit.metadata.get("title"))
        if value
    }
    if identifiers & unit_identifiers:
        return True
    text = normalize_text(unit.text)
    return any((needle := normalize_text(value)) and len(needle) >= 4 and needle in text for value in gold.get("texts", []))
