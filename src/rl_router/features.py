"""Deterministic, inexpensive features available before route selection."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_BRACKETED_ENTITY_RE = re.compile(r"\[[^\]]+\]")
_CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][\w'-]*\s+){0,3}[A-Z][\w'-]*\b")
_RELATION_TERMS = {
    "actor", "actors", "author", "authored", "born", "capital", "created",
    "directed", "director", "language", "located", "married", "parent",
    "producer", "released", "starred", "written", "writer",
}
_MULTIHOP_TERMS = {
    "both", "between", "first", "former", "following", "same", "shared",
    "through", "whose", "with", "which",
}
_COMPARISON_TERMS = {"after", "before", "larger", "more", "most", "than", "when"}


def feature_names() -> tuple[str, ...]:
    """Stable ordered names used by the persisted linear policy."""
    return (
        "bias",
        "token_count",
        "character_count",
        "bracketed_entity_count",
        "capitalized_phrase_count",
        "relation_term_count",
        "multihop_term_count",
        "comparison_term_count",
        "question_mark",
    )


def query_features(question: str) -> dict[str, float]:
    """Return small interpretable route-selection features for one question.

    These deliberately avoid an LLM, R-GCN, and query projector.  They can be
    measured before any of the three expensive retrieval routes are run.
    """
    text = str(question or "")
    tokens = [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
    return {
        "bias": 1.0,
        "token_count": float(len(tokens)),
        "character_count": float(len(text)),
        "bracketed_entity_count": float(len(_BRACKETED_ENTITY_RE.findall(text))),
        "capitalized_phrase_count": float(len(_CAPITALIZED_PHRASE_RE.findall(text))),
        "relation_term_count": float(sum(token in _RELATION_TERMS for token in tokens)),
        "multihop_term_count": float(sum(token in _MULTIHOP_TERMS for token in tokens)),
        "comparison_term_count": float(sum(token in _COMPARISON_TERMS for token in tokens)),
        "question_mark": float("?" in text),
    }
