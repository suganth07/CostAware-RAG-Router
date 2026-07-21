from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]+")
BRACKET_ENTITY_RE = re.compile(r"\[([^\]]+)\]")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "which",
        "who",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectionExample:
    """One query with every retained positive and a set of hard negatives."""

    sample_id: str
    question: str
    positive_capsule_ids: tuple[str, ...]
    hard_negative_capsule_ids: tuple[str, ...]
    slice_name: str


def mine_metaqa_path_projection_examples(
    samples: Iterable[object],
    capsule_search_text_by_id: Mapping[str, str],
    capsule_edge_ids_by_id: Mapping[str, Sequence[str]],
    graph_edges: Sequence[tuple[str, str, str]],
    *,
    capsule_node_ids_by_id: Mapping[str, Sequence[str]] | None = None,
    max_examples: int | None = None,
    max_positives: int | None = 16,
    max_negatives: int = 20,
) -> tuple[list[ProjectionExample], dict[str, object]]:
    """Mine MetaQA router labels from gold paths rather than answer endpoints.

    The router's job is to choose a capsule from which path search can begin.
    Therefore a positive capsule contains a first edge on a <=3-hop path from
    the bracketed query entity to any gold answer. It need not contain the final
    answer itself. Multiple valid first edges and overlapping capsules are all
    retained as positives.
    """
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or None")
    if max_positives is not None and max_positives < 1:
        raise ValueError("max_positives must be positive or None")
    if max_negatives < 1:
        raise ValueError("max_negatives must be positive")

    capsule_ids = sorted(capsule_search_text_by_id)
    capsule_tokens = {
        capsule_id: frozenset(
            normalized_tokens(capsule_search_text_by_id[capsule_id], remove_stopwords=True)
        )
        for capsule_id in capsule_ids
    }
    token_to_capsules: dict[str, set[str]] = {}
    for capsule_id, tokens in capsule_tokens.items():
        for token in tokens:
            token_to_capsules.setdefault(token, set()).add(capsule_id)

    edge_to_capsules: dict[str, set[str]] = {}
    for capsule_id, edge_ids in capsule_edge_ids_by_id.items():
        for edge_id in edge_ids:
            edge_to_capsules.setdefault(str(edge_id), set()).add(capsule_id)
    node_to_capsules: dict[str, set[str]] = {}
    for capsule_id, node_ids in (capsule_node_ids_by_id or {}).items():
        for node_id in node_ids:
            node_to_capsules.setdefault(str(node_id), set()).add(capsule_id)

    adjacency: dict[str, list[tuple[str, str]]] = {}
    edge_endpoints: dict[str, tuple[str, str]] = {}
    for edge_id, source_id, target_id in graph_edges:
        edge_endpoints[edge_id] = (source_id, target_id)
        adjacency.setdefault(source_id, []).append((target_id, edge_id))
        adjacency.setdefault(target_id, []).append((source_id, edge_id))
    for neighbors in adjacency.values():
        neighbors.sort(key=lambda row: (row[0], row[1]))

    examples: list[ProjectionExample] = []
    slice_counts: dict[str, int] = {}
    positive_counts: list[int] = []
    retained_counts: list[int] = []
    first_edge_counts: list[int] = []
    skipped_no_seed = 0
    skipped_no_path = 0
    skipped_no_capsule = 0

    for sample_number, sample in enumerate(samples):
        question = str(getattr(sample, "question", ""))
        seed_match = BRACKET_ENTITY_RE.search(question)
        if not seed_match:
            skipped_no_seed += 1
            continue
        seed_id = _entity_id(seed_match.group(1))
        answer_ids = {
            _entity_id(str(answer))
            for answer in getattr(sample, "answers", ())
            if str(answer).strip()
        }
        metadata = dict(getattr(sample, "metadata", {}) or {})
        max_hops = _hop_limit(metadata.get("hop"))
        first_edge_ids = _gold_first_edge_ids(
            adjacency,
            seed_id=seed_id,
            answer_ids=answer_ids,
            max_hops=max_hops,
        )
        if not first_edge_ids:
            skipped_no_path += 1
            continue

        positive_ids: set[str] = set()
        for edge_id in first_edge_ids:
            positive_ids.update(edge_to_capsules.get(edge_id, ()))
            if not edge_to_capsules.get(edge_id) and capsule_node_ids_by_id:
                source_id, target_id = edge_endpoints[edge_id]
                neighbor_id = target_id if source_id == seed_id else source_id
                positive_ids.update(
                    node_to_capsules.get(seed_id, set())
                    & node_to_capsules.get(neighbor_id, set())
                )
        if not positive_ids:
            skipped_no_capsule += 1
            continue

        question_tokens = frozenset(normalized_tokens(question, remove_stopwords=True))
        ranked_positives = sorted(
            positive_ids,
            key=lambda capsule_id: (
                len(capsule_tokens[capsule_id] & question_tokens),
                -len(capsule_tokens[capsule_id]),
                capsule_id,
            ),
            reverse=True,
        )
        retained_positive_ids = (
            ranked_positives
            if max_positives is None
            else ranked_positives[:max_positives]
        )

        ranked_query_tokens = sorted(
            (
                (len(token_to_capsules[token]), token)
                for token in question_tokens
                if token in token_to_capsules
            ),
            key=lambda row: (row[0], row[1]),
        )
        negative_candidates: set[str] = set()
        for _, token in ranked_query_tokens[:4]:
            negative_candidates.update(token_to_capsules[token])
        ranked_negatives = sorted(
            negative_candidates - positive_ids,
            key=lambda capsule_id: (
                len(capsule_tokens[capsule_id] & question_tokens),
                -len(capsule_tokens[capsule_id]),
                capsule_id,
            ),
            reverse=True,
        )
        negative_ids = ranked_negatives[:max_negatives]
        if len(negative_ids) < max_negatives:
            for capsule_id in _rotated_ids(capsule_ids, sample_number):
                if capsule_id not in positive_ids and capsule_id not in negative_ids:
                    negative_ids.append(capsule_id)
                    if len(negative_ids) >= max_negatives:
                        break
        if not negative_ids:
            continue

        slice_name = str(metadata.get("hop") or "default")
        examples.append(
            ProjectionExample(
                sample_id=str(getattr(sample, "id", sample_number)),
                question=question,
                positive_capsule_ids=tuple(retained_positive_ids),
                hard_negative_capsule_ids=tuple(negative_ids),
                slice_name=slice_name,
            )
        )
        slice_counts[slice_name] = slice_counts.get(slice_name, 0) + 1
        positive_counts.append(len(positive_ids))
        retained_counts.append(len(retained_positive_ids))
        first_edge_counts.append(len(first_edge_ids))
        if max_examples is not None and len(examples) >= max_examples:
            break

    report: dict[str, object] = {
        "examples": len(examples),
        "slice_counts": dict(sorted(slice_counts.items())),
        "skipped_no_seed": skipped_no_seed,
        "skipped_no_path": skipped_no_path,
        "skipped_no_capsule": skipped_no_capsule,
        "mean_gold_first_edges": _mean(first_edge_counts),
        "mean_all_positive_capsules": _mean(positive_counts),
        "max_all_positive_capsules": max(positive_counts, default=0),
        "mean_retained_positive_capsules": _mean(retained_counts),
        "max_retained_positive_capsules": max(retained_counts, default=0),
        "positive_cap": max_positives,
        "hard_negative_cap": max_negatives,
        "label_rule": "capsule_contains_gold_path_first_edge_or_endpoint_pair",
    }
    return examples, report


def normalized_tokens(text: str, *, remove_stopwords: bool = False) -> tuple[str, ...]:
    tokens = tuple(TOKEN_RE.findall(str(text).lower()))
    if not remove_stopwords:
        return tokens
    return tuple(token for token in tokens if token not in STOPWORDS)


def mine_projection_examples(
    samples: Iterable[object],
    capsule_search_text_by_id: Mapping[str, str],
    *,
    max_examples: int | None = None,
    max_positives: int | None = 16,
    max_negatives: int = 20,
) -> tuple[list[ProjectionExample], dict[str, object]]:
    """Mine deterministic multi-positive supervision from gold evidence.

    An answer is positive only when its complete normalized phrase occurs in a
    capsule's search text. This is substantially safer than token overlap for
    short/common answers such as years and genres. Dataset-provided evidence
    titles are accepted as additional exact phrases; longer evidence facts are
    used to rank already-positive capsules, not to create loose positives.

    `max_positives=None` retains the full positive set and is intended for
    evaluation. Training should cap it to bound memory and prevent very common
    answers from dominating a batch.
    """
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive or None")
    if max_positives is not None and max_positives < 1:
        raise ValueError("max_positives must be positive or None")
    if max_negatives < 1:
        raise ValueError("max_negatives must be positive")

    capsule_ids = sorted(capsule_search_text_by_id)
    capsule_tokens = {
        capsule_id: normalized_tokens(capsule_search_text_by_id[capsule_id])
        for capsule_id in capsule_ids
    }
    capsule_token_sets = {
        capsule_id: frozenset(tokens) for capsule_id, tokens in capsule_tokens.items()
    }
    capsule_normalized_text = {
        capsule_id: f" {' '.join(tokens)} "
        for capsule_id, tokens in capsule_tokens.items()
    }
    token_to_capsules: dict[str, set[str]] = {}
    for capsule_id, tokens in capsule_token_sets.items():
        for token in tokens:
            token_to_capsules.setdefault(token, set()).add(capsule_id)

    examples: list[ProjectionExample] = []
    slice_counts: dict[str, int] = {}
    skipped_no_gold = 0
    skipped_no_positive = 0
    all_positive_counts: list[int] = []
    retained_positive_counts: list[int] = []

    for sample_number, sample in enumerate(samples):
        answers = [str(value) for value in getattr(sample, "answers", ()) if str(value).strip()]
        metadata = dict(getattr(sample, "metadata", {}) or {})
        answer_phrases = _unique_phrases(answers)
        evidence_phrases, evidence_fact_tokens = _evidence_supervision(metadata)
        gold_phrases = _deduplicate_phrases([*answer_phrases, *evidence_phrases])
        if not gold_phrases:
            skipped_no_gold += 1
            continue

        question_tokens = frozenset(
            normalized_tokens(getattr(sample, "question", ""), remove_stopwords=True)
        )
        positive_candidate_ids: set[str] = set()
        for phrase in gold_phrases:
            postings = [token_to_capsules.get(token, set()) for token in set(phrase)]
            if postings and all(postings):
                positive_candidate_ids.update(set.intersection(*postings))

        # Common relation words (for example "starred") occur in most MetaQA
        # capsules. Hard-negative mining only needs the most discriminative
        # query terms, so use at most four terms with the smallest posting lists.
        ranked_query_tokens = sorted(
            (
                (len(token_to_capsules[token]), token)
                for token in question_tokens
                if token in token_to_capsules
            ),
            key=lambda row: (row[0], row[1]),
        )
        negative_candidate_ids: set[str] = set()
        for _, token in ranked_query_tokens[:4]:
            negative_candidate_ids.update(token_to_capsules[token])
        scored_positives: list[tuple[tuple[float, ...], str]] = []
        for capsule_id in positive_candidate_ids:
            tokens = capsule_tokens[capsule_id]
            token_set = capsule_token_sets[capsule_id]
            normalized_text = capsule_normalized_text[capsule_id]
            answer_matches = sum(
                _contains_normalized_phrase(normalized_text, phrase)
                for phrase in answer_phrases
            )
            evidence_matches = sum(
                _contains_normalized_phrase(normalized_text, phrase)
                for phrase in evidence_phrases
            )
            question_overlap = len(token_set & question_tokens)
            evidence_overlap = len(token_set & evidence_fact_tokens)
            if answer_matches or evidence_matches:
                score = (
                    float(answer_matches),
                    float(evidence_matches),
                    float(evidence_overlap),
                    float(question_overlap),
                    -float(len(tokens)),
                )
                scored_positives.append((score, capsule_id))

        if not scored_positives:
            skipped_no_positive += 1
            continue

        scored_positives.sort(key=lambda row: (row[0], row[1]), reverse=True)
        all_positive_ids = [capsule_id for _, capsule_id in scored_positives]
        retained_positive_ids = (
            all_positive_ids
            if max_positives is None
            else all_positive_ids[:max_positives]
        )
        positive_set = set(all_positive_ids)
        scored_negatives: list[tuple[tuple[float, ...], str]] = []
        for capsule_id in negative_candidate_ids - positive_set:
            tokens = capsule_tokens[capsule_id]
            token_set = capsule_token_sets[capsule_id]
            score = (
                float(len(token_set & question_tokens)),
                float(len(token_set & evidence_fact_tokens)),
                -float(len(tokens)),
            )
            scored_negatives.append((score, capsule_id))
        scored_negatives.sort(key=lambda row: (row[0], row[1]), reverse=True)
        negative_ids = [
            capsule_id
            for _, capsule_id in scored_negatives
            if capsule_id not in positive_set
        ][:max_negatives]
        if len(negative_ids) < max_negatives:
            for capsule_id in _rotated_ids(capsule_ids, sample_number):
                if capsule_id not in positive_set and capsule_id not in negative_ids:
                    negative_ids.append(capsule_id)
                    if len(negative_ids) >= max_negatives:
                        break
        if not negative_ids:
            continue

        slice_name = str(metadata.get("hop") or metadata.get("type") or "default")
        examples.append(
            ProjectionExample(
                sample_id=str(getattr(sample, "id", sample_number)),
                question=str(getattr(sample, "question", "")),
                positive_capsule_ids=tuple(retained_positive_ids),
                hard_negative_capsule_ids=tuple(negative_ids),
                slice_name=slice_name,
            )
        )
        all_positive_counts.append(len(all_positive_ids))
        retained_positive_counts.append(len(retained_positive_ids))
        slice_counts[slice_name] = slice_counts.get(slice_name, 0) + 1
        if max_examples is not None and len(examples) >= max_examples:
            break

    report: dict[str, object] = {
        "examples": len(examples),
        "slice_counts": dict(sorted(slice_counts.items())),
        "skipped_no_gold": skipped_no_gold,
        "skipped_no_positive": skipped_no_positive,
        "mean_all_positive_capsules": _mean(all_positive_counts),
        "max_all_positive_capsules": max(all_positive_counts, default=0),
        "mean_retained_positive_capsules": _mean(retained_positive_counts),
        "max_retained_positive_capsules": max(retained_positive_counts, default=0),
        "positive_cap": max_positives,
        "hard_negative_cap": max_negatives,
        "label_rule": "exact_normalized_answer_or_evidence_phrase",
    }
    return examples, report


def any_positive_ranking_metrics(
    ranks: Sequence[int],
    positive_counts: Sequence[int],
    *,
    total_capsules: int | Sequence[int],
    ks: Sequence[int] = (1, 5, 10, 20),
) -> dict[str, float | int]:
    """Summarize best-positive ranks and matching random-ranker baselines."""
    if len(ranks) != len(positive_counts):
        raise ValueError("ranks and positive_counts must have equal length")
    totals = (
        [int(total_capsules)] * len(ranks)
        if isinstance(total_capsules, int)
        else [int(value) for value in total_capsules]
    )
    if len(totals) != len(ranks):
        raise ValueError("candidate totals and ranks must have equal length")
    if any(total < 1 for total in totals):
        raise ValueError("total_capsules values must be positive")
    if not ranks:
        return {"evaluated_examples": 0, "any_positive_mrr": 0.0}

    metrics: dict[str, float | int] = {
        "evaluated_examples": len(ranks),
        "any_positive_mrr": _mean([1.0 / max(int(rank), 1) for rank in ranks]),
        "mean_best_positive_rank": _mean([float(rank) for rank in ranks]),
        "mean_positive_capsules": _mean([float(value) for value in positive_counts]),
        "max_positive_capsules": max(positive_counts, default=0),
        "random_any_positive_mrr": _mean(
            [_random_best_rank_mrr(total, count) for total, count in zip(totals, positive_counts)]
        ),
    }
    for k in ks:
        metrics[f"any_positive_recall_at_{k}"] = _mean(
            [1.0 if rank <= min(max(int(k), 1), total) else 0.0 for rank, total in zip(ranks, totals)]
        )
        metrics[f"random_any_positive_recall_at_{k}"] = _mean(
            [
                _random_recall(total, count, min(max(int(k), 1), total))
                for total, count in zip(totals, positive_counts)
            ]
        )
    return metrics


def _evidence_supervision(metadata: Mapping[str, object]) -> tuple[list[tuple[str, ...]], frozenset[str]]:
    phrase_texts: list[str] = []
    fact_texts: list[str] = []
    for evidence in metadata.get("evidence_list") or []:
        if isinstance(evidence, Mapping):
            phrase_texts.extend(
                str(evidence.get(key) or "") for key in ("title", "source")
            )
            fact_texts.append(str(evidence.get("fact") or ""))
    supporting = metadata.get("supporting_facts") or {}
    if isinstance(supporting, Mapping):
        phrase_texts.extend(str(value) for value in supporting.get("title") or [])
    for evidence in metadata.get("evidences") or []:
        if isinstance(evidence, (list, tuple)) and evidence:
            phrase_texts.append(str(evidence[0]))
            fact_texts.append(" ".join(str(part) for part in evidence[1:]))
        else:
            fact_texts.append(str(evidence))
    phrases = _unique_phrases(phrase_texts)
    fact_tokens = frozenset(
        normalized_tokens(" ".join(fact_texts), remove_stopwords=True)
    )
    return phrases, fact_tokens


def _gold_first_edge_ids(
    adjacency: Mapping[str, Sequence[tuple[str, str]]],
    *,
    seed_id: str,
    answer_ids: set[str],
    max_hops: int,
) -> set[str]:
    result: set[str] = set()
    if seed_id not in adjacency or not answer_ids:
        return result
    for first_node, first_edge_id in adjacency[seed_id]:
        if first_node in answer_ids:
            result.add(first_edge_id)
            continue
        if max_hops < 2:
            continue
        found = False
        for second_node, _ in adjacency.get(first_node, ()):
            if second_node == seed_id:
                continue
            if second_node in answer_ids:
                found = True
                break
            if max_hops < 3:
                continue
            for third_node, _ in adjacency.get(second_node, ()):
                if third_node in answer_ids:
                    found = True
                    break
            if found:
                break
        if found:
            result.add(first_edge_id)
    return result


def _entity_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")


def _hop_limit(value: object) -> int:
    match = re.search(r"([123])", str(value or ""))
    return int(match.group(1)) if match else 3


def _unique_phrases(texts: Iterable[str]) -> list[tuple[str, ...]]:
    return _deduplicate_phrases(
        normalized_tokens(text) for text in texts if str(text).strip()
    )


def _deduplicate_phrases(phrases: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for phrase in phrases:
        if phrase and phrase not in seen:
            result.append(phrase)
            seen.add(phrase)
    return result


def _contains_normalized_phrase(normalized_text: str, phrase: tuple[str, ...]) -> bool:
    return f" {' '.join(phrase)} " in normalized_text


def _rotated_ids(values: Sequence[str], offset: int) -> Iterable[str]:
    if not values:
        return ()
    start = offset % len(values)
    return (*values[start:], *values[:start])


def _random_recall(total: int, positives: int, k: int) -> float:
    positives = min(max(int(positives), 0), total)
    if positives == 0 or k == 0:
        return 0.0
    if k >= total - positives + 1:
        return 1.0
    log_miss = 0.0
    for draw in range(k):
        log_miss += math.log(total - positives - draw) - math.log(total - draw)
    return 1.0 - math.exp(log_miss)


def _random_best_rank_mrr(total: int, positives: int) -> float:
    positives = min(max(int(positives), 0), total)
    if positives == 0:
        return 0.0
    probability = positives / total
    expected = probability
    for rank in range(1, total - positives + 1):
        probability *= (total - rank - positives + 1) / (total - rank)
        expected += probability / (rank + 1)
    return expected


def _mean(values: Sequence[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
