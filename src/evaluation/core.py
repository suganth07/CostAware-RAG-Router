from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable, Sequence

from src.adapters.datasets.base import DatasetBundle, EvaluationSample

EVALUATION_SCHEMA_VERSION = 3
DEFAULT_RECALL_K = (1, 3, 5, 10, 20)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_BOOLEAN = {"yes", "no"}


@dataclass(slots=True)
class EvidenceUnit:
    id: str
    text: str
    score: float = 0.0
    kind: str = "source"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceUnit":
        return cls(
            id=str(value.get("id") or ""),
            text=str(value.get("text") or ""),
            score=float(value.get("score") or 0.0),
            kind=str(value.get("kind") or "source"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(slots=True)
class RetrievalRecord:
    system: str
    sample_id: str
    question: str
    context_text: str
    generated_answer: str = ""
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    capsule_ids: list[str] = field(default_factory=list)
    seed_node_ids: list[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "sample_id": self.sample_id,
            "question": self.question,
            "context_text": self.context_text,
            "generated_answer": self.generated_answer,
            "evidence_units": [unit.to_dict() for unit in self.evidence_units],
            "capsule_ids": list(self.capsule_ids),
            "seed_node_ids": list(self.seed_node_ids),
            "latency_seconds": self.latency_seconds,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalRecord":
        return cls(
            system=str(value.get("system") or ""),
            sample_id=str(value.get("sample_id") or ""),
            question=str(value.get("question") or ""),
            context_text=str(value.get("context_text") or ""),
            generated_answer=str(value.get("generated_answer") or ""),
            evidence_units=[
                EvidenceUnit.from_dict(unit) for unit in value.get("evidence_units") or []
            ],
            capsule_ids=[str(value) for value in value.get("capsule_ids") or []],
            seed_node_ids=[str(value) for value in value.get("seed_node_ids") or []],
            latency_seconds=float(value.get("latency_seconds") or 0.0),
            details=dict(value.get("details") or {}),
        )


@dataclass(slots=True)
class GoldEvidenceItem:
    identifiers: set[str] = field(default_factory=set)
    texts: list[str] = field(default_factory=list)


def normalize_text(value: object) -> str:
    return " ".join(_TOKEN_RE.findall(str(value).lower()))


def answer_kind(answers: Sequence[str]) -> str:
    normalized = [normalize_text(answer) for answer in answers]
    if normalized and all(answer in _BOOLEAN for answer in normalized):
        return "boolean"
    if any("insufficient information" in answer for answer in normalized):
        return "null"
    return "span"


def exact_match(prediction: str, answers: Sequence[str]) -> float | None:
    if not prediction.strip():
        return None
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(answer) for answer in answers))


def token_f1(prediction: str, answers: Sequence[str]) -> float | None:
    if not prediction.strip():
        return None
    prediction_tokens = normalize_text(prediction).split()
    if not prediction_tokens:
        return 0.0
    scores = []
    for answer in answers:
        answer_tokens = normalize_text(answer).split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if not answer_tokens or overlap == 0:
            scores.append(0.0)
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def boolean_accuracy(prediction: str, answers: Sequence[str]) -> float | None:
    if answer_kind(answers) != "boolean" or not prediction.strip():
        return None
    predicted = _extract_boolean(prediction)
    if predicted is None:
        return 0.0
    gold = {normalize_text(answer) for answer in answers}
    return float(predicted in gold)


def context_answer_recall(context: str, answers: Sequence[str]) -> float | None:
    if answer_kind(answers) != "span":
        return None
    haystack = normalize_text(context)
    return float(any(normalize_text(answer) in haystack for answer in answers if answer))


def gold_evidence_items(sample: EvaluationSample) -> list[GoldEvidenceItem]:
    metadata = sample.metadata
    evidence_list = metadata.get("evidence_list") or []
    if evidence_list:
        result = []
        for evidence in evidence_list:
            if not isinstance(evidence, dict):
                continue
            identifiers = {
                _normalize_identifier(value)
                for value in (evidence.get("url"), evidence.get("title"))
                if value
            }
            texts = [
                str(value)
                for value in (evidence.get("title"), evidence.get("fact"))
                if value
            ]
            result.append(GoldEvidenceItem(identifiers=identifiers, texts=texts))
        return result

    supporting = metadata.get("supporting_facts") or {}
    titles = supporting.get("title") if isinstance(supporting, dict) else []
    if titles:
        return [
            GoldEvidenceItem(
                identifiers={_normalize_identifier(title)},
                texts=[str(title)],
            )
            for title in dict.fromkeys(str(title) for title in titles)
        ]

    evidences = metadata.get("evidences") or []
    if evidences:
        return [
            GoldEvidenceItem(texts=[" ".join(str(part) for part in evidence)])
            for evidence in evidences
            if evidence
        ]

    if answer_kind(sample.answers) == "span":
        # MetaQA has no explicit supporting-fact annotation. Treat each gold
        # endpoint as an answer-evidence proxy so graph paths remain measurable.
        return [GoldEvidenceItem(texts=[answer]) for answer in sample.answers if answer]
    return []


def evaluate_record(
    sample: EvaluationSample,
    record: RetrievalRecord,
    *,
    recall_k: Sequence[int] = DEFAULT_RECALL_K,
) -> dict[str, Any]:
    gold_items = gold_evidence_items(sample)
    row: dict[str, Any] = {
        "system": record.system,
        "sample_id": sample.id,
        "question": sample.question,
        "answers": list(sample.answers),
        "answer_kind": answer_kind(sample.answers),
        "question_type": sample.metadata.get("question_type")
        or sample.metadata.get("type"),
        "latency_seconds": record.latency_seconds,
        "context_chars": len(record.context_text),
        "retrieved_units": len(record.evidence_units),
        "selected_capsules": len(record.capsule_ids),
        "linked_seeds": len(record.seed_node_ids),
        "context_answer_recall": context_answer_recall(
            record.context_text, sample.answers
        ),
        "exact_match": exact_match(record.generated_answer, sample.answers),
        "token_f1": token_f1(record.generated_answer, sample.answers),
        "boolean_accuracy": boolean_accuracy(
            record.generated_answer, sample.answers
        ),
        "supporting_evidence_recall": _evidence_recall(
            gold_items, record.evidence_units
        ),
        "evidence_mrr": _evidence_mrr(gold_items, record.evidence_units),
        "generated_answer": record.generated_answer,
        "context_text": record.context_text,
        "capsule_ids": list(record.capsule_ids),
        "seed_node_ids": list(record.seed_node_ids),
        "cache_hit": bool(record.details.get("cache_hit")),
    }
    for k in sorted(set(int(value) for value in recall_k if int(value) > 0)):
        row[f"evidence_recall_at_{k}"] = _evidence_recall(
            gold_items, record.evidence_units[:k]
        )
        row[f"answer_recall_at_{k}"] = _answer_recall_at_k(
            sample.answers, record.evidence_units[:k]
        )

    relevant_capsules = [
        str(value) for value in sample.metadata.get("relevant_capsule_ids") or []
    ]
    row["capsule_mrr"] = _identifier_mrr(relevant_capsules, record.capsule_ids)
    for k in sorted(set(int(value) for value in recall_k if int(value) > 0)):
        row[f"capsule_recall_at_{k}"] = _identifier_recall(
            relevant_capsules, record.capsule_ids[:k]
        )
    return row


def summarize_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in {"context_chars", "retrieved_units", "selected_capsules", "linked_seeds"}
        }
    )
    means = {
        metric: _mean([row.get(metric) for row in rows])
        for metric in metric_names
    }
    means.update(
        {
            "queries": len(rows),
            "latency_p50_seconds": _percentile(
                [row.get("latency_seconds") for row in rows], 0.50
            ),
            "latency_p95_seconds": _percentile(
                [row.get("latency_seconds") for row in rows], 0.95
            ),
            "mean_context_chars": _mean([row.get("context_chars") for row in rows]),
            "mean_retrieved_units": _mean([row.get("retrieved_units") for row in rows]),
            "mean_selected_capsules": _mean([row.get("selected_capsules") for row in rows]),
            "cache_hit_rate": _mean([float(bool(row.get("cache_hit"))) for row in rows]),
            "answer_kind_counts": dict(Counter(str(row.get("answer_kind")) for row in rows)),
        }
    )
    return means


def fingerprint_bundle(
    bundle: DatasetBundle,
    samples: Sequence[EvaluationSample],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"schema:{EVALUATION_SCHEMA_VERSION}|dataset:{bundle.name}".encode())
    for document in bundle.documents:
        digest.update(document.id.encode("utf-8", errors="replace"))
        digest.update(hashlib.sha256(document.text.encode("utf-8", errors="replace")).digest())
    for node in sorted(bundle.graph_nodes, key=lambda value: value.id):
        digest.update(node.id.encode("utf-8", errors="replace"))
        digest.update(str(node.properties.get("name") or "").encode("utf-8", errors="replace"))
    for edge in sorted(bundle.graph_edges, key=lambda value: value.id):
        digest.update(
            f"{edge.id}|{edge.source_id}|{edge.type}|{edge.target_id}".encode(
                "utf-8", errors="replace"
            )
        )
    for sample in samples:
        digest.update(
            json.dumps(
                [sample.id, sample.question, sample.answers],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def fingerprint_run(payload: Any) -> str:
    normalized = _jsonable(payload)
    serialized = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _extract_boolean(value: str) -> str | None:
    tokens = _TOKEN_RE.findall(value.lower())
    found = [token for token in tokens if token in _BOOLEAN]
    if not found or len(set(found)) > 1:
        return None
    return found[0]


def _evidence_recall(
    gold_items: Sequence[GoldEvidenceItem],
    units: Sequence[EvidenceUnit],
) -> float | None:
    if not gold_items:
        return None
    recalled = sum(any(_unit_matches(item, unit) for unit in units) for item in gold_items)
    return recalled / len(gold_items)


def _evidence_mrr(
    gold_items: Sequence[GoldEvidenceItem],
    units: Sequence[EvidenceUnit],
) -> float | None:
    if not gold_items:
        return None
    for rank, unit in enumerate(units, start=1):
        if any(_unit_matches(item, unit) for item in gold_items):
            return 1.0 / rank
    return 0.0


def _unit_matches(item: GoldEvidenceItem, unit: EvidenceUnit) -> bool:
    unit_id = _normalize_identifier(unit.id)
    aliases = {
        _normalize_identifier(value)
        for value in (
            unit.id,
            unit.metadata.get("document_id"),
            unit.metadata.get("url"),
            unit.metadata.get("title"),
        )
        if value
    }
    if item.identifiers & ({unit_id} | aliases):
        return True
    haystack = normalize_text(unit.text)
    for text in item.texts:
        needle = normalize_text(text)
        if len(needle) >= 4 and needle in haystack:
            return True
    return False


def _answer_recall_at_k(
    answers: Sequence[str],
    units: Sequence[EvidenceUnit],
) -> float | None:
    if answer_kind(answers) != "span":
        return None
    haystack = normalize_text(" ".join(unit.text for unit in units))
    return float(any(normalize_text(answer) in haystack for answer in answers if answer))


def _identifier_recall(gold: Sequence[str], retrieved: Sequence[str]) -> float | None:
    if not gold:
        return None
    retrieved_set = set(retrieved)
    return len(set(gold) & retrieved_set) / len(set(gold))


def _identifier_mrr(gold: Sequence[str], retrieved: Sequence[str]) -> float | None:
    if not gold:
        return None
    gold_set = set(gold)
    for rank, value in enumerate(retrieved, start=1):
        if value in gold_set:
            return 1.0 / rank
    return 0.0


def _normalize_identifier(value: object) -> str:
    return str(value).strip().rstrip("/").casefold()


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _percentile(values: Iterable[Any], fraction: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    position = int(round((len(numeric) - 1) * fraction))
    return numeric[position]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
