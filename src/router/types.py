"""Typed, JSON-safe data contracts shared by collection, scoring, and training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any

from src.evaluation.core import EvidenceUnit, RetrievalRecord
from src.router.constants import ACTION_IDS


@dataclass(slots=True)
class QuestionRecord:
    question_id: str
    dataset: str
    source_split: str
    question: str
    accepted_answers: list[str]
    gold_evidence: list[dict[str, Any]] = field(default_factory=list)
    answer_type: str = "span"
    sampling_strata: dict[str, Any] = field(default_factory=dict)
    group_keys: dict[str, str] = field(default_factory=dict)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CandidateOutcome:
    question_id: str
    candidate_id: str
    repeat_id: int
    run_id: str
    pipeline_fingerprint: str
    corpus_fingerprint: str
    model_fingerprint: str
    hardware_fingerprint: str
    retrieval_record: RetrievalRecord
    generated_answer: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    end_to_end_latency_ms: float | None = None
    peak_cpu_memory_mb: float | None = None
    peak_gpu_memory_mb: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    timeout: bool = False
    error_type: str | None = None
    error_message_hash: str | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if self.candidate_id not in ACTION_IDS:
            raise ValueError(f"Unknown candidate_id: {self.candidate_id}")
        if self.repeat_id < 0:
            raise ValueError("repeat_id cannot be negative")

    @property
    def key(self) -> tuple[str, str, int, str]:
        return (self.question_id, self.candidate_id, self.repeat_id, self.run_id)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieval_record"] = self.retrieval_record.to_dict()
        # Hybrid RAG retains its constituent RAGContext objects in diagnostic
        # metadata.  Raw outcomes must remain JSONL-safe and resumable even
        # when a candidate exposes rich Python objects in `usage` or `details`.
        return _json_safe(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CandidateOutcome":
        copied = dict(value)
        copied["retrieval_record"] = RetrievalRecord.from_dict(copied["retrieval_record"])
        return cls(**copied)


@dataclass(slots=True)
class MetricVector:
    exact_or_alias_match: float | None = None
    token_or_set_f1: float | None = None
    boolean_accuracy: float | None = None
    numeric_date_accuracy: float | None = None
    semantic_equivalence: float | None = None
    embedding_similarity_raw: float | None = None
    embedding_similarity_calibrated: float | None = None
    contradiction_probability: float | None = None
    entity_conflict: bool = False
    numeric_conflict: bool = False
    polarity_conflict: bool = False
    supporting_evidence_recall: float | None = None
    context_precision: float | None = None
    evidence_ndcg_at_k: float | None = None
    evidence_mrr: float | None = None
    claim_faithfulness: float | None = None
    answer_completeness: float | None = None
    answer_relevance: float | None = None
    correct_refusal: float | None = None
    failure_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoredOutcome:
    outcome: CandidateOutcome
    metrics: MetricVector
    quality: float | None
    compute_units: float | None
    feasible: bool | None = None
    infeasibility: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactManifest:
    schema_version: int
    artifact_kind: str
    fingerprint: str
    created_at_utc: str
    row_count: int
    actions: tuple[str, ...]
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_safe(value: Any) -> Any:
    """Convert diagnostics to stable JSON primitives without losing structure."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    # Keep collection alive for non-essential diagnostics, while recording the
    # original type so a new unexpected object remains visible in artifacts.
    return {"_unserializable_type": f"{type(value).__module__}.{type(value).__qualname__}", "_repr": repr(value)}
