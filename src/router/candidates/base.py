"""Candidate boundary: policy code never sees concrete pipeline result classes."""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

from src.adapters.datasets.base import EvaluationSample
from src.evaluation.runner import retrieval_record_from_output
from src.router.types import CandidateOutcome, QuestionRecord


class CandidateAdapter(ABC):
    candidate_id: str

    @abstractmethod
    def retrieve(self, question: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def answer(self, question: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def fingerprint_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def execute(
        self,
        *,
        record: QuestionRecord,
        sample: EvaluationSample,
        repeat_id: int,
        run_id: str,
        corpus_fingerprint: str,
        model_fingerprint: str,
        hardware_fingerprint: str,
        run_generation: bool,
    ) -> CandidateOutcome:
        started = time.perf_counter()
        try:
            output = self.answer(record.question) if run_generation else self.retrieve(record.question)
            total_ms = (time.perf_counter() - started) * 1000.0
            normalized = retrieval_record_from_output(
                system=self.candidate_id,
                sample=sample,
                output=output,
                latency_seconds=total_ms / 1000.0,
            )
            usage = dict(getattr(output, "usage", {}) or {})
            snapshot = _instrumentation_snapshot(usage)
            retrieval_ms, generation_ms = _stage_latencies(snapshot)
            return CandidateOutcome(
                question_id=record.question_id,
                candidate_id=self.candidate_id,
                repeat_id=repeat_id,
                run_id=run_id,
                pipeline_fingerprint=_fingerprint(self.fingerprint_payload()),
                corpus_fingerprint=corpus_fingerprint,
                model_fingerprint=model_fingerprint,
                hardware_fingerprint=hardware_fingerprint,
                retrieval_record=normalized,
                generated_answer=normalized.generated_answer,
                usage=usage,
                retrieval_latency_ms=retrieval_ms,
                generation_latency_ms=generation_ms,
                end_to_end_latency_ms=total_ms,
                input_tokens=_token_count(usage, "input"),
                output_tokens=_token_count(usage, "output"),
            )
        except Exception as exc:
            total_ms = (time.perf_counter() - started) * 1000.0
            empty = retrieval_record_from_output(
                system=self.candidate_id,
                sample=sample,
                output=type("EmptyOutput", (), {"context_text": "", "details": {}})(),
                latency_seconds=total_ms / 1000.0,
            )
            return CandidateOutcome(
                question_id=record.question_id,
                candidate_id=self.candidate_id,
                repeat_id=repeat_id,
                run_id=run_id,
                pipeline_fingerprint=_fingerprint(self.fingerprint_payload()),
                corpus_fingerprint=corpus_fingerprint,
                model_fingerprint=model_fingerprint,
                hardware_fingerprint=hardware_fingerprint,
                retrieval_record=empty,
                end_to_end_latency_ms=total_ms,
                error_type=type(exc).__name__,
                error_message_hash=hashlib.sha256(str(exc).encode("utf-8", errors="replace")).hexdigest(),
            )


class PipelineCandidateAdapter(CandidateAdapter):
    """Thin adapter for existing pipeline objects with `retrieve` and `answer`."""

    def __init__(self, candidate_id: str, pipeline: Any, *, fingerprint: dict[str, Any]) -> None:
        self.candidate_id = candidate_id
        self.pipeline = pipeline
        self._fingerprint = dict(fingerprint)

    def retrieve(self, question: str) -> Any:
        return self.pipeline.retrieve(question)

    def answer(self, question: str) -> Any:
        return self.pipeline.answer(question)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, **self._fingerprint}


def _fingerprint(value: dict[str, Any]) -> str:
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _instrumentation_snapshot(usage: dict[str, Any]) -> dict[str, Any]:
    for value in usage.values():
        if isinstance(value, dict) and "stage_totals_ms" in value:
            return value
    return {}


def _stage_latencies(snapshot: dict[str, Any]) -> tuple[float | None, float | None]:
    totals = dict(snapshot.get("stage_totals_ms") or {})
    if not totals:
        return None, None
    generation = sum(float(value) for stage, value in totals.items() if "generate" in stage)
    return max(0.0, sum(float(value) for value in totals.values()) - generation), generation


def _token_count(usage: dict[str, Any], kind: str) -> int | None:
    keys = (f"{kind}_tokens", f"{kind}_token_count")
    for value in usage.values():
        if isinstance(value, dict):
            for key in keys:
                if value.get(key) is not None:
                    return int(value[key])
    return None
