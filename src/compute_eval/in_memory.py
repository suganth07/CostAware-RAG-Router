from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from logger import get_logger
from src.compute_eval.metrics import OperationMetric
from src.compute_eval.noop import NoopComputeTracker


class InMemoryComputeTracker(NoopComputeTracker):
    """Collect timing and count data for comparing RAG variants."""

    def __init__(self, *, log_stages: bool = True, namespace: str = "") -> None:
        self.metrics: list[OperationMetric] = []
        self.counts: dict[str, int] = {}
        self.values: dict[str, Any] = {}
        self.log_stages = log_stages
        self.namespace = namespace
        self._logger = get_logger(__name__)

    @contextmanager
    def track(
        self,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        namespaced_stage = f"{self.namespace}.{stage}" if self.namespace else stage
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            metric = OperationMetric(
                stage=namespaced_stage,
                elapsed_ms=elapsed_ms,
                metadata=dict(metadata or {}),
            )
            self.metrics.append(metric)
            if self.log_stages:
                self._logger.info(
                    "stage=%s elapsed_ms=%.2f metadata=%s",
                    namespaced_stage,
                    elapsed_ms,
                    metric.metadata,
                )

    def add_count(self, name: str, value: int = 1) -> None:
        key = f"{self.namespace}.{name}" if self.namespace else name
        self.counts[key] = self.counts.get(key, 0) + value

    def set_value(self, name: str, value: Any) -> None:
        key = f"{self.namespace}.{name}" if self.namespace else name
        self.values[key] = value

    def snapshot(self) -> dict[str, Any]:
        stage_totals: dict[str, float] = {}
        for metric in self.metrics:
            stage_totals[metric.stage] = stage_totals.get(metric.stage, 0.0) + metric.elapsed_ms

        return {
            "metrics": [
                {
                    "stage": metric.stage,
                    "elapsed_ms": metric.elapsed_ms,
                    "metadata": metric.metadata,
                }
                for metric in self.metrics
            ],
            "counts": dict(self.counts),
            "values": dict(self.values),
            "stage_totals_ms": stage_totals,
        }
