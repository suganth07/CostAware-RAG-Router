from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


class NoopComputeTracker:
    @contextmanager
    def track(
        self,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        yield

    def add_count(self, name: str, value: int = 1) -> None:
        return None

    def set_value(self, name: str, value: Any) -> None:
        return None

    def snapshot(self) -> dict[str, Any]:
        return {"metrics": [], "counts": {}, "values": {}, "stage_totals_ms": {}}
