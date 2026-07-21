from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


class ComputeTracker(Protocol):
    def track(
        self,
        stage: str,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[None]:
        raise NotImplementedError

    def add_count(self, name: str, value: int = 1) -> None:
        raise NotImplementedError

    def set_value(self, name: str, value: Any) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError
