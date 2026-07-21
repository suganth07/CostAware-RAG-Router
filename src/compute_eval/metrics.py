from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class OperationMetric:
    stage: str
    elapsed_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
