from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CapsuleScoreWeights:
    text: float = 0.35
    entity: float = 0.25
    structural: float = 0.25
    source: float = 0.10
    proximity: float = 0.05
