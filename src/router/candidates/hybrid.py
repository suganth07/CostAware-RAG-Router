from __future__ import annotations

from src.router.candidates.base import PipelineCandidateAdapter


class HybridRAGCandidate(PipelineCandidateAdapter):
    def __init__(self, pipeline, *, fingerprint: dict) -> None:
        super().__init__("hybrid_rag", pipeline, fingerprint=fingerprint)
