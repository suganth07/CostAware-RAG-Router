from __future__ import annotations

from src.router.candidates.base import PipelineCandidateAdapter


class GraphRAGCandidate(PipelineCandidateAdapter):
    def __init__(self, pipeline, *, fingerprint: dict) -> None:
        super().__init__("graph_rag", pipeline, fingerprint=fingerprint)
