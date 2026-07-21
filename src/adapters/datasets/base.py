from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from src.adapters.storage.graph_store import GraphEdge, GraphNode, GraphStore
from src.helpers import Document


@dataclass(slots=True)
class EvaluationSample:
    id: str
    question: str
    answers: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.question = str(self.question)
        self.answers = [str(answer) for answer in self.answers]
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class DatasetBundle:
    name: str
    documents: list[Document] = field(default_factory=list)
    eval_samples: list[EvaluationSample] = field(default_factory=list)
    graph_nodes: list[GraphNode] = field(default_factory=list)
    graph_edges: list[GraphEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def seed_graph_store(self, graph_store: GraphStore) -> None:
        if self.graph_nodes:
            graph_store.upsert_nodes(self.graph_nodes)
        if self.graph_edges:
            graph_store.upsert_edges(self.graph_edges)


class DatasetAdapter(ABC):
    @abstractmethod
    def load(
        self,
        *,
        limit_documents: Optional[int] = None,
        limit_eval_samples: Optional[int] = None,
    ) -> DatasetBundle:
        raise NotImplementedError
