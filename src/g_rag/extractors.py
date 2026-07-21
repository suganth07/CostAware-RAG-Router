from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from logger import get_logger
from src.adapters.llm import LLM
from src.adapters.storage.graph_store import GraphEdge, GraphNode
from src.helpers.documents import Chunk

_BRACKET_ENTITY_RE = re.compile(r"\[([^\[\]]{1,120})\]")
_ENTITY_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9&'-]*)(?:\s+[A-Z][A-Za-z0-9&'-]*){0,4}\b")
_NON_ENTITY = {
    "The", "A", "An", "This", "That", "It", "He", "She", "They", "We", "I",
    "Who", "Whom", "Whose", "Which", "What", "Where", "When", "Why", "How",
    "Do", "Does", "Did", "Is", "Are", "Was", "Were", "Can", "Could", "Would", "Should",
}


@dataclass(slots=True)
class GraphTriple:
    source_id: str
    target_id: str
    type: str = "RELATED_TO"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionResult:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)


class EntityRelationExtractor(ABC):
    @abstractmethod
    def extract(self, chunk: Chunk) -> ExtractionResult:
        raise NotImplementedError


class SimpleEntityRelationExtractor(EntityRelationExtractor):
    """Dependency-free extractor for initial experiments and smoke tests."""

    def extract(self, chunk: Chunk) -> ExtractionResult:
        entity_names = _ordered_unique(
            [
                *(
                    match.group(1).strip()
                    for match in _BRACKET_ENTITY_RE.finditer(chunk.text)
                    if match.group(1).strip()
                ),
                *(
                    match.group(0).strip()
                    for match in _ENTITY_RE.finditer(chunk.text)
                    if match.group(0).strip() not in _NON_ENTITY
                ),
            ]
        )
        nodes = [
            GraphNode(
                id=_entity_id(name),
                labels=("Entity",),
                properties={
                    "name": name,
                    "source_document_id": chunk.document_id,
                    "source_chunk_id": chunk.id,
                },
            )
            for name in entity_names
        ]

        edges: list[GraphEdge] = []
        for left, right in zip(nodes, nodes[1:]):
            edge_id = f"{chunk.id}:co_occurs:{left.id}:{right.id}"
            edges.append(
                GraphEdge(
                    id=edge_id,
                    source_id=left.id,
                    target_id=right.id,
                    type="CO_OCCURS_WITH",
                    properties={
                        "source_document_id": chunk.document_id,
                        "source_chunk_id": chunk.id,
                    },
                )
            )

        return ExtractionResult(nodes=nodes, edges=edges)


class LLMEntityRelationExtractor(EntityRelationExtractor):
    """LLM-backed extractor for stronger graph construction."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self._logger = get_logger(__name__)

    def extract(self, chunk: Chunk) -> ExtractionResult:
        prompt = (
            "Extract a compact knowledge graph from the text. Return only JSON with this shape: "
            '{"entities":[{"id":"canonical_id","name":"Name","labels":["Entity"],"properties":{}}],'
            '"relations":[{"source_id":"canonical_id","target_id":"canonical_id","type":"RELATION","properties":{}}]}'
            "\n\nText:\n"
            f"{chunk.text}"
        )
        response = self.llm.generate(prompt)
        payload = _load_json_object(response.text)

        nodes = [
            GraphNode(
                id=_entity_id(str(entity.get("id") or entity.get("name"))),
                labels=tuple(entity.get("labels") or ("Entity",)),
                properties={
                    "name": entity.get("name") or entity.get("id"),
                    "source_document_id": chunk.document_id,
                    "source_chunk_id": chunk.id,
                    **dict(entity.get("properties") or {}),
                },
            )
            for entity in payload.get("entities", [])
            if entity.get("id") or entity.get("name")
        ]
        node_ids = {node.id for node in nodes}

        edges: list[GraphEdge] = []
        for relation in payload.get("relations", []):
            source_id = _entity_id(str(relation.get("source_id") or ""))
            target_id = _entity_id(str(relation.get("target_id") or ""))
            if not source_id or not target_id:
                continue
            if source_id not in node_ids:
                nodes.append(GraphNode(id=source_id, properties={"source_chunk_id": chunk.id}))
                node_ids.add(source_id)
            if target_id not in node_ids:
                nodes.append(GraphNode(id=target_id, properties={"source_chunk_id": chunk.id}))
                node_ids.add(target_id)

            relation_type = str(relation.get("type") or "RELATED_TO").upper()
            edges.append(
                GraphEdge(
                    id=f"{chunk.id}:{source_id}:{relation_type}:{target_id}",
                    source_id=source_id,
                    target_id=target_id,
                    type=relation_type,
                    properties={
                        "source_document_id": chunk.document_id,
                        "source_chunk_id": chunk.id,
                        **dict(relation.get("properties") or {}),
                    },
                )
            )

        return ExtractionResult(nodes=nodes, edges=edges)


def _ordered_unique(values: list[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        result.append(normalized)
    return result


def _entity_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized


def _load_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Extractor response must be a JSON object")
    return parsed
