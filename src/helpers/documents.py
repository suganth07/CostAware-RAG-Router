from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.text = str(self.text)
        self.metadata = dict(self.metadata)


@dataclass(slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.document_id = str(self.document_id)
        self.text = str(self.text)
        self.index = int(self.index)
        self.metadata = dict(self.metadata)


DocumentInput = Document | dict[str, Any] | str


def normalize_documents(documents: Iterable[DocumentInput]) -> list[Document]:
    return [_normalize_document(document, index) for index, document in enumerate(documents)]


def _normalize_document(document: DocumentInput, index: int) -> Document:
    if isinstance(document, Document):
        return document
    if isinstance(document, str):
        return Document(id=f"doc:{index}", text=document)
    return Document(
        id=str(document.get("id") or f"doc:{index}"),
        text=str(document.get("text") or ""),
        metadata=dict(document.get("metadata") or {}),
    )
