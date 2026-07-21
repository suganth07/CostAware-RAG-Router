from __future__ import annotations

from collections.abc import Iterable

from src.helpers.documents import Chunk, Document


class TextChunker:
    """Reusable text chunker for RAG, GraphRAG, and later uploaded files."""

    def __init__(self, *, chunk_size: int = 900, overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                sentence_boundary = max(
                    text.rfind(".", start, end),
                    text.rfind("?", start, end),
                    text.rfind("!", start, end),
                    text.rfind("\n", start, end),
                )
                if sentence_boundary > start + (self.chunk_size // 2):
                    end = sentence_boundary + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:chunk:{chunk_index}",
                        document_id=document.id,
                        text=chunk_text,
                        index=chunk_index,
                        metadata=dict(document.metadata),
                    )
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = max(0, end - self.overlap)

        return chunks
