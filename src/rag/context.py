from __future__ import annotations

from src.rag.types import RetrievalResult


class RAGContextBuilder:
    def __init__(self, *, max_chars: int = 12_000) -> None:
        self.max_chars = max_chars

    def build(self, results: list[RetrievalResult], *, title: str = "Retrieval evidence") -> str:
        if not results:
            return ""

        lines = [f"[{title}]"]
        for index, result in enumerate(results, start=1):
            source = (
                result.metadata.get("document_id")
                or result.metadata.get("source")
                or result.metadata.get("source_document_id")
                or result.id
            )
            lines.append(
                f"{index}. source={source} retriever={result.source} score={result.score:.4f}\n"
                f"{result.text}"
            )

        context = "\n\n".join(lines)
        if len(context) <= self.max_chars:
            return context
        return context[: self.max_chars].rsplit("\n", 1)[0]
