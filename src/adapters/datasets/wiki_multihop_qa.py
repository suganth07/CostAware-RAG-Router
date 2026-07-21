from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logger import get_logger
from src.adapters.datasets.base import DatasetAdapter, DatasetBundle, EvaluationSample
from src.adapters.datasets.hf_disk import load_dataset_split
from src.adapters.datasets.paths import resolve_dataset_root
from src.helpers import Document


@dataclass(slots=True)
class WikiMultiHopQAConfig:
    root: Path = Path("data/datasets/2WikiMultiHopQA")
    split: str = "train"


class WikiMultiHopQAAdapter(DatasetAdapter):
    """Adapter for the local 2WikiMultiHopQA dataset."""

    def __init__(self, config: WikiMultiHopQAConfig | None = None) -> None:
        self.config = config or WikiMultiHopQAConfig()
        self._logger = get_logger(__name__)

    def load(
        self,
        *,
        limit_documents: Optional[int] = None,
        limit_eval_samples: Optional[int] = None,
    ) -> DatasetBundle:
        root = resolve_dataset_root(self.config.root)
        dataset = load_dataset_split(root, self.config.split)
        document_limit = limit_documents if limit_documents is not None else limit_eval_samples
        rows_for_documents = _limited(dataset, document_limit)
        rows_for_eval = _limited(dataset, limit_eval_samples)

        documents = [
            Document(
                id=str(row.get("id") or f"2wiki:doc:{index}"),
                text=_context_to_text(row.get("context") or {}),
                metadata={
                    "dataset": "2WikiMultiHopQA",
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "type": row.get("type"),
                },
            )
            for index, row in enumerate(rows_for_documents)
        ]

        eval_samples = [
            EvaluationSample(
                id=str(row.get("id") or f"2wiki:{self.config.split}:{index}"),
                question=str(row.get("question") or ""),
                answers=[str(row.get("answer") or "")],
                metadata={
                    "dataset": "2WikiMultiHopQA",
                    "type": row.get("type"),
                    "supporting_facts": row.get("supporting_facts") or {},
                    "evidences": row.get("evidences") or [],
                },
            )
            for index, row in enumerate(rows_for_eval)
        ]

        self._logger.info(
            "loaded 2WikiMultiHopQA documents=%s eval_samples=%s",
            len(documents),
            len(eval_samples),
        )
        return DatasetBundle(
            name="2WikiMultiHopQA",
            documents=documents,
            eval_samples=eval_samples,
            metadata={"split": self.config.split, "root": str(root)},
        )


def _limited(dataset, limit: Optional[int]):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _context_to_text(context: dict) -> str:
    titles = context.get("title") or []
    sentences_by_title = context.get("sentences") or []
    sections = []
    for title, sentences in zip(titles, sentences_by_title):
        sections.append(f"{title}\n{' '.join(sentences)}")
    return "\n\n".join(sections)
