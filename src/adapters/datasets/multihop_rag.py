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
class MultiHopRAGDatasetConfig:
    root: Path = Path("data/datasets/MultiHop-RAG")
    split: str = "train"


class MultiHopRAGDatasetAdapter(DatasetAdapter):
    """Adapter for yixuantt/MultiHopRAG saved with load_from_disk."""

    def __init__(self, config: MultiHopRAGDatasetConfig | None = None) -> None:
        self.config = config or MultiHopRAGDatasetConfig()
        self._logger = get_logger(__name__)

    def load(
        self,
        *,
        limit_documents: Optional[int] = None,
        limit_eval_samples: Optional[int] = None,
    ) -> DatasetBundle:
        root = resolve_dataset_root(self.config.root)
        corpus = load_dataset_split(root / "corpus", self.config.split)
        qa = load_dataset_split(root / "qa", self.config.split)

        documents = []
        for index, row in enumerate(_limited(corpus, limit_documents)):
            documents.append(
                Document(
                    id=str(row.get("url") or f"multihop-rag:doc:{index}"),
                    text=_join_present(row.get("title"), row.get("body")),
                    metadata={
                        "dataset": "MultiHop-RAG",
                        "category": row.get("category"),
                        "author": row.get("author"),
                        "published_at": row.get("published_at"),
                        "url": row.get("url"),
                        "source": row.get("source"),
                    },
                )
            )

        eval_samples = []
        for index, row in enumerate(_limited(qa, limit_eval_samples)):
            eval_samples.append(
                EvaluationSample(
                    id=f"multihop-rag:{self.config.split}:{index}",
                    question=str(row.get("query") or ""),
                    answers=[str(row.get("answer") or "")],
                    metadata={
                        "dataset": "MultiHop-RAG",
                        "question_type": row.get("question_type"),
                        "evidence_count": len(row.get("evidence_list") or []),
                        "evidence_list": row.get("evidence_list") or [],
                    },
                )
            )

        self._logger.info(
            "loaded MultiHop-RAG documents=%s eval_samples=%s",
            len(documents),
            len(eval_samples),
        )
        return DatasetBundle(
            name="MultiHop-RAG",
            documents=documents,
            eval_samples=eval_samples,
            metadata={"split": self.config.split, "root": str(root)},
        )


def _limited(dataset, limit: Optional[int]):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _join_present(*parts: object) -> str:
    return "\n\n".join(str(part).strip() for part in parts if part)
