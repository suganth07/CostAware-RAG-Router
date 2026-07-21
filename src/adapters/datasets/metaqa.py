from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from logger import get_logger
from src.adapters.datasets.base import DatasetAdapter, DatasetBundle, EvaluationSample
from src.adapters.datasets.paths import resolve_dataset_root
from src.adapters.storage.graph_store import GraphEdge, GraphNode
from src.helpers import Document


@dataclass(slots=True)
class MetaQADatasetConfig:
    root: Path = Path("data/datasets/metaqa")
    hop: str = "1-hop"
    variant: str = "vanilla"
    split: str = "train"


class MetaQADatasetAdapter(DatasetAdapter):
    """Adapter for MetaQA text QA and movie knowledge-base triples."""

    def __init__(self, config: MetaQADatasetConfig | None = None) -> None:
        self.config = config or MetaQADatasetConfig()
        self._logger = get_logger(__name__)

    def load(
        self,
        *,
        limit_documents: Optional[int] = None,
        limit_eval_samples: Optional[int] = None,
    ) -> DatasetBundle:
        root = resolve_dataset_root(self.config.root)
        kb_root = _resolve_metaqa_kb_root(root)
        triples = _read_kb(kb_root / "kb.txt", limit_documents)
        qtype_path = _find_metaqa_file(
            root,
            self.config.hop,
            f"qa_{self.config.split}_qtype.txt",
        )
        qtypes = _read_optional_lines(qtype_path) if qtype_path else []
        qa_path = _find_metaqa_file(
            root,
            self.config.hop,
            self.config.variant,
            f"qa_{self.config.split}.txt",
        )
        if qa_path is None:
            raise FileNotFoundError(
                "Could not find MetaQA QA file for "
                f"hop={self.config.hop}, variant={self.config.variant}, split={self.config.split}"
            )

        documents = []
        graph_nodes: dict[str, GraphNode] = {}
        graph_edges = []
        for index, (subject, relation, obj) in enumerate(triples):
            subject_id = _entity_id(subject)
            object_id = _entity_id(obj)
            graph_nodes[subject_id] = GraphNode(
                id=subject_id,
                labels=("MovieOrEntity",),
                properties={"name": subject, "dataset": "MetaQA"},
            )
            graph_nodes[object_id] = GraphNode(
                id=object_id,
                labels=("MovieOrEntity",),
                properties={"name": obj, "dataset": "MetaQA"},
            )
            graph_edges.append(
                GraphEdge(
                    id=f"metaqa:{index}:{subject_id}:{relation}:{object_id}",
                    source_id=subject_id,
                    target_id=object_id,
                    type=_relation_id(relation),
                    properties={"dataset": "MetaQA", "relation": relation},
                )
            )
            documents.append(
                Document(
                    id=f"metaqa:kb:{index}",
                    text=f"{subject} {relation.replace('_', ' ')} {obj}.",
                    metadata={
                        "dataset": "MetaQA",
                        "subject": subject,
                        "relation": relation,
                        "object": obj,
                    },
                )
            )

        eval_samples = []
        for index, line in enumerate(_read_limited_lines(qa_path, limit_eval_samples)):
            if "\t" not in line:
                continue
            question, answers = line.split("\t", 1)
            eval_samples.append(
                EvaluationSample(
                    id=f"metaqa:{self.config.hop}:{self.config.variant}:{self.config.split}:{index}",
                    question=question,
                    answers=[answer for answer in answers.split("|") if answer],
                    metadata={
                        "dataset": "MetaQA",
                        "hop": self.config.hop,
                        "variant": self.config.variant,
                        "split": self.config.split,
                        "question_type": qtypes[index] if index < len(qtypes) else None,
                    },
                )
            )

        self._logger.info(
            "loaded MetaQA documents=%s eval_samples=%s graph_nodes=%s graph_edges=%s",
            len(documents),
            len(eval_samples),
            len(graph_nodes),
            len(graph_edges),
        )
        return DatasetBundle(
            name="MetaQA",
            documents=documents,
            eval_samples=eval_samples,
            graph_nodes=list(graph_nodes.values()),
            graph_edges=graph_edges,
            metadata={
                "root": str(root),
                "kb_root": str(kb_root),
                "qa_path": str(qa_path),
                "hop": self.config.hop,
                "variant": self.config.variant,
                "split": self.config.split,
            },
        )


def _resolve_metaqa_kb_root(root: Path) -> Path:
    root = Path(root)
    if (root / "kb.txt").exists():
        return root
    matches = sorted(root.glob("**/MetaQA/kb.txt"))
    if not matches:
        raise FileNotFoundError(f"Could not find MetaQA kb.txt under {root}")
    return matches[0].parent


def _find_metaqa_file(root: Path, *parts: str) -> Optional[Path]:
    root = Path(root)
    direct = root.joinpath(*parts)
    if direct.exists():
        return direct

    expected_suffix = Path(*parts).parts
    matches = sorted(
        candidate
        for candidate in root.rglob(parts[-1])
        if candidate.is_file() and candidate.parts[-len(expected_suffix) :] == expected_suffix
    )
    return matches[0] if matches else None


def _read_kb(path: Path, limit: Optional[int]) -> list[tuple[str, str, str]]:
    triples = []
    for line in _read_limited_lines(path, limit):
        parts = line.split("|")
        if len(parts) != 3:
            continue
        triples.append((parts[0], parts[1], parts[2]))
    return triples


def _read_limited_lines(path: Path, limit: Optional[int]) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
            if limit is not None and len(lines) >= limit:
                break
    return lines


def _read_optional_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return _read_limited_lines(path, None)


def _entity_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")


def _relation_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value.strip().upper()).strip("_")
