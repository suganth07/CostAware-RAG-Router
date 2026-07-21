from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.datasets import get_dataset_adapter


def main() -> None:
    multihop = get_dataset_adapter("multihop_rag").load(
        limit_documents=2,
        limit_eval_samples=2,
    )
    assert multihop.documents
    assert multihop.eval_samples
    assert multihop.documents[0].text
    assert multihop.eval_samples[0].question

    wiki = get_dataset_adapter("2wiki", split="validation").load(
        limit_documents=2,
        limit_eval_samples=2,
    )
    assert wiki.documents
    assert wiki.eval_samples
    assert wiki.documents[0].text
    assert wiki.eval_samples[0].answers

    metaqa = get_dataset_adapter(
        "metaqa",
        hop="1-hop",
        variant="vanilla",
        split="train",
    ).load(limit_documents=3, limit_eval_samples=2)
    assert metaqa.documents
    assert metaqa.eval_samples
    assert metaqa.graph_nodes
    assert metaqa.graph_edges


if __name__ == "__main__":
    main()
