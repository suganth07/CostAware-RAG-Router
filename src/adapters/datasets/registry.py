from __future__ import annotations

from typing import Any

from src.adapters.datasets.base import DatasetAdapter
from src.adapters.datasets.metaqa import MetaQADatasetAdapter, MetaQADatasetConfig
from src.adapters.datasets.multihop_rag import MultiHopRAGDatasetAdapter, MultiHopRAGDatasetConfig
from src.adapters.datasets.wiki_multihop_qa import WikiMultiHopQAAdapter, WikiMultiHopQAConfig


def get_dataset_adapter(name: str, **kwargs: Any) -> DatasetAdapter:
    normalized = name.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"multihop_rag", "multihoprag"}:
        return MultiHopRAGDatasetAdapter(MultiHopRAGDatasetConfig(**kwargs))
    if normalized in {"2wiki", "2wiki_multihop_qa", "wiki_multihop_qa"}:
        return WikiMultiHopQAAdapter(WikiMultiHopQAConfig(**kwargs))
    if normalized == "metaqa":
        return MetaQADatasetAdapter(MetaQADatasetConfig(**kwargs))
    raise ValueError(f"Unknown dataset adapter: {name}")
