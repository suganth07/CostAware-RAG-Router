from __future__ import annotations

from pathlib import Path
from typing import Any


def load_dataset_split(path: str | Path, split: str) -> Any:
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset adapters require the 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found in dataset at {path}")
        return dataset[split]
    if split not in {"", "train"}:
        raise KeyError(f"Dataset at {path} has no split map; requested split '{split}'")
    return dataset
