from __future__ import annotations

from typing import Any
import urllib.request

import pytest

from src.adapters.embedding.ollama import OllamaEmbeddingAdapter, OllamaEmbeddingConfig


class RecordingOllamaEmbeddingAdapter(OllamaEmbeddingAdapter):
    def __init__(self, config: OllamaEmbeddingConfig) -> None:
        super().__init__(config)
        self.inputs: list[list[str]] = []

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        batch = list(payload["input"])
        self.inputs.append(batch)
        return {
            "model": payload["model"],
            "embeddings": [[float(len(self.inputs)), float(index)] for index, _ in enumerate(batch)],
            "prompt_eval_count": len(batch),
        }


def test_ollama_embedding_adapter_batches_inputs_in_order() -> None:
    adapter = RecordingOllamaEmbeddingAdapter(
        OllamaEmbeddingConfig(model="test-embedding", batch_size=2)
    )

    result = adapter.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])

    assert adapter.inputs == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert result.vectors == [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
    assert result.model == "test-embedding"
    assert result.raw["batch_count"] == 3
    assert result.usage["prompt_eval_count"] == 5


def test_ollama_embedding_adapter_allows_unbatched_request() -> None:
    adapter = RecordingOllamaEmbeddingAdapter(
        OllamaEmbeddingConfig(model="test-embedding", batch_size=None)
    )

    adapter.embed_texts(["a", "b", "c"])

    assert adapter.inputs == [["a", "b", "c"]]


def test_ollama_embedding_adapter_rejects_invalid_batch_size() -> None:
    adapter = RecordingOllamaEmbeddingAdapter(
        OllamaEmbeddingConfig(model="test-embedding", batch_size=0)
    )

    with pytest.raises(ValueError, match="batch_size"):
        adapter.embed_texts(["a"])


def test_ollama_timeout_error_includes_batch_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)
    adapter = OllamaEmbeddingAdapter(
        OllamaEmbeddingConfig(
            model="test-embedding",
            base_url="http://ollama.test",
            timeout_seconds=0.001,
        )
    )

    with pytest.raises(TimeoutError, match="Lower OllamaEmbeddingConfig.batch_size"):
        adapter._post_json("/api/embed", {"model": "test-embedding", "input": ["a", "b"]})
