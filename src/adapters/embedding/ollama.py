from __future__ import annotations

from collections.abc import Iterable
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Sequence

from logger import get_logger
from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult


@dataclass(slots=True)
class OllamaEmbeddingConfig:
    model: str = "embeddinggemma"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    keep_alive: str | None = None
    truncate: bool | None = None
    options: dict[str, Any] = field(default_factory=dict)
    batch_size: int | None = 32


class OllamaEmbeddingAdapter(EmbeddingAdapter):
    """Ollama embedding adapter using the current /api/embed endpoint."""

    def __init__(self, config: OllamaEmbeddingConfig) -> None:
        self.config = config
        self._logger = get_logger(__name__)

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(vectors=[], model=self.config.model)

        input_texts = list(texts)
        vectors: list[list[float]] = []
        responses: list[dict[str, Any]] = []

        batch_size = self.config.batch_size or len(input_texts)
        batch_count = (len(input_texts) + batch_size - 1) // batch_size
        started = time.perf_counter()
        self._logger.info(
            "embedding request started model=%s texts=%s batches=%s batch_size=%s",
            self.config.model,
            len(input_texts),
            batch_count,
            batch_size,
        )
        for batch_index, batch in enumerate(self._batches(input_texts), start=1):
            raw = self._post_json("/api/embed", self._embedding_payload(batch))
            batch_vectors = _vectors_from_ollama(raw)
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"Ollama returned {len(batch_vectors)} embeddings for {len(batch)} input texts"
                )
            vectors.extend(batch_vectors)
            responses.append(raw)
            if batch_index == 1 or batch_index == batch_count or batch_index % 100 == 0:
                self._logger.info(
                    "embedding request progress model=%s batches=%s/%s vectors=%s elapsed_s=%.2f",
                    self.config.model,
                    batch_index,
                    batch_count,
                    len(vectors),
                    time.perf_counter() - started,
                )

        raw_result = _merge_raw_responses(responses, self.config.model)

        return EmbeddingResult(
            vectors=vectors,
            model=str(raw_result.get("model") or self.config.model),
            raw=raw_result,
            usage=_merge_usage(responses),
        )

    def _embedding_payload(self, texts: Sequence[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": list(texts),
        }
        if self.config.keep_alive is not None:
            payload["keep_alive"] = self.config.keep_alive
        if self.config.truncate is not None:
            payload["truncate"] = self.config.truncate
        if self.config.options:
            payload["options"] = self.config.options
        return payload

    def _batches(self, texts: Sequence[str]) -> Iterable[list[str]]:
        batch_size = self.config.batch_size
        if batch_size is None:
            yield list(texts)
            return
        if batch_size < 1:
            raise ValueError("Ollama embedding batch_size must be at least 1 or None")
        for start in range(0, len(texts), batch_size):
            yield list(texts[start : start + batch_size])

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}{path}"
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            input_count = len(payload.get("input") or [])
            raise TimeoutError(
                "Timed out calling Ollama "
                f"model={self.config.model!r} endpoint={url!r} "
                f"after {self.config.timeout_seconds}s for {input_count} input texts. "
                "Lower OllamaEmbeddingConfig.batch_size or increase timeout_seconds."
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Failed to call Ollama at {url}: {exc}") from exc


def _vectors_from_ollama(raw: dict[str, Any]) -> list[list[float]]:
    if "embeddings" in raw:
        return [[float(value) for value in vector] for vector in raw["embeddings"]]
    if "embedding" in raw:
        return [[float(value) for value in raw["embedding"]]]
    raise ValueError("Ollama embedding response did not include 'embeddings'")


def _usage_from_ollama(raw: dict[str, Any]) -> dict[str, Any]:
    keys = ("total_duration", "load_duration", "prompt_eval_count")
    return {key: raw[key] for key in keys if key in raw}


def _merge_raw_responses(responses: Sequence[dict[str, Any]], fallback_model: str) -> dict[str, Any]:
    if not responses:
        return {"model": fallback_model}
    if len(responses) == 1:
        return responses[0]
    return {
        "model": responses[0].get("model") or fallback_model,
        "batch_count": len(responses),
        "batches": responses,
    }


def _merge_usage(responses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for response in responses:
        for key, value in _usage_from_ollama(response).items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
    return merged
