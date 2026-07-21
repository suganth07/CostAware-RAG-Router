from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult


@dataclass(slots=True)
class LlamaCppEmbeddingConfig:
    model: str = "nomic-embed-text"
    base_url: str = "http://127.0.0.1:11435"
    timeout_seconds: float = 180.0
    batch_size: int = 256
    max_concurrent_requests: int = 1


class LlamaCppEmbeddingAdapter(EmbeddingAdapter):
    """Batched OpenAI-compatible adapter for a local llama.cpp server."""

    def __init__(self, config: LlamaCppEmbeddingConfig | None = None) -> None:
        self.config = config or LlamaCppEmbeddingConfig()

    def embed_texts(self, texts: Sequence[str]) -> EmbeddingResult:
        values = [str(text) for text in texts]
        if not values:
            return EmbeddingResult(vectors=[], model=self.config.model)
        request_size = max(1, self.config.batch_size)
        batches = [
            values[start : start + request_size]
            for start in range(0, len(values), request_size)
        ]
        workers = min(max(1, self.config.max_concurrent_requests), len(batches))
        if workers == 1:
            responses = [self._embed_batch(batch) for batch in batches]
        else:
            # llama.cpp assigns independent HTTP requests to its shared server
            # slots.  Several medium requests feed the GPU more consistently
            # than one very large request while preserving response order.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                responses = list(executor.map(self._embed_batch, batches))

        vectors: list[list[float]] = []
        prompt_tokens = 0
        for batch_vectors, batch_prompt_tokens in responses:
            vectors.extend(batch_vectors)
            prompt_tokens += batch_prompt_tokens
        return EmbeddingResult(
            vectors=vectors,
            model=self.config.model,
            usage={"prompt_eval_count": prompt_tokens},
            raw={"backend": "llama_cpp", "base_url": self.config.base_url},
        )

    def _embed_batch(self, batch: Sequence[str]) -> tuple[list[list[float]], int]:
        response = self._post(
            "/v1/embeddings",
            {"model": self.config.model, "input": list(batch)},
        )
        data = sorted(
            list(response.get("data") or []),
            key=lambda item: int(item.get("index", 0)),
        )
        batch_vectors = [
            [float(value) for value in item.get("embedding") or []]
            for item in data
        ]
        if len(batch_vectors) != len(batch):
            raise ValueError(
                "llama.cpp returned a different embedding count "
                f"({len(batch_vectors)} != {len(batch)})"
            )
        usage = dict(response.get("usage") or {})
        return batch_vectors, int(usage.get("prompt_tokens") or 0)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(
                f"Timed out calling llama.cpp embeddings at {request.full_url}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Failed to call llama.cpp embeddings at {request.full_url}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError("llama.cpp embedding response must be an object")
        return value
