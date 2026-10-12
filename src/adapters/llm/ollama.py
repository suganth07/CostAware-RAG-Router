from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from logger import get_logger
from src.adapters.llm.adapter import ChatMessage, LLM, LLMResponse


@dataclass(slots=True)
class OllamaLLMConfig:
    model: str
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    temperature: Optional[float] = None
    num_predict: Optional[int] = None
    keep_alive: Optional[str] = None
    default_options: dict[str, Any] = field(default_factory=dict)


class OllamaLLM(LLM):
    """Ollama text-generation adapter using the local HTTP API."""

    def __init__(self, config: OllamaLLMConfig) -> None:
        self.config = config
        self._logger = get_logger(__name__)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(kwargs),
        }
        if system_prompt:
            payload["system"] = system_prompt
        if self.config.keep_alive:
            payload["keep_alive"] = self.config.keep_alive

        raw = self._post_json("/api/generate", payload)
        return LLMResponse(
            text=str(raw.get("response") or ""),
            model=str(raw.get("model") or self.config.model),
            raw=raw,
            usage=_usage_from_ollama(raw),
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.as_dict() for message in messages],
            "stream": False,
            "options": self._options(kwargs),
        }
        if self.config.keep_alive:
            payload["keep_alive"] = self.config.keep_alive

        raw = self._post_json("/api/chat", payload)
        message = raw.get("message") or {}
        return LLMResponse(
            text=str(message.get("content") or ""),
            model=str(raw.get("model") or self.config.model),
            raw=raw,
            usage=_usage_from_ollama(raw),
        )

    def _options(self, overrides: dict[str, Any]) -> dict[str, Any]:
        options = dict(self.config.default_options)
        if self.config.temperature is not None:
            options["temperature"] = self.config.temperature
        if self.config.num_predict is not None:
            options["num_predict"] = self.config.num_predict
        options.update(overrides.pop("options", {}) or {})
        options.update(overrides)
        return options

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
            raise TimeoutError(
                "Timed out waiting for Ollama to generate a response "
                f"after {self.config.timeout_seconds}s using model={self.config.model!r}."
            ) from exc
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            detail_suffix = f" Response: {details}" if details else ""
            raise ConnectionError(
                f"Ollama returned HTTP {exc.code} for {url}.{detail_suffix}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Failed to call Ollama at {url}: {exc}") from exc


def _usage_from_ollama(raw: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    return {key: raw[key] for key in keys if key in raw}
