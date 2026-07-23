"""Persistent prompt/model cache for reproducible local critic calls."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CachedJudgeResponse:
    text: str
    model: str


class CriticResponseCache:
    """SQLite cache keyed by full system prompt, payload, model, and version."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> "CriticResponseCache":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS critic_response_cache (
                cache_key TEXT PRIMARY KEY,
                response_text TEXT NOT NULL,
                response_model TEXT NOT NULL
            )
            """
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def get(self, *, prompt: str, system_prompt: str, model_fingerprint: str, prompt_version: str, stage: str) -> CachedJudgeResponse | None:
        connection = self._require_connection()
        key = self.key(prompt=prompt, system_prompt=system_prompt, model_fingerprint=model_fingerprint, prompt_version=prompt_version, stage=stage)
        row = connection.execute(
            "SELECT response_text, response_model FROM critic_response_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return CachedJudgeResponse(text=str(row[0]), model=str(row[1])) if row else None

    def put(self, *, prompt: str, system_prompt: str, model_fingerprint: str, prompt_version: str, stage: str, response: CachedJudgeResponse) -> None:
        connection = self._require_connection()
        key = self.key(prompt=prompt, system_prompt=system_prompt, model_fingerprint=model_fingerprint, prompt_version=prompt_version, stage=stage)
        connection.execute(
            "INSERT OR REPLACE INTO critic_response_cache(cache_key, response_text, response_model) VALUES (?, ?, ?)",
            (key, response.text, response.model),
        )
        connection.commit()

    @staticmethod
    def key(*, prompt: str, system_prompt: str, model_fingerprint: str, prompt_version: str, stage: str) -> str:
        payload = "\x1f".join((prompt, system_prompt, model_fingerprint, prompt_version, stage))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("CriticResponseCache must be used as a context manager")
        return self._connection
