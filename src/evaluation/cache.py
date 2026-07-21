from __future__ import annotations

import json
import re
from pathlib import Path

from src.evaluation.core import EVALUATION_SCHEMA_VERSION, RetrievalRecord


class EvaluationCache:
    """Atomic JSONL cache for expensive retrieval and generation records."""

    def __init__(
        self,
        *,
        root: str | Path = "artifacts/cache/evaluations",
    ) -> None:
        self.root = Path(root)

    def load(self, namespace: str, fingerprint: str) -> dict[str, RetrievalRecord]:
        path = self.path(namespace, fingerprint)
        if not path.exists():
            return {}
        result: dict[str, RetrievalRecord] = {}
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    # A process killed during an append can leave only the final
                    # line incomplete; earlier checkpoints remain usable.
                    continue
                if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
                    continue
                key = str(payload.get("key") or "")
                if key:
                    result[key] = RetrievalRecord.from_dict(payload["record"])
        return result

    def save(
        self,
        namespace: str,
        fingerprint: str,
        records: dict[str, RetrievalRecord],
    ) -> Path:
        path = self.path(namespace, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for key in sorted(records):
                file.write(
                    json.dumps(
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "key": key,
                            "record": records[key].to_dict(),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        temporary.replace(path)
        return path

    def append(
        self,
        namespace: str,
        fingerprint: str,
        records: dict[str, RetrievalRecord],
    ) -> Path:
        """Append a checkpoint batch; `load` keeps the last value per key."""
        path = self.path(namespace, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as file:
            for key in sorted(records):
                file.write(
                    json.dumps(
                        {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "key": key,
                            "record": records[key].to_dict(),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return path

    def path(self, namespace: str, fingerprint: str) -> Path:
        safe_namespace = re.sub(r"[^a-zA-Z0-9_.-]+", "_", namespace).strip("_")
        return self.root / (safe_namespace or "evaluation") / f"{fingerprint}.jsonl"
