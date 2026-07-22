"""Content fingerprints and manifests make route outcomes reproducible."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from src.router.constants import ACTION_IDS, ROUTER_SCHEMA_VERSION
from src.router.types import ArtifactManifest


def fingerprint_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hardware_fingerprint() -> str:
    import platform
    return fingerprint_payload({"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor()})


def build_manifest(*, artifact_kind: str, rows: Iterable[Any], payload: dict[str, Any]) -> ArtifactManifest:
    values = list(rows)
    canonical_rows = [item.to_dict() if hasattr(item, "to_dict") else item for item in values]
    fingerprint = fingerprint_payload({"artifact_kind": artifact_kind, "payload": payload, "rows": canonical_rows})
    return ArtifactManifest(
        schema_version=ROUTER_SCHEMA_VERSION,
        artifact_kind=artifact_kind,
        fingerprint=fingerprint,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        row_count=len(values),
        actions=ACTION_IDS,
        payload=dict(payload),
    )
