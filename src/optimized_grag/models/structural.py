from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Protocol

from src.optimized_grag.partitioning.types import GraphCapsule


class StructuralEncoder(Protocol):
    def encode_capsule(self, capsule: GraphCapsule) -> list[float]:
        raise NotImplementedError

    def project_query(self, query_vector: list[float]) -> list[float]:
        raise NotImplementedError


class FoldedQueryProjector:
    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension

    def project(self, query_vector: list[float]) -> list[float]:
        if not query_vector:
            return [0.0] * self.dimension
        folded = [0.0] * self.dimension
        for index, value in enumerate(query_vector):
            folded[index % self.dimension] += float(value)
        return _l2_normalize(folded)


class FeatureStructuralEncoder:
    """Runtime fallback until trained R-GCN capsule embeddings are exported."""

    def __init__(self, *, dimension: int = 8) -> None:
        self.dimension = dimension
        self.projector = FoldedQueryProjector(dimension)

    def encode_capsule(self, capsule: GraphCapsule) -> list[float]:
        stats = capsule.stats
        raw = [
            math.log1p(stats.node_count),
            math.log1p(stats.edge_count),
            stats.density,
            stats.connected_component_ratio,
            stats.hub_count / max(stats.node_count, 1),
            stats.overlap_ratio,
            math.log1p(len(capsule.relation_types)),
            math.log1p(len(capsule.source_document_ids)),
        ]
        if len(raw) < self.dimension:
            raw.extend([0.0] * (self.dimension - len(raw)))
        return _l2_normalize(raw[: self.dimension])

    def project_query(self, query_vector: list[float]) -> list[float]:
        return self.projector.project(query_vector)


class TrainedStructuralEncoder:
    """Loads exported capsule R-GCN embeddings and optional query projector artifacts."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        projector_artifact_dir: str | Path | None = None,
        fallback: StructuralEncoder | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.projector_artifact_dir = Path(projector_artifact_dir or artifact_dir)
        self.fallback = fallback or FeatureStructuralEncoder()
        self.capsule_vectors = self._load_capsule_vectors()
        self.structural_output_dim = _vector_dim(self.capsule_vectors) or 64
        self.output_dim = self.structural_output_dim
        self.artifact_fingerprint = self._metadata_value(
            "capsule_rgcn_metadata.json", "capsule_fingerprint"
        )
        self.projector_fingerprint = self._metadata_value(
            "query_projection_metadata.json",
            "capsule_fingerprint",
            directory=self.projector_artifact_dir,
        )
        self.routing_vectors = self._load_routing_vectors()
        self.routing_output_dim = (
            _vector_dim(self.routing_vectors) or self.structural_output_dim
        )
        self.projector_output_dim = self.routing_output_dim
        self.projector = self._load_projector()
        self.runtime_match_rate: float | None = None
        self.runtime_fingerprint: str | None = None
        self.trained_space_enabled = False
        self.routing_space_kind = (
            "semantic_capsule_text" if self.routing_vectors else "structural_rgcn"
        )

    @property
    def is_available(self) -> bool:
        return bool(self.capsule_vectors)

    def bind_capsules(
        self,
        capsules: list[GraphCapsule],
        *,
        min_match_rate: float = 0.95,
    ) -> float:
        """Enable trained space only when it matches the runtime capsule build.

        Exported vectors are keyed by capsule ID. Using a projector trained for
        a mostly different capsule set mixes incompatible vector spaces and can
        silently damage ranking, so a low-match artifact is disabled as a unit.
        """
        ids = [capsule.id for capsule in capsules]
        matches = sum(capsule_id in self.capsule_vectors for capsule_id in ids)
        routing_matches = (
            sum(capsule_id in self.routing_vectors for capsule_id in ids)
            if self.routing_vectors
            else matches
        )
        self.runtime_match_rate = matches / len(ids) if ids else 0.0
        self.runtime_fingerprint = capsule_fingerprint(capsules)
        fingerprint_matches = bool(self.artifact_fingerprint) and (
            self.artifact_fingerprint == self.runtime_fingerprint
            and self.projector_fingerprint == self.artifact_fingerprint
        )
        self.trained_space_enabled = (
            bool(self.capsule_vectors)
            and self.projector is not None
            and fingerprint_matches
            and self.runtime_match_rate >= min_match_rate
            and routing_matches / max(len(ids), 1) >= min_match_rate
        )
        return self.runtime_match_rate

    def encode_capsule(self, capsule: GraphCapsule) -> list[float]:
        vector = self.capsule_vectors.get(capsule.id) if self.trained_space_enabled else None
        if vector is not None:
            return vector
        # A graph/config change can legitimately create a capsule that was not
        # present in the exported training run. A zero vector silently removes
        # structural information and makes reranking depend on artifact match
        # accidents, so preserve a usable structural signal via the fallback.
        return _resize_and_normalize(
            self.fallback.encode_capsule(capsule),
            self.structural_output_dim,
        )

    def encode_routing_capsule(self, capsule: GraphCapsule) -> list[float]:
        if self.trained_space_enabled and self.routing_vectors:
            vector = self.routing_vectors.get(capsule.id)
            if vector is not None:
                return vector
        return self.encode_capsule(capsule)

    def project_query(self, query_vector: list[float]) -> list[float]:
        if not self.trained_space_enabled:
            projected = self.fallback.project_query(query_vector)
            return _resize_and_normalize(projected, self.routing_output_dim)
        if self.projector is None:
            projected = self.fallback.project_query(query_vector)
            return _resize_and_normalize(projected, self.routing_output_dim)
        try:
            import torch
        except ImportError:
            projected = self.fallback.project_query(query_vector)
            return _resize_and_normalize(projected, self.routing_output_dim)

        model, input_dim = self.projector
        values = list(query_vector[:input_dim])
        if len(values) < input_dim:
            values.extend([0.0] * (input_dim - len(values)))
        with torch.no_grad():
            tensor = torch.tensor([values], dtype=torch.float32)
            projected = model(tensor).squeeze(0).cpu().tolist()
        return _l2_normalize([float(value) for value in projected])

    def project_structural_query(self, query_vector: list[float]) -> list[float]:
        """Return a structural-space query only for legacy structural targets."""
        if self.routing_vectors:
            return []
        return self.project_query(query_vector)

    def _load_capsule_vectors(self) -> dict[str, list[float]]:
        manifest = self._metadata_value(
            "capsule_rgcn_metadata.json", "projection_manifest"
        )
        filename = manifest.split(":", 1)[0] if manifest else "capsule_rgcn_embeddings.npz"
        path = self.artifact_dir / filename
        if not path.exists():
            return {}
        try:
            import numpy as np
        except ImportError:
            return {}
        payload = np.load(path, allow_pickle=True)
        ids = [str(value) for value in payload["capsule_ids"]]
        embeddings = payload["embeddings"].astype("float32")
        return {
            capsule_id: _l2_normalize([float(value) for value in vector])
            for capsule_id, vector in zip(ids, embeddings)
        }

    def _load_projector(self):
        metadata_path = self.projector_artifact_dir / "query_projection_metadata.json"
        weights_path = self.projector_artifact_dir / "query_projection_mlp.pt"
        if not metadata_path.exists() or not weights_path.exists():
            return None
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            return None

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        input_dim = int(metadata.get("input_dim") or 0)
        output_dim = int(metadata.get("output_dim") or self.output_dim)
        architecture = str(metadata.get("architecture") or "legacy_mlp")
        if input_dim <= 0 or output_dim <= 0:
            return None

        class ProjectionMLP(nn.Module):
            def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
                super().__init__()
                if architecture == "identity_residual_mlp":
                    self.base = (
                        nn.Identity()
                        if input_dim == output_dim
                        else nn.Linear(input_dim, output_dim, bias=False)
                    )
                    self.residual = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, output_dim),
                    )
                else:
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, output_dim),
                    )

            def forward(self, x):
                if architecture == "identity_residual_mlp":
                    return F.normalize(self.base(x) + self.residual(x), dim=-1)
                return F.normalize(self.net(x), dim=-1)

        model = ProjectionMLP(input_dim, output_dim)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        self.projector_output_dim = output_dim
        self.routing_output_dim = output_dim
        return model, input_dim

    def _load_routing_vectors(self) -> dict[str, list[float]]:
        metadata_path = self.projector_artifact_dir / "query_projection_metadata.json"
        if not metadata_path.exists():
            return {}
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if str(metadata.get("target_space") or "") != "semantic_capsule_text":
            return {}
        manifest = str(metadata.get("target_manifest") or "")
        path = self.projector_artifact_dir / manifest
        if not manifest or not path.exists():
            return {}
        try:
            import numpy as np
        except ImportError:
            return {}
        payload = np.load(path, allow_pickle=True)
        ids = [str(value) for value in payload["capsule_ids"]]
        embeddings = payload["embeddings"].astype("float32")
        return {
            capsule_id: _l2_normalize([float(value) for value in vector])
            for capsule_id, vector in zip(ids, embeddings)
        }

    def _metadata_value(
        self,
        filename: str,
        key: str,
        *,
        directory: Path | None = None,
    ) -> str | None:
        path = (directory or self.artifact_dir) / filename
        if not path.exists():
            return None
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = metadata.get(key)
        return str(value) if value else None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def capsule_fingerprint(capsules: list[GraphCapsule]) -> str:
    """Stable identity for the exact capsule graph used by model artifacts."""
    digest = hashlib.sha256()
    for capsule in sorted(capsules, key=lambda value: value.id):
        digest.update(capsule.id.encode("utf-8", errors="replace"))
        for node_id in sorted(capsule.node_ids):
            digest.update(b"N")
            digest.update(node_id.encode("utf-8", errors="replace"))
        for edge_id in sorted(capsule.edge_ids):
            digest.update(b"E")
            digest.update(edge_id.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _vector_dim(vectors: dict[str, list[float]]) -> int:
    for vector in vectors.values():
        return len(vector)
    return 0


def _resize_and_normalize(values: list[float], dimension: int) -> list[float]:
    if len(values) < dimension:
        values = [*values, *([0.0] * (dimension - len(values)))]
    return _l2_normalize(values[:dimension])
