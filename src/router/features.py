"""Runtime-safe router features available before choosing a RAG candidate."""

from __future__ import annotations

# Keep the initial scalar feature set identical to the proven experimental
# router while the richer probe/entity modules are introduced incrementally.
from src.rl_router.features import feature_names, query_features

__all__ = ["feature_names", "query_features"]
