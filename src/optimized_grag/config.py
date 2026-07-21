from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OptimizedGraphRAGConfig:
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_region_nodes: int = 1000
    max_capsule_nodes: int = 300
    max_capsule_edges: int = 1000
    max_hops: int = 2
    hub_percentile: float = 95.0
    max_hub_neighbors: int = 20
    max_overlap_ratio: float = 0.15
    anchors_per_region: int = 5
    max_capsules_per_region: int = 5
    ensure_node_coverage: bool = True
    first_stage_candidate_capsules: int = 100
    top_k_capsules: int = 10
    max_merged_nodes: int = 1200
    max_candidate_paths: int = 200
    max_path_length: int = 3
    final_paths_to_llm: int = 12
    max_seed_neighbors: int = 100
    seed_expansion_depth: int = 1
    max_seed_expansion_nodes: int = 1200
    direct_seed_path_search: bool = True
    exact_seed_capsule_routing: bool = True
    snippets_per_capsule: int = 3
    max_source_snippets: int = 24
    max_context_chars: int = 12_000
    system_prompt: str = (
        "You answer questions using compact graph capsule evidence. "
        "Use only the supplied evidence; if it is insufficient, say what is missing."
    )
