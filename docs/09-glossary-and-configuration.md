# 9. Glossary and Configuration Reference

## Glossary

| Term | Meaning |
|---|---|
| Node | An entity, concept, or source chunk. |
| Edge | A typed relation between two nodes. |
| Seed | A graph node linked directly from the question. |
| Hub | A very high-degree node whose expansion needs stronger limits. |
| Region | A bounded graph partition used to organize capsule construction. |
| Anchor | The central node used to grow a capsule. |
| Capsule | A bounded overlapping local graph used as a retrieval unit. |
| Supergraph | A graph whose nodes are capsules and whose edges represent overlap. |
| R-GCN | Relation-aware graph convolutional network. |
| Projector | Identity-residual MLP reranking semantic capsule vectors with gold-path supervision. |
| Evidence path | Ordered nodes/typed edges supporting an answer. |
| MRR | Mean reciprocal rank of the first relevant result. |
| Recall@K | Fraction/presence of gold evidence within the first K results. |
| Fingerprint | Stable hash proving artifacts match runtime capsules. |

## Important optimized defaults for unified MetaQA

| Setting | Value | Purpose |
|---|---:|---|
| `max_region_nodes` | 1,000 | Bound partition size. |
| `max_capsule_nodes` | 250 | Prevent giant retrieval units. |
| `max_capsule_edges` | 900 | Bound capsule density/memory. |
| `max_hops` | 2 | Offline capsule ego radius. |
| `anchors_per_region` | 5 | Initial capsule diversity. |
| `first_stage_candidate_capsules` | 100 | Cheap retrieval pool. |
| `top_k_capsules` | 12 | Capsules entering local search. |
| `max_merged_nodes` | 2,400 | Fallback capsule-merged graph bound. |
| `max_candidate_paths` | 11 | Total candidate path ceiling. |
| `max_path_length` | 3 | Runtime search ceiling, not a supplied hop label. |
| `final_paths_to_llm` | 11 | Evidence paths rendered. |
| `max_seed_neighbors` | 200 | Per-node expansion fanout. |
| `direct_seed_path_search` | true | Walk the original graph for exact linked seeds. |
| `exact_seed_capsule_routing` | true | Use node membership instead of dense routing for exact seeds. |
| `seed_expansion_depth` | 3 | Fallback original-graph safety net. |
| `max_seed_expansion_nodes` | 6,000 | Fallback safety-net node ceiling across seeds. |
| `max_context_chars` | 12,000 | Final prompt context ceiling. |

The CLI `--max-hops` overrides `graph_depth`, `max_path_length`, and
`seed_expansion_depth` together for a fair query-time comparison. It does not
override the offline capsule-building `max_hops` value.

## Budget relationships

Increasing capsules improves routing recall but enlarges the local graph.
Increasing seed expansion protects cross-capsule paths but can increase beam
work. Increasing candidate paths improves depth/relation coverage; increasing
final paths directly consumes context. Tune in that order and validate Recall@K,
latency, and context size together.

## What “adaptive” means here

The retriever is adaptive because it receives no hop label, keeps candidates at
several depths, uses question/entity/relation signals to rank them, and preserves
per-seed fairness. It is still bounded by explicit ceilings. Those ceilings are
intentional controls, not knowledge of the gold reasoning path.
