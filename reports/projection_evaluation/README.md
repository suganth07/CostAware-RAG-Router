# MetaQA Gold-Path Projector Evaluation

Date: 2026-07-11

These are standalone capsule-ranking results on the first 500 held-out `test`
questions per hop. A positive is any capsule containing the first edge of a
valid path from the bracketed query entity to a gold answer. This evaluates the
router's launch decision; the earlier answer-containing-capsule metric was
misaligned for multi-hop retrieval.

The learned model is an identity-residual projector over cached semantic capsule
embeddings. `Raw semantic` is the exact epoch-0 baseline, and `Random` accounts
for the number of valid capsules per query.

| Slice | Labeled queries | Label coverage | Raw semantic MRR | Learned MRR | Learned Recall@1 | Recall@10 | Recall@20 | Random MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-hop | 483 | 96.6% | 0.2864 | 0.2873 | 27.1% | 31.7% | 32.5% | 0.00119 |
| 2-hop | 428 | 85.6% | 0.2643 | 0.3035 | 28.3% | 33.6% | 34.8% | 0.00135 |
| 3-hop | 435 | 87.0% | 0.3449 | 0.4079 | 37.7% | 45.5% | 47.1% | 0.00230 |

Learning is essentially neutral on 1-hop, improves MRR by 14.8% over raw
semantic retrieval on 2-hop, and improves MRR by 18.3% on 3-hop. Compared with
random ranking, learned MRR is approximately 242x, 224x, and 178x higher.

This supports a precise claim: **path-supervised residual projection improves
semantic capsule ranking for multi-hop MetaQA, especially 2-hop and 3-hop.** It
does not prove that R-GCN projection works; direct projection into the pure
R-GCN link-reconstruction space failed and is no longer used as the text target.

Machine-readable `*_semantic_path_projector_500.json` and
`*_raw_semantic_path_500.json` files contain the complete metrics and artifact
fingerprints. Obsolete answer-target and 100-query exploratory reports were
removed to prevent accidental comparison with the corrected metric.
