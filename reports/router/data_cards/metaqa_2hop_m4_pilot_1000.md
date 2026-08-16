# MetaQA 2-hop M4 pilot data card

## Scope

M4 full-corpus 1,000-question pilot; grouped split infrastructure verification only.

This card records observed dataset provenance and coverage. It does not claim reward calibration, policy training, or production readiness.

## Questions

- Total frozen questions: `1000`
- Dataset counts: `MetaQA`: 1000
- Source split counts: `train`: 1000
- Answer-type counts: `span`: 1000

## Candidate outcomes

- Total raw outcomes: `3000`
- Candidate counts: `graph_rag`: 1000, `hybrid_rag`: 1000, `optimized_graph_rag`: 1000
- Candidate failures/timeouts: `hybrid_rag`: 0, `graph_rag`: 0, `optimized_graph_rag`: 0

| Candidate | Measured rows | Mean latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|
| `hybrid_rag` | 1000 | 43799.97 | 98698.85 |
| `graph_rag` | 1000 | 41992.14 | 78438.40 |
| `optimized_graph_rag` | 1000 | 6190.18 | 8769.77 |

## Metric availability

| Metric | Present rows | Availability |
|---|---:|---:|
| `answer_completeness` | 3000 | 100.0% |
| `answer_relevance` | 3000 | 100.0% |
| `context_precision` | 3000 | 100.0% |
| `contradiction_probability` | 3000 | 100.0% |
| `entity_conflict` | 3000 | 100.0% |
| `evidence_mrr` | 3000 | 100.0% |
| `evidence_ndcg_at_k` | 3000 | 100.0% |
| `exact_or_alias_match` | 3000 | 100.0% |
| `failure_penalty` | 3000 | 100.0% |
| `numeric_conflict` | 3000 | 100.0% |
| `numeric_date_accuracy` | 537 | 17.9% |
| `polarity_conflict` | 3000 | 100.0% |
| `supporting_evidence_recall` | 3000 | 100.0% |
| `token_or_set_f1` | 3000 | 100.0% |

## Grouped split manifest

- Assignment rows: `1000`
- Split counts: `locked_test`: 100, `train`: 800, `validation`: 100
- Group leakage is checked by `build_split_manifest.py` before this card is generated.

## Limitations and next gate

- Optional semantic and critic metrics may be null until their configured models and human-calibration requirements are satisfied.
- A pilot card is not a substitute for the 15,000–30,000-question M5 full-information dataset.
- Do not use this card to authorize M6/M7 training before reward calibration and the M5 data gates are complete.
