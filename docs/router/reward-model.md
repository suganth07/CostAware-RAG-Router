# Deterministic Reward and Budget Model

The quality composer keeps missing metrics as `null`; it never silently treats
an unavailable semantic/critic score as zero.

Initial quality blocks are answer correctness (50%), faithfulness (20%),
evidence quality (20%), and completeness (10%). The active weights are
renormalized only across available blocks.

Hard numeric/date and boolean-polarity conflicts return quality `-1.0`. This
prevents future semantic similarity from rewarding an answer with the wrong
number or polarity.

The first reduced compute unit records graph nodes, graph edges, context size,
and available input/output tokens. GPU/CPU timing can be added later without
changing the `CandidateOutcome` contract.

The budget-aware oracle selects the highest-quality feasible action; if every
action violates a configured limit, it records the least-violation fallback.
Mean and p95 constraints are calculated per action across the scored collection
slice (including repeats), while `max_request_latency_ms` remains a
per-question hard limit. This avoids treating one noisy measurement as a
tail-latency estimate.
