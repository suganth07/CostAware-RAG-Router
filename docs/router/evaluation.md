# Router Evaluation Rules

Evaluate every router choice against the observed full-information outcomes for
the same questions. Required baselines are always Hybrid RAG, always GraphRAG,
always Optimized GraphRAG, random routing, a rule-based heuristic, an oracle
imitation policy, and the full-information oracle upper bound.

Report quality, mean/p50/p95 latency, tail latency/CVaR, compute, timeout rate,
action mix, feasibility violations, and regret versus oracle. Use paired
question-level comparisons because every candidate is measured on the same
question.

MetaQA hop labels may be used for **post-hoc slices** only. They are forbidden
from the router's runtime feature vector. Before a final claim, use official
train/test boundaries plus template/entity leakage audits and a locked test
manifest.
