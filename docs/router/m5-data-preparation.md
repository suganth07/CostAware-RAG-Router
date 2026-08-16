# M5 data-preparation foundation

## What this stage prepares

M5 must eventually provide the full offline dataset for the outcome model and
the constrained contextual-bandit policy: 15,000–30,000 questions, all three
candidate outcomes per question, grouped train/validation/locked-test splits,
a genuine cross-domain holdout, and a data card.

The current implementation prepares the safe structure only. It does not turn
the 1,000-question M4 pilot into an M5 training dataset and does not authorize
M6/M7 training.

## Grouped split manifest

Use frozen `QuestionRecord` rows to create a deterministic group-level split.
Every question sharing the selected template, entity, or template/entity group
is assigned to exactly one of `train`, `validation`, or `locked_test`. This
prevents template or entity leakage from inflating later validation scores.

```bash
python scripts/router/build_split_manifest.py \
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl \
  --output-dir artifacts/router/manifests \
  --name metaqa_2hop_m4_pilot_1000_template \
  --group-by template \
  --seed 7
```

This writes:

- `*.assignments.jsonl`: one `question_id`, `group_id`, and offline split per
  frozen question.
- `*.manifest.json`: row/group/dataset counts, seed, requested fractions, and
  a fingerprint.

For a future multi-dataset full collection, hold out an entire dataset as a
cross-domain evaluation slice:

```bash
python scripts/router/build_split_manifest.py \
  --questions artifacts/router/outcomes/full_collection.questions.jsonl \
  --output-dir artifacts/router/manifests \
  --name full_collection_template \
  --group-by template \
  --cross-domain-dataset 2WikiMultiHopQA \
  --seed 7
```

Use the exact dataset name recorded in the frozen question artifact. The script
fails rather than silently allowing a group to cross split boundaries.

## Data card

Create a Markdown provenance and coverage card after outcomes and deterministic
scores are available:

```bash
python scripts/router/create_data_card.py \
  --questions artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.questions.jsonl \
  --outcomes artifacts/router/outcomes/metaqa_2hop_m4_pilot_1000.jsonl \
  --scored-outcomes artifacts/router/datasets/metaqa_2hop_m4_pilot_1000_scored.jsonl \
  --split-assignments artifacts/router/manifests/metaqa_2hop_m4_pilot_1000_template.assignments.jsonl \
  --title "MetaQA 2-hop M4 pilot data card" \
  --scope "M4 full-corpus pilot; split-tooling verification only." \
  --output reports/router/data_cards/metaqa_2hop_m4_pilot_1000.md
```

The card reports question provenance, answer-type distribution, candidate
coverage, failures, latency distribution, metric availability, and split
counts. It explicitly states unavailable semantic/critic metrics and does not
invent reward calibration.

## Current verified pilot result

The current M4 pilot split has 1,000 MetaQA questions, allocated as 800 train,
100 validation, and 100 locked-test questions over 198 template groups. It has
no cross-domain holdout because it contains only MetaQA. The card is available
at `reports/router/data_cards/metaqa_2hop_m4_pilot_1000.md`.

## What still blocks learning

Before M6/M7, the plan still requires the M4 reward-calibration gate,
human-reviewed labels, configured optional metric/critic decisions, and the
larger M5 full-information collection. The skipped repeated-latency collection
also remains a documented limitation: do not use the pilot's single timings as
validated latency quantiles.
