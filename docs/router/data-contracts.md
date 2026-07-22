# Router Data Contracts and Collection

Raw outcomes are stored as JSONL because they contain nested retrieval records.
Every completed question/repeat has exactly one record for each action. The
collector checkpoints atomically after every record, then safely resumes using
`(question_id, candidate_id, repeat_id, run_id)`.

## Files created by collection

- `artifacts/router/outcomes/*.jsonl`: raw candidate outcomes.
- matching `.questions.jsonl`: answers and training-only annotations.
- matching `.manifest.json`: code/data/model/hardware fingerprints and index
  reports.

Gold answers, evidence, split names, and hop labels are in `QuestionRecord`
for offline scoring only. They must not enter the runtime feature vector.

`build_training_dataset.py` refuses incomplete action groups. It creates a
full-information record containing quality, latency, compute, failure status,
feasibility mask, and budget-aware oracle action for all three candidates.
