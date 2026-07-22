"""Resumable, randomized full-information candidate outcome collection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from src.adapters.datasets.base import EvaluationSample
from src.router.candidates import CandidateRegistry
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.exceptions import OutcomeContractError
from src.router.types import CandidateOutcome, QuestionRecord


class OutcomeCollector:
    def __init__(self, *, registry: CandidateRegistry, run_id: str, corpus_fingerprint: str, model_fingerprint: str, hardware_fingerprint: str, seed: int = 7) -> None:
        self.registry = registry
        self.run_id = run_id
        self.corpus_fingerprint = corpus_fingerprint
        self.model_fingerprint = model_fingerprint
        self.hardware_fingerprint = hardware_fingerprint
        self.seed = seed

    def collect(
        self,
        *,
        pairs: Iterable[tuple[QuestionRecord, EvaluationSample]],
        output_path: str | Path,
        repeats: int = 1,
        run_generation: bool = False,
        resume: bool = True,
    ) -> list[CandidateOutcome]:
        existing = [CandidateOutcome.from_dict(row) for row in read_jsonl(output_path)] if resume else []
        incompatible = {item.run_id for item in existing if item.run_id != self.run_id}
        if incompatible:
            raise OutcomeContractError(
                "Refusing to mix a different run fingerprint into an existing outcome file. "
                "Choose a new --output path or rerun with --no-resume."
            )
        completed = {item.key for item in existing}
        outcomes = list(existing)
        rng = random.Random(self.seed)
        for question_record, sample in pairs:
            for repeat_id in range(repeats):
                actions = list(self.registry.actions)
                rng.shuffle(actions)
                for action in actions:
                    key = (question_record.question_id, action, repeat_id, self.run_id)
                    if key in completed:
                        continue
                    outcome = self.registry.get(action).execute(
                        record=question_record,
                        sample=sample,
                        repeat_id=repeat_id,
                        run_id=self.run_id,
                        corpus_fingerprint=self.corpus_fingerprint,
                        model_fingerprint=self.model_fingerprint,
                        hardware_fingerprint=self.hardware_fingerprint,
                        run_generation=run_generation,
                    )
                    outcomes.append(outcome)
                    completed.add(key)
                    # Checkpoint after every outcome so killed long runs resume safely.
                    write_jsonl_atomic(output_path, [item.to_dict() for item in outcomes])
        return outcomes
