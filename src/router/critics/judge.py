"""Local structured critic runner."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from src.adapters.llm.adapter import LLM
from src.router.critics.prompts import CRITIC_PROMPT_VERSION, SYSTEM_PROMPT, build_critic_prompt
from src.router.critics.schemas import parse_critique
from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord


class StructuredCritic:
    def __init__(self, llm: LLM, *, judge_run_id: str | None = None) -> None:
        self.llm = llm
        self.judge_run_id = judge_run_id or f"critic-{uuid4().hex[:12]}"

    def judge(self, question: QuestionRecord, outcome: CandidateOutcome) -> CritiqueRecord:
        response = self.llm.generate(
            build_critic_prompt(question, outcome),
            system_prompt=SYSTEM_PROMPT,
            options={"temperature": 0},
        )
        parsed = parse_critique(response.text)
        return CritiqueRecord(
            question_id=outcome.question_id,
            candidate_id=outcome.candidate_id,
            repeat_id=outcome.repeat_id,
            run_id=outcome.run_id,
            judge_model=response.model,
            prompt_version=CRITIC_PROMPT_VERSION,
            judge_run_id=self.judge_run_id,
            raw_response_hash=hashlib.sha256(response.text.encode("utf-8")).hexdigest(),
            **parsed,
        )
