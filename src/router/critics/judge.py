"""Local structured critic runner."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from src.adapters.llm.adapter import LLM
from src.router.critics.prompts import ANSWER_SYSTEM_PROMPT, CRITIC_PROMPT_VERSION, SYSTEM_PROMPT, build_answer_correctness_prompt, build_critic_prompt
from src.router.critics.schemas import parse_answer_critique, parse_critique
from src.router.types import CandidateOutcome, CritiqueRecord, QuestionRecord


class StructuredCritic:
    def __init__(self, llm: LLM, *, judge_run_id: str | None = None) -> None:
        self.llm = llm
        self.judge_run_id = judge_run_id or f"critic-{uuid4().hex[:12]}"

    def judge(self, question: QuestionRecord, outcome: CandidateOutcome) -> CritiqueRecord:
        answer_response = self.llm.generate(
            build_answer_correctness_prompt(question, outcome),
            system_prompt=ANSWER_SYSTEM_PROMPT,
            options={"temperature": 0},
        )
        evidence_response = self.llm.generate(
            build_critic_prompt(question, outcome),
            system_prompt=SYSTEM_PROMPT,
            options={"temperature": 0},
        )
        answer = parse_answer_critique(answer_response.text)
        evidence = parse_critique(evidence_response.text)
        return CritiqueRecord(
            question_id=outcome.question_id,
            candidate_id=outcome.candidate_id,
            repeat_id=outcome.repeat_id,
            run_id=outcome.run_id,
            judge_model=answer_response.model,
            prompt_version=CRITIC_PROMPT_VERSION,
            judge_run_id=self.judge_run_id,
            correctness=answer["correctness"],
            groundedness=evidence["groundedness"],
            evidence_completeness=evidence["evidence_completeness"],
            reasoning_completeness=evidence["reasoning_completeness"],
            answer_relevance=answer["answer_relevance"],
            answerable_from_context=evidence["answerable_from_context"],
            hallucination_detected=evidence["hallucination_detected"],
            failure_type=answer["failure_type"],
            failure_type_normalized=answer["failure_type_normalized"],
            confidence=answer["confidence"],
            supporting_claims=evidence["supporting_claims"],
            unsupported_claims=evidence["unsupported_claims"],
            missing_information=evidence["missing_information"],
            critique=f"Answer correctness: {answer['critique']}\n\nEvidence grounding: {evidence['critique']}",
            raw_response_hash=hashlib.sha256((answer_response.text + "\n" + evidence_response.text).encode("utf-8")).hexdigest(),
        )
