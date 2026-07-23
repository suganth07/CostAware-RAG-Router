"""Generate a deterministic, candidate-balanced pilot set of critic judgments."""

from __future__ import annotations

import argparse
import json
import random
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.llm.ollama import OllamaLLM, OllamaLLMConfig
from src.router.critics import CriticResponseCache, StructuredCritic
from src.router.critics.prompts import CRITIC_PROMPT_VERSION
from src.router.data.serialization import read_jsonl, write_jsonl_atomic
from src.router.types import CandidateOutcome, QuestionRecord


def main() -> None:
    args = parse_args()
    outcomes = [CandidateOutcome.from_dict(row) for row in read_jsonl(args.outcomes)]
    questions = {row["question_id"]: QuestionRecord(**row) for row in read_jsonl(args.questions)}
    selected = balanced_sample(outcomes, limit=args.max_records, seed=args.seed)
    existing = read_jsonl(args.output) if args.resume else []
    # Make the normalization flag explicit for artifacts written before the
    # recoverable empty-enum rule was introduced.
    for row in existing:
        row.setdefault("failure_type_normalized", False)
    existing_keys = {
        (row["question_id"], row["candidate_id"], int(row["repeat_id"]), row["run_id"], row["judge_model"], row["prompt_version"])
        for row in existing
    }
    verify_ollama(args.ollama_base_url, args.judge_model, args.timeout_seconds)
    llm = OllamaLLM(OllamaLLMConfig(
        model=args.judge_model,
        base_url=args.ollama_base_url,
        temperature=0,
        timeout_seconds=args.timeout_seconds,
        num_predict=args.max_tokens,
    ))
    results = list(existing)
    failures: list[dict[str, str]] = []
    with CriticResponseCache(args.critic_cache) as cache:
        critic = StructuredCritic(llm, judge_model_fingerprint=args.judge_model, cache=cache, judge_run_id=args.judge_run_id)
        for outcome in selected:
            key = (outcome.question_id, outcome.candidate_id, outcome.repeat_id, outcome.run_id, args.judge_model, CRITIC_PROMPT_VERSION)
            if key in existing_keys:
                continue
            try:
                record = critic.judge(questions[outcome.question_id], outcome)
                results.append(record.to_dict())
                existing_keys.add(record.key)
                write_jsonl_atomic(args.output, results)
            except Exception as exc:
                failure = {"question_id": outcome.question_id, "candidate_id": outcome.candidate_id, "repeat_id": str(outcome.repeat_id), "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                if args.fail_fast:
                    raise RuntimeError(json.dumps(failure)) from exc
    # Always refresh this companion artifact so it cannot retain failures from
    # a previous run after a successful resumed run.
    write_jsonl_atomic(args.output.with_suffix(".failures.jsonl"), failures)
    print(json.dumps({"critic_records": str(args.output), "selected": len(selected), "written": len(results), "failures": len(failures)}, indent=2))


def balanced_sample(outcomes: list[CandidateOutcome], *, limit: int, seed: int) -> list[CandidateOutcome]:
    if limit < 1:
        raise ValueError("--max-records must be at least one")
    groups: dict[str, list[CandidateOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if not outcome.error_type and outcome.generated_answer.strip():
            groups[outcome.candidate_id].append(outcome)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[CandidateOutcome] = []
    actions = sorted(groups)
    while actions and len(selected) < limit:
        next_actions = []
        for action in actions:
            if len(selected) >= limit:
                break
            if groups[action]:
                selected.append(groups[action].pop())
            if groups[action]:
                next_actions.append(action)
        actions = next_actions
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect strict candidate-blind LLM critic judgments.")
    parser.add_argument("--outcomes", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output", default="artifacts/router/critiques/pilot_critic.jsonl", type=Path)
    parser.add_argument("--judge-model", default="qwen3:4b-instruct")
    parser.add_argument("--critic-cache", default="artifacts/router/cache/critic_responses.sqlite3")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--judge-run-id", default=None)
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def verify_ollama(base_url: str, judge_model: str, timeout_seconds: float) -> None:
    """Fail once before sampling if the local judge service is unavailable."""
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=min(timeout_seconds, 10.0)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(
            f"Ollama is not reachable at {base_url}. Start it with `ollama serve` "
            f"(or open the Ollama desktop app), then rerun. Details: {exc}"
        ) from exc
    available = {str(item.get("name")) for item in payload.get("models", []) if isinstance(item, dict)}
    if judge_model not in available:
        raise SystemExit(
            f"Ollama is running but judge model {judge_model!r} is not installed. "
            f"Run `ollama pull {judge_model}` and rerun."
        )


if __name__ == "__main__":
    main()
