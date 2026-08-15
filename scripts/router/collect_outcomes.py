"""Collect resumable full-information outcomes for the three router candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.adapters.datasets import get_dataset_adapter
from src.adapters.embedding import CachingEmbeddingAdapter, OllamaEmbeddingAdapter, OllamaEmbeddingConfig, local_embedding_cache_namespace
from src.adapters.llm import OllamaLLM, OllamaLLMConfig
from src.router.candidates import build_default_registry
from src.router.data import OutcomeCollector, build_manifest, fingerprint_payload, question_record_from_sample, stratified_sample
from src.router.data.manifests import hardware_fingerprint
from src.router.data.serialization import write_jsonl_atomic
from src.router.data.validation import validate_complete_matrix


def main() -> None:
    args = parse_args()
    bundle = _load_bundle(args)
    samples = _select_samples(bundle.eval_samples, args)
    if not samples:
        raise RuntimeError("No evaluation samples were found")
    records = [question_record_from_sample(sample) for sample in samples]
    pairs = list(zip(records, samples))
    embedding = CachingEmbeddingAdapter(
        OllamaEmbeddingAdapter(OllamaEmbeddingConfig(model=args.embedding_model, base_url=args.ollama_url, keep_alive="30m", timeout_seconds=args.timeout, batch_size=args.embedding_batch_size, options={"num_gpu": 999})),
        cache_path=args.embedding_cache,
        namespace=local_embedding_cache_namespace(args.embedding_model),
        model_name=args.embedding_model,
    )
    llm = OllamaLLM(OllamaLLMConfig(model=args.llm_model, base_url=args.ollama_url, temperature=0.0, num_predict=160, keep_alive="30m", default_options={"num_gpu": 999}, timeout_seconds=args.timeout))
    registry, index_reports = build_default_registry(
        bundle=bundle,
        embedding_adapter=embedding,
        llm=llm,
        normal_index_batch_size=args.index_batch_size,
    )
    corpus_fingerprint = fingerprint_payload({"bundle": bundle.metadata, "documents": [(document.id, document.text) for document in bundle.documents], "graph_edges": [edge.id for edge in bundle.graph_edges]})
    model_fingerprint = fingerprint_payload({"embedding_model": args.embedding_model, "llm_model": args.llm_model, "generation": args.run_generation})
    run_payload = {"dataset": args.dataset, "split": args.split, "questions": [record.to_dict() for record in records], "corpus": corpus_fingerprint, "model": model_fingerprint, "repeats": args.repeats, "seed": args.seed}
    run_id = fingerprint_payload(run_payload)
    collector = OutcomeCollector(registry=registry, run_id=run_id, corpus_fingerprint=corpus_fingerprint, model_fingerprint=model_fingerprint, hardware_fingerprint=hardware_fingerprint(), seed=args.seed)
    output = Path(args.output)
    outcomes = collector.collect(pairs=pairs, output_path=output, repeats=args.repeats, run_generation=args.run_generation, resume=not args.no_resume)
    validate_complete_matrix(outcomes, repeats=args.repeats)
    questions_path = output.with_suffix(".questions.jsonl")
    write_jsonl_atomic(questions_path, [record.to_dict() for record in records])
    manifest = build_manifest(artifact_kind="router_outcomes", rows=outcomes, payload={**run_payload, "run_id": run_id, "index_reports": index_reports, "learned_structural_models_enabled": False, "optimized_structural_encoder": "FeatureStructuralEncoder"})
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"outcomes": str(output), "questions": str(questions_path), "manifest": str(manifest_path), "outcome_count": len(outcomes), "run_id": run_id}, indent=2))


def _load_bundle(args):
    # A frozen subset can contain arbitrary IDs from an earlier stratified
    # collection. Load the complete split before resolving those IDs; limiting
    # first would silently make most valid subset IDs unavailable.
    eval_limit = None if args.questions_file else args.max_questions
    if args.dataset.startswith("metaqa_"):
        hop = args.dataset.removeprefix("metaqa_").removesuffix("hop")
        adapter = get_dataset_adapter("metaqa", hop=f"{hop}-hop", variant="vanilla", split=args.split)
        bundle = adapter.load(limit_documents=None, limit_eval_samples=eval_limit)
        if not args.full_corpus:
            # Keep native graph complete; trim text documents only for smoke wiring.
            bundle.documents = bundle.documents[: args.max_documents]
        return bundle
    adapter = get_dataset_adapter("2wiki", split=args.split) if args.dataset == "2wiki" else get_dataset_adapter("multihop_rag", split=args.split)
    return adapter.load(limit_documents=None if args.full_corpus else args.max_documents, limit_eval_samples=eval_limit)


def _select_samples(available, args):
    if args.questions_file:
        selected_ids = _question_ids_from_file(args.questions_file)
        available_by_id = {sample.id: sample for sample in available}
        missing = [question_id for question_id in selected_ids if question_id not in available_by_id]
        if missing:
            raise RuntimeError(f"{len(missing)} requested question IDs are absent from dataset/split; first={missing[0]!r}")
        return [available_by_id[question_id] for question_id in selected_ids]
    return stratified_sample(available, limit=args.max_questions, seed=args.seed) if args.stratified else available[: args.max_questions]


def _question_ids_from_file(path: str) -> list[str]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if not value:
                continue
            rows.append(str(json.loads(value).get("question_id")) if value.startswith("{") else value)
    if not rows or any(not value or value == "None" for value in rows):
        raise RuntimeError("--questions-file must contain JSONL QuestionRecord rows or one question_id per line")
    if len(rows) != len(set(rows)):
        raise RuntimeError("--questions-file contains duplicate question IDs")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect complete three-action router outcomes with resume support.")
    parser.add_argument("--dataset", choices=("metaqa_1hop", "metaqa_2hop", "metaqa_3hop", "2wiki", "multihop_rag"), required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--questions-file", help="JSONL QuestionRecord subset or one question_id per line; overrides --max-questions/--stratified")
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--full-corpus", action="store_true")
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--llm-model", default="qwen3:4b-instruct")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--index-batch-size", type=int, default=512, help="NormalRAG cache-checkpoint batch size during registry construction.")
    parser.add_argument("--embedding-cache", default="artifacts/cache/embeddings.sqlite3")
    parser.add_argument("--output", default="artifacts/router/outcomes/pilot.jsonl")
    return parser.parse_args()


if __name__ == "__main__":
    main()
