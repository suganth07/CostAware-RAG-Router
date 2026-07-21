from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

from src.adapters.datasets.base import EvaluationSample
from src.evaluation.cache import EvaluationCache
from src.evaluation.core import (
    EvidenceUnit,
    RetrievalRecord,
    evaluate_record,
    summarize_metrics,
)


class CachedEvaluationRunner:
    def __init__(
        self,
        *,
        cache: EvaluationCache | None = None,
        max_workers: int = 1,
        checkpoint_interval: int = 25,
    ) -> None:
        self.cache = cache or EvaluationCache()
        self.max_workers = max(1, int(max_workers))
        self.checkpoint_interval = max(1, int(checkpoint_interval))

    def needs_execution(
        self,
        *,
        namespace: str,
        fingerprint: str,
        system: str,
        samples: Sequence[EvaluationSample],
        refresh: bool = False,
    ) -> bool:
        if refresh:
            return bool(samples)
        cached = self.cache.load(namespace, fingerprint)
        return any(_cache_key(system, sample) not in cached for sample in samples)

    def run(
        self,
        *,
        namespace: str,
        fingerprint: str,
        system: str,
        samples: Sequence[EvaluationSample],
        execute: Callable[[EvaluationSample], Any],
        refresh: bool = False,
    ) -> tuple[list[RetrievalRecord], list[dict[str, Any]]]:
        cached = {} if refresh else self.cache.load(namespace, fingerprint)
        records_by_key = dict(cached)
        missing = [
            sample for sample in samples if _cache_key(system, sample) not in records_by_key
        ]

        if missing:
            if refresh:
                self.cache.save(namespace, fingerprint, {})
            checkpoint: dict[str, RetrievalRecord] = {}

            def store(key: str, record: RetrievalRecord) -> None:
                records_by_key[key] = record
                checkpoint[key] = record
                if len(checkpoint) >= self.checkpoint_interval:
                    self.cache.append(namespace, fingerprint, checkpoint)
                    checkpoint.clear()

            if self.max_workers == 1:
                for sample in missing:
                    key, record = self._execute_one(system, sample, execute)
                    store(key, record)
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(self._execute_one, system, sample, execute): sample
                        for sample in missing
                    }
                    for future in as_completed(futures):
                        key, record = future.result()
                        store(key, record)
            if checkpoint:
                self.cache.append(namespace, fingerprint, checkpoint)

        records: list[RetrievalRecord] = []
        rows: list[dict[str, Any]] = []
        for sample in samples:
            key = _cache_key(system, sample)
            record = RetrievalRecord.from_dict(records_by_key[key].to_dict())
            record.details["cache_hit"] = key in cached and not refresh
            records.append(record)
            rows.append(evaluate_record(sample, record))
        return records, rows

    @staticmethod
    def _execute_one(
        system: str,
        sample: EvaluationSample,
        execute: Callable[[EvaluationSample], Any],
    ) -> tuple[str, RetrievalRecord]:
        started = time.perf_counter()
        output = execute(sample)
        elapsed = time.perf_counter() - started
        return _cache_key(system, sample), retrieval_record_from_output(
            system=system,
            sample=sample,
            output=output,
            latency_seconds=elapsed,
        )


def retrieval_record_from_output(
    *,
    system: str,
    sample: EvaluationSample,
    output: Any,
    latency_seconds: float,
) -> RetrievalRecord:
    generated_answer = str(getattr(output, "answer", "") or "")
    context = getattr(output, "context", output)
    units: list[EvidenceUnit] = []
    capsule_ids: list[str] = []
    details: dict[str, Any] = dict(getattr(context, "details", {}) or {})

    if hasattr(context, "evidence_paths"):
        for snippet in getattr(context, "source_snippets", []) or []:
            units.append(
                EvidenceUnit(
                    id=snippet.document_id or snippet.chunk_id or snippet.id,
                    text=snippet.text,
                    score=float(snippet.score),
                    kind="source_chunk",
                    metadata={
                        **dict(snippet.metadata),
                        "document_id": snippet.document_id,
                        "chunk_id": snippet.chunk_id,
                        "capsule_id": snippet.capsule_id,
                    },
                )
            )
        # Paths follow source units for evidence Recall@K/MRR. On graph-native
        # datasets such as MetaQA there are no source chunks, so paths naturally
        # remain the primary ranked evidence units.
        for index, path in enumerate(context.evidence_paths, start=1):
            units.append(
                EvidenceUnit(
                    id=f"path:{index}:{'-'.join(edge.id for edge in path.edges)}",
                    text=path.text,
                    score=float(path.score),
                    kind="graph_path",
                    metadata=dict(path.metadata),
                )
            )
        capsule_ids = [
            candidate.capsule.id for candidate in getattr(context, "selected_capsules", [])
        ]
        selected_capsules = getattr(context, "selected_capsules", [])
        details.update(
            {
                "evidence_paths": len(getattr(context, "evidence_paths", [])),
                "source_snippets": len(getattr(context, "source_snippets", [])),
                "graph_nodes": sum(
                    len(candidate.capsule.nodes) for candidate in selected_capsules
                ),
                "graph_edges": sum(
                    len(candidate.capsule.edges) for candidate in selected_capsules
                ),
            }
        )
    elif hasattr(context, "vector_results"):
        for result in context.vector_results:
            document = result.document
            document_id = (
                document.metadata.get("document_id")
                or document.metadata.get("url")
                or document.id
            )
            units.append(
                EvidenceUnit(
                    id=str(document_id),
                    text=document.text,
                    score=float(result.score),
                    kind="vector_chunk",
                    metadata={**document.metadata, "document_id": str(document_id)},
                )
            )
        neighborhood = context.graph_neighborhood
        nodes = {node.id: node for node in neighborhood.nodes}
        for edge in neighborhood.edges:
            source = nodes.get(edge.source_id)
            target = nodes.get(edge.target_id)
            source_name = source.properties.get("name") if source else edge.source_id
            target_name = target.properties.get("name") if target else edge.target_id
            units.append(
                EvidenceUnit(
                    id=edge.id,
                    text=f"{source_name} -[{edge.type}]-> {target_name}",
                    kind="graph_edge",
                    metadata={"edge_type": edge.type},
                )
            )
        details.update(
            {
                "vector_results": len(context.vector_results),
                "graph_nodes": len(neighborhood.nodes),
                "graph_edges": len(neighborhood.edges),
            }
        )
    elif hasattr(context, "results"):
        for result in context.results:
            units.append(
                EvidenceUnit(
                    id=str(result.metadata.get("document_id") or result.id),
                    text=result.text,
                    score=float(result.score),
                    kind=str(result.source),
                    metadata=dict(result.metadata),
                )
            )

    return RetrievalRecord(
        system=system,
        sample_id=sample.id,
        question=sample.question,
        context_text=str(getattr(context, "context_text", "") or ""),
        generated_answer=generated_answer,
        evidence_units=units,
        capsule_ids=capsule_ids,
        seed_node_ids=[str(value) for value in getattr(context, "seed_node_ids", [])],
        latency_seconds=latency_seconds,
        details=details,
    )


def write_evaluation_report(
    output_dir: str | Path,
    *,
    records: Sequence[RetrievalRecord],
    rows: Sequence[dict[str, Any]],
    run_config: dict[str, Any],
) -> dict[str, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / "retrieval_records.jsonl"
    metrics_path = directory / "query_metrics.csv"
    summary_path = directory / "summary_metrics.json"
    config_path = directory / "run_config.json"

    with records_path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    _write_csv(metrics_path, rows)
    systems = sorted({str(row.get("system")) for row in rows})
    summary = {
        "systems": {
            system: summarize_metrics(
                [row for row in rows if str(row.get("system")) == system]
            )
            for system in systems
        }
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "records": records_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "config": config_path,
    }


def _cache_key(system: str, sample: EvaluationSample) -> str:
    return f"{system}:{sample.id}"


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )
