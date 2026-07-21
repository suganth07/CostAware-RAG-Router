from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.embedding.adapter import EmbeddingAdapter, EmbeddingResult
from src.adapters.llm.adapter import LLM, LLMResponse
from src.adapters.storage.graph_store import InMemoryGraphStore
from src.adapters.storage.vector_store import InMemoryVectorStore
from src.compute_eval import InMemoryComputeTracker
from src.helpers import Document
from src.optimized_grag import OptimizedGraphRAG, OptimizedGraphRAGConfig


class FakeEmbedding(EmbeddingAdapter):
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    float(len(text)),
                    float(lower.count("apple")),
                    float(lower.count("china")),
                    float(lower.count("huawei")),
                    float(lower.count("sales")),
                    float(lower.count("competition")),
                    float(lower.count("supply")),
                    float(lower.count("market")),
                ]
            )
        return EmbeddingResult(vectors=vectors, model="fake-embedding")


class FakeLLM(LLM):
    def generate(self, prompt, *, system_prompt=None, **kwargs):
        return LLMResponse(
            text="Apple sales evidence is routed through compact graph capsules.",
            model="fake-llm",
        )


def main() -> None:
    optimized = OptimizedGraphRAG(
        graph_store=InMemoryGraphStore(),
        vector_store=InMemoryVectorStore(),
        embedding_adapter=FakeEmbedding(),
        llm=FakeLLM(),
        instrumentation=InMemoryComputeTracker(log_stages=False),
        config=OptimizedGraphRAGConfig(
            max_region_nodes=20,
            max_capsule_nodes=8,
            max_capsule_edges=20,
            anchors_per_region=3,
            max_capsules_per_region=3,
            first_stage_candidate_capsules=5,
            top_k_capsules=2,
            max_candidate_paths=10,
            final_paths_to_llm=3,
        ),
    )
    report = optimized.index_documents(
        [
            Document(
                id="d1",
                text="Apple sells iPhone in China. Apple competes with Huawei in China.",
            ),
            Document(
                id="d2",
                text="Huawei gained market share. China smartphone sales declined for Apple.",
            ),
            Document(
                id="d3",
                text="Apple Supply Chain includes Foxconn and iPhone Manufacturing.",
            ),
        ]
    )
    answer = optimized.answer("Why did Apple sales fall in China?")

    assert report.capsules > 0
    assert answer.answer
    assert answer.context.selected_capsules
    assert answer.context.context_text
    assert answer.usage["instrumentation"]["metrics"]


if __name__ == "__main__":
    main()
