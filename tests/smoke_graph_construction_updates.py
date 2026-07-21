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
from src.optimized_grag.graph_construction import DocumentGraphConstructor


class FakeEmbedding(EmbeddingAdapter):
    def embed_texts(self, texts):
        return EmbeddingResult(
            vectors=[
                [
                    float(len(text)),
                    float(text.lower().count("apple")),
                    float(text.lower().count("china")),
                    float(text.lower().count("tesla")),
                ]
                for text in texts
            ],
            model="fake-embedding",
        )


class FakeLLM(LLM):
    def generate(self, prompt, *, system_prompt=None, **kwargs):
        return LLMResponse(text="ok", model="fake-llm")


def main() -> None:
    constructor = DocumentGraphConstructor()
    construction = constructor.from_documents(
        [
            Document(id="d1", text="Apple sells iPhone in China."),
            Document(id="d2", text="Tesla ships cars in China."),
        ],
        dataset_name="smoke",
    )
    assert construction.graph.nodes
    assert construction.report is not None
    assert construction.report.documents == 2

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
    optimized.index_documents([Document(id="base", text="Apple competes with Huawei in China.")])
    update = optimized.update_documents(
        [Document(id="new", text="Tesla competes with BYD in China.")],
        dataset_name="incremental-smoke",
    )
    assert update.graph_nodes > 0
    assert update.affected_node_ids
    assert update.capsules > 0
    assert optimized.answer("Who competes in China?").answer


if __name__ == "__main__":
    main()
