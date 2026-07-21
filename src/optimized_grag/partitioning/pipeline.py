from __future__ import annotations

from dataclasses import dataclass, field, replace

from logger import get_logger
from src.compute_eval import ComputeTracker, NoopComputeTracker
from src.optimized_grag.config import OptimizedGraphRAGConfig
from src.optimized_grag.graph import LocalGraph
from src.optimized_grag.partitioning.anchors import AnchorSelector, AnchorSelectorConfig
from src.optimized_grag.partitioning.capsule_builder import CapsuleBuilder, CapsuleBuilderConfig
from src.optimized_grag.partitioning.hubs import HubDetector, HubDetectorConfig, HubProfile
from src.optimized_grag.partitioning.regions import RegionPartitioner, RegionPartitionerConfig
from src.optimized_grag.partitioning.supergraph import CapsuleSupergraph, build_capsule_supergraph
from src.optimized_grag.partitioning.types import GraphCapsule, Region
from src.optimized_grag.partitioning.validation import CapsuleValidator, CapsuleValidatorConfig


@dataclass(slots=True)
class CapsuleBuildResult:
    hubs: HubProfile
    regions: list[Region]
    capsules: list[GraphCapsule]
    supergraph: CapsuleSupergraph
    metadata: dict = field(default_factory=dict)


class CapsuleConstructionPipeline:
    def __init__(
        self,
        *,
        config: OptimizedGraphRAGConfig,
        instrumentation: ComputeTracker | None = None,
    ) -> None:
        self.config = config
        self.instrumentation = instrumentation or NoopComputeTracker()
        self.hub_detector = HubDetector(HubDetectorConfig(config.hub_percentile))
        self.partitioner = RegionPartitioner(RegionPartitionerConfig(config.max_region_nodes))
        self.anchor_selector = AnchorSelector(AnchorSelectorConfig(config.anchors_per_region))
        self.builder = CapsuleBuilder(
            CapsuleBuilderConfig(
                max_hops=config.max_hops,
                max_capsule_nodes=config.max_capsule_nodes,
                max_capsule_edges=config.max_capsule_edges,
                max_hub_neighbors=config.max_hub_neighbors,
                max_overlap_ratio=config.max_overlap_ratio,
                max_capsules_per_region=config.max_capsules_per_region,
            )
        )
        self.validator = CapsuleValidator(
            CapsuleValidatorConfig(
                max_capsule_nodes=config.max_capsule_nodes,
                max_capsule_edges=config.max_capsule_edges,
                max_overlap_ratio=config.max_overlap_ratio,
            )
        )
        self._logger = get_logger(__name__)

    def build(self, graph: LocalGraph) -> CapsuleBuildResult:
        with self.instrumentation.track("optimized.detect_hubs", {"nodes": len(graph.nodes)}):
            hubs = self.hub_detector.detect(graph)
        with self.instrumentation.track("optimized.partition_regions", {"nodes": len(graph.nodes)}):
            regions = self.partitioner.partition(graph)

        capsules: list[GraphCapsule] = []
        with self.instrumentation.track("optimized.build_capsules", {"regions": len(regions)}):
            for region in regions:
                anchors = self.anchor_selector.select(graph, region, hubs)
                for capsule in self.builder.build_for_region(graph, region, anchors, hubs):
                    repaired = self.validator.repair(graph, capsule, hubs)
                    if repaired is not None:
                        capsules.append(repaired)

        capsules = _deduplicate_capsules(capsules)
        coverage_capsules = 0
        if self.config.ensure_node_coverage:
            with self.instrumentation.track(
                "optimized.ensure_capsule_coverage",
                {"nodes": len(graph.nodes), "capsules": len(capsules)},
            ):
                capsules, coverage_capsules = self._ensure_node_coverage(
                    graph=graph,
                    regions=regions,
                    hubs=hubs,
                    capsules=capsules,
                )

        covered_node_ids = set().union(*(capsule.node_ids for capsule in capsules)) if capsules else set()
        uncovered_node_ids = set(graph.nodes) - covered_node_ids
        if uncovered_node_ids:
            raise RuntimeError(
                "Capsule coverage invariant failed for "
                f"{len(uncovered_node_ids)} graph nodes; examples={sorted(uncovered_node_ids)[:5]}"
            )

        with self.instrumentation.track("optimized.build_capsule_supergraph", {"capsules": len(capsules)}):
            supergraph = build_capsule_supergraph(capsules)

        self._logger.info(
            "built capsules=%s regions=%s hubs=%s",
            len(capsules),
            len(regions),
            len(hubs.hub_node_ids),
        )
        return CapsuleBuildResult(
            hubs=hubs,
            regions=regions,
            capsules=capsules,
            supergraph=supergraph,
            metadata={
                "hub_threshold": hubs.threshold,
                "hub_count": len(hubs.hub_node_ids),
                "region_count": len(regions),
                "capsule_count": len(capsules),
                "coverage_capsule_count": coverage_capsules,
                "covered_node_count": len(covered_node_ids),
                "uncovered_node_count": len(uncovered_node_ids),
                "node_coverage_rate": (
                    len(covered_node_ids) / len(graph.nodes) if graph.nodes else 1.0
                ),
                "construction_schema_version": 3,
            },
        )

    def _ensure_node_coverage(
        self,
        *,
        graph: LocalGraph,
        regions: list[Region],
        hubs: HubProfile,
        capsules: list[GraphCapsule],
    ) -> tuple[list[GraphCapsule], int]:
        """Add bounded ego capsules only where normal anchor coverage has gaps."""
        covered = set().union(*(capsule.node_ids for capsule in capsules)) if capsules else set()
        result = list(capsules)
        added = 0

        for region in regions:
            for node_id in sorted(region.node_ids - covered):
                if node_id in covered:
                    continue
                capsule = self.builder.build(graph, region, node_id, hubs)
                if capsule is None:
                    continue
                repaired = self.validator.repair(graph, capsule, hubs)
                if repaired is None:
                    continue
                repaired = replace(
                    repaired,
                    metadata={**repaired.metadata, "coverage_fallback": True},
                )
                result.append(repaired)
                covered.update(repaired.node_ids)
                added += 1

        return _deduplicate_capsules(result), added


def _deduplicate_capsules(capsules: list[GraphCapsule]) -> list[GraphCapsule]:
    kept: list[GraphCapsule] = []
    signatures: set[tuple[str, ...]] = set()
    for capsule in capsules:
        signature = tuple(sorted(capsule.node_ids))
        if signature in signatures:
            continue
        signatures.add(signature)
        kept.append(capsule)
    return kept
