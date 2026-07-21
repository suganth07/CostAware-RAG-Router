from src.compute_eval import InMemoryComputeTracker, NoopComputeTracker, OperationMetric

InMemoryInstrumentation = InMemoryComputeTracker
NoopInstrumentation = NoopComputeTracker

__all__ = ["InMemoryInstrumentation", "NoopInstrumentation", "OperationMetric"]
