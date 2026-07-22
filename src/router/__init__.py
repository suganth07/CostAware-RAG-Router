"""Production-oriented, offline-trained router foundation for three RAG systems."""

from src.router.config import BudgetConfig, RouterConfig
from src.router.constants import ACTION_IDS
from src.router.reward.oracle import select_budget_aware_oracle
from src.router.types import CandidateOutcome, MetricVector, QuestionRecord

__all__ = [
    "ACTION_IDS",
    "BudgetConfig",
    "CandidateOutcome",
    "MetricVector",
    "QuestionRecord",
    "RouterConfig",
    "select_budget_aware_oracle",
]
