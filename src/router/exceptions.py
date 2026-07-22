class RouterError(Exception):
    """Base exception for router-specific failures."""


class ArtifactCompatibilityError(RouterError):
    """Raised when a persisted router artifact is incompatible with runtime."""


class OutcomeContractError(RouterError):
    """Raised when an outcome cannot safely enter the training dataset."""
