"""Custom exceptions for mlkit."""

from __future__ import annotations


class MLKitError(Exception):
    """Base exception for all mlkit errors."""


class ValidationError(MLKitError):
    """Raised when data fails schema or contract validation."""


class DataLeakageError(MLKitError):
    """Raised when severe data leakage or target correlation is detected."""


class QualityGateError(MLKitError):
    """Raised when a quality gate check fails and execution must halt."""


class BaselineComparisonError(MLKitError):
    """Raised when a candidate model fails to outperform a naive baseline."""


class NotFittedError(MLKitError):
    """Raised when attempting to transform or predict with an unfitted step."""


class RegistryError(MLKitError):
    """Raised when registering or retrieving an invalid step from the registry."""
