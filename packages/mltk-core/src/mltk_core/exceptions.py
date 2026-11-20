"""Custom exceptions for MLTK."""

from __future__ import annotations


class MltkError(Exception):
    """Base exception for all MLTK errors."""


class SchemaError(MltkError):
    """Raised when DataFrame schema, column existence, or types do not match expectations."""


class NotFittedError(MltkError):
    """Raised when transform or evaluate is called before the component is fitted."""


class ContractViolation(MltkError):
    """Raised when a Pandera or runtime data contract validation fails."""


class LeakageError(MltkError):
    """Raised when data leakage or potential target contamination is detected."""
