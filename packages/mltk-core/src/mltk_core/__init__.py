"""MLTK Core: Foundation interfaces, contracts, base classes, and utilities."""

from __future__ import annotations

from mltk_core.base import (
    BaseFrameTransformer,
    BaseMonitor,
    ColumnType,
    infer_column_types,
)
from mltk_core.check import Check, CheckResult, Report
from mltk_core.config import MltkBaseSettings
from mltk_core.contract import ContractViolation, define_schema, validate_contract
from mltk_core.exceptions import (
    LeakageError,
    MltkError,
    NotFittedError,
    SchemaError,
)
from mltk_core.logging import get_logger, get_run_id, set_run_id
from mltk_core.metadata import ArtifactMetadata, compute_data_hash, get_git_sha

__version__ = "0.1.0"

__all__ = [
    "ArtifactMetadata",
    "BaseFrameTransformer",
    "BaseMonitor",
    "Check",
    "CheckResult",
    "ColumnType",
    "ContractViolation",
    "LeakageError",
    "MltkBaseSettings",
    "MltkError",
    "NotFittedError",
    "Report",
    "SchemaError",
    "compute_data_hash",
    "define_schema",
    "get_git_sha",
    "get_logger",
    "get_run_id",
    "infer_column_types",
    "set_run_id",
    "validate_contract",
]
