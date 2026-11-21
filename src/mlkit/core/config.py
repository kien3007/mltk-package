"""Unified task configuration using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class TaskConfig(BaseModel):
    """
    Unified ML task configuration controlling all 6 pipeline stages.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    task_type: Literal["classification", "regression"] = "classification"
    target: str = "target"
    id_col: str | None = None
    time_col: str | None = None
    group_col: str | None = None
    seed: int = 42

    # Validation & Evaluation
    test_size: float = 0.2
    cv_folds: int = 5
    stratify: bool = True

    # Gates
    quality_gate: bool = True
    baseline_gate: bool = True

    # Stage-specific settings
    metric_thresholds: dict[str, float] = Field(default_factory=dict)
    qc: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)
    selection: dict[str, Any] = Field(default_factory=dict)
    modelling: dict[str, Any] = Field(default_factory=dict)
    monitoring: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TaskConfig:
        """Load TaskConfig from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Export TaskConfig to a YAML file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
