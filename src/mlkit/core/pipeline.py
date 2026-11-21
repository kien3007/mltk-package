"""Unified MLKit Pipeline orchestrating sequential execution and reporting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from mlkit.core.bundle import load_bundle, save_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.errors import NotFittedError
from mlkit.core.report import Report


class Pipeline(BaseEstimator):
    """
    Sequential pipeline managing Steps and Estimator, collecting reports and metrics.
    """

    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        self.steps = steps
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    @property
    def named_steps(self) -> dict[str, Any]:
        return dict(self.steps)

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
    ) -> Pipeline:
        """Fit all steps sequentially on DataFrame."""
        current_X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in current_X.columns]

        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1

            if hasattr(step, "fit"):
                # Pass cfg if supported
                try:
                    step.fit(current_X, y, cfg=cfg)
                except TypeError:
                    step.fit(current_X, y)

            if not is_last and hasattr(step, "transform"):
                current_X = step.transform(current_X)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform through all transformer steps (excluding final model estimator)."""
        if not self._is_fitted:
            raise NotFittedError("Pipeline is not fitted yet.")

        current_X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if not is_last and hasattr(step, "transform"):
                current_X = step.transform(current_X)
        return current_X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data and run prediction on the final estimator."""
        if not self._is_fitted:
            raise NotFittedError("Pipeline is not fitted yet.")

        X_transformed = self.transform(X)
        final_step = self.steps[-1][1]
        if hasattr(final_step, "predict"):
            return final_step.predict(X_transformed)
        raise AttributeError(f"Final step '{self.steps[-1][0]}' does not support predict.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data and run predict_proba on the final estimator."""
        if not self._is_fitted:
            raise NotFittedError("Pipeline is not fitted yet.")

        X_transformed = self.transform(X)
        final_step = self.steps[-1][1]
        if hasattr(final_step, "predict_proba"):
            return final_step.predict_proba(X_transformed)
        raise AttributeError(f"Final step '{self.steps[-1][0]}' does not support predict_proba.")

    def report(self) -> Report:
        """Aggregate reports across all constituent steps."""
        master_report = Report(title="Pipeline Execution Report")
        for name, step in self.steps:
            if hasattr(step, "report_") and isinstance(step.report_, Report):
                master_report.merge(step.report_)
        return master_report

    def save(
        self,
        path: str | Path,
        cfg: TaskConfig,
        X_sample: pd.DataFrame | None = None,
        baseline: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Save this pipeline bundle to disk."""
        return save_bundle(
            self,
            path=path,
            cfg=cfg,
            X_sample=X_sample,
            baseline=baseline,
            metrics=metrics,
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[Pipeline, TaskConfig, dict[str, Any]]:
        """Load saved pipeline bundle from disk."""
        return load_bundle(path)
