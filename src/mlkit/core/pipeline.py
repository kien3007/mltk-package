"""Unified MLKit Pipeline orchestrating sequential execution and reporting."""

from __future__ import annotations

import inspect
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

    @staticmethod
    def _accepts(fn: Any, param: str) -> bool:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False
        return param in sig.parameters or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
        **fit_params: Any,
    ) -> Pipeline:
        """Fit all steps sequentially on DataFrame.

        Intermediate steps are fitted with ``fit_transform`` so that steps which
        produce out-of-fold training outputs (e.g. target encoding) can do so.
        Extra keyword arguments are routed sklearn-style: ``<step>__<param>=value``.
        """
        current_X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in current_X.columns]

        routed: dict[str, dict[str, Any]] = {name: {} for name, _ in self.steps}
        for key, value in fit_params.items():
            step_name, sep, param = key.partition("__")
            if not sep or step_name not in routed:
                raise ValueError(f"Invalid fit parameter '{key}'. Use '<step>__<param>'.")
            routed[step_name][param] = value

        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            kwargs = dict(routed[name])

            if not is_last and hasattr(step, "fit_transform"):
                if self._accepts(step.fit_transform, "cfg"):
                    kwargs["cfg"] = cfg
                current_X = step.fit_transform(current_X, y, **kwargs)
                if not isinstance(current_X, pd.DataFrame):
                    current_X = pd.DataFrame(current_X)
                continue

            if hasattr(step, "fit"):
                if self._accepts(step.fit, "cfg"):
                    kwargs["cfg"] = cfg
                step.fit(current_X, y, **kwargs)

            if not is_last and hasattr(step, "transform"):
                current_X = step.transform(current_X)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform through all transformer steps (excluding final model estimator)."""
        if not self._is_fitted:
            raise NotFittedError("Pipeline is not fitted yet.")

        current_X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for i, (_name, step) in enumerate(self.steps):
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
        for _name, step in self.steps:
            if hasattr(step, "report_") and isinstance(step.report_, Report):
                master_report.merge(step.report_)
        return master_report

    def to_html_report(
        self,
        output_path: str | Path | None = None,
        summary_stats: dict[str, Any] | None = None,
        figures: list[tuple[str, Any]] | None = None,
    ) -> str:
        """Export all step reports and diagnostic figures to an interactive standalone HTML report."""
        from mlkit.viz.html_report import generate_html_report

        return generate_html_report(
            self.report(),
            title="MLKit Pipeline Execution & Diagnostic Report",
            summary_stats=summary_stats,
            figures=figures,
            output_path=output_path,
        )

    def save(
        self,
        path: str | Path,
        cfg: TaskConfig,
        X_sample: pd.DataFrame | None = None,
        baseline: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        report: Report | None = None,
    ) -> Path:
        """Save this pipeline bundle to disk (the aggregated step report is included)."""
        return save_bundle(
            self,
            path=path,
            cfg=cfg,
            X_sample=X_sample,
            baseline=baseline,
            metrics=metrics,
            report=report if report is not None else self.report(),
        )

    @classmethod
    def load(cls, path: str | Path) -> tuple[Pipeline, TaskConfig, dict[str, Any]]:
        """Load saved pipeline bundle from disk."""
        return load_bundle(path)
