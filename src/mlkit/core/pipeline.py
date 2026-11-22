"""Unified MLKit Pipeline orchestrating sequential execution and reporting."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from mlkit.core.bundle import load_bundle, save_bundle
from mlkit.core.config import TaskConfig
from mlkit.core.errors import NotFittedError
from mlkit.core.report import Report


class Pipeline(BaseEstimator):
    """
    Sequential pipeline managing Steps and Estimator, collecting reports and metrics.
    """

    def __init__(self, steps: list[tuple[str, Any]], cv_scope: str = "pipeline") -> None:
        """
        Parameters
        ----------
        steps : list of (name, step)
            Transformers followed by a final estimator.
        cv_scope : {"pipeline", "model"}
            ``"pipeline"`` (default): when the final step supports it (``ModelArena``), every
            CV fold re-fits all upstream steps on the fold's training rows only, so encoders,
            scalers and feature selectors never see validation labels. ``"model"``: upstream
            steps are fitted once on all training rows and only the model is cross-validated
            (faster, but CV scores can be optimistic when supervised selection is used).
        """
        if cv_scope not in ("pipeline", "model"):
            raise ValueError("cv_scope must be 'pipeline' or 'model'")
        self.steps = steps
        self.cv_scope = cv_scope
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

        final_name, final = self.steps[-1]
        upstream = self.steps[:-1]
        final_kwargs = dict(routed[final_name])

        if (
            getattr(self, "cv_scope", "model") == "pipeline"
            and upstream
            and getattr(final, "supports_fold_data", False)
        ):
            final_kwargs["fold_data"] = self._build_fold_data(
                current_X, y, cfg, routed, final_kwargs
            )

        current_X = self._fit_upstream(upstream, current_X, y, cfg, routed)

        if hasattr(final, "fit"):
            if self._accepts(final.fit, "cfg"):
                final_kwargs["cfg"] = cfg
            final.fit(current_X, y, **final_kwargs)

        self._is_fitted = True
        return self

    def _fit_upstream(
        self,
        steps: list[tuple[str, Any]],
        X: pd.DataFrame,
        y: Any,
        cfg: TaskConfig | None,
        routed: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        """Fit (with ``fit_transform``) each transformer in ``steps``; return transformed X."""
        current_X = X
        for name, step in steps:
            kwargs = dict(routed.get(name, {}))
            if hasattr(step, "fit_transform"):
                if self._accepts(step.fit_transform, "cfg"):
                    kwargs["cfg"] = cfg
                current_X = step.fit_transform(current_X, y, **kwargs)
            else:
                if self._accepts(step.fit, "cfg"):
                    kwargs["cfg"] = cfg
                step.fit(current_X, y, **kwargs)
                current_X = step.transform(current_X)
            if not isinstance(current_X, pd.DataFrame):
                current_X = pd.DataFrame(current_X)
        return current_X

    @staticmethod
    def _take(value: Any, idx: np.ndarray, n_rows: int) -> Any:
        """Row-subset ``value`` if it is row-aligned with the training data."""
        if isinstance(value, (pd.Series, pd.DataFrame)) and len(value) == n_rows:
            return value.iloc[idx]
        if isinstance(value, np.ndarray) and len(value) == n_rows:
            return value[idx]
        if isinstance(value, list) and len(value) == n_rows:
            return [value[i] for i in idx]
        return value

    def _build_fold_data(
        self,
        X: pd.DataFrame,
        y: Any,
        cfg: TaskConfig | None,
        routed: dict[str, dict[str, Any]],
        final_kwargs: dict[str, Any],
    ) -> list[tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]]:
        """Re-fit clones of all upstream steps inside every CV fold (pipeline-level CV)."""
        final = self.steps[-1][1]
        n = len(X)
        y_arr = np.asarray(y)
        splits = final.make_splits(
            X,
            y_arr,
            cfg=cfg,
            groups=final_kwargs.get("groups"),
            time_index=final_kwargs.get("time_index"),
        )
        folds = []
        for tr, va in splits:
            fold_steps = [(name, clone(step)) for name, step in self.steps[:-1]]
            fold_routed = {
                name: {k: self._take(v, tr, n) for k, v in params.items()}
                for name, params in routed.items()
            }
            y_tr = self._take(y, tr, n) if isinstance(y, (pd.Series, list)) else y_arr[tr]
            X_tr = self._fit_upstream(fold_steps, X.iloc[tr], y_tr, cfg, fold_routed)
            X_va = X.iloc[va]
            for _, step in fold_steps:
                X_va = step.transform(X_va)
            folds.append((X_tr, y_arr[tr], X_va, y_arr[va]))
        return folds

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
