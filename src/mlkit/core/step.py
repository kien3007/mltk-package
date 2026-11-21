"""Base Step interface conforming to standard fit/transform/report_ contract."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mlkit.core.config import TaskConfig
from mlkit.core.errors import NotFittedError
from mlkit.core.report import Report


class BaseStep(BaseEstimator, TransformerMixin):
    """
    Unified Base Step for all MLKit pipeline components.

    Guarantees:
    - Input immutability (does not modify input DataFrame in place)
    - DataFrame return type with preserved or updated column names
    - Report generation accessible via self.report_
    - Compatibility with both scikit-learn API and MLKit TaskConfig
    """

    def __init__(self) -> None:
        self.report_ = Report(title=self.__class__.__name__)
        self.feature_names_in_: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
    ) -> BaseStep:
        """Fit step on training data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_in_ = [str(c) for c in X.columns]
        self._fit(X.copy(), y=y, cfg=cfg)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        if not self._is_fitted:
            raise NotFittedError(f"{self.__class__.__name__} is not fitted yet.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        res = self._transform(X.copy())
        if not isinstance(res, pd.DataFrame):
            res = pd.DataFrame(res)
        return res

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
        **fit_params: Any,
    ) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(X, y=y, cfg=cfg).transform(X)

    def run(
        self,
        X: pd.DataFrame,
        cfg: TaskConfig | None = None,
    ) -> tuple[pd.DataFrame, Report]:
        """Convenience method running fit_transform and returning (df_out, report)."""
        df_out = self.fit_transform(X, cfg=cfg)
        return df_out, self.report_

    def _fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
        cfg: TaskConfig | None = None,
    ) -> BaseStep:
        """Subclass override for fit logic."""
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subclass override for transform logic."""
        return X

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return feature names after transformation."""
        return list(self.feature_names_in_)
