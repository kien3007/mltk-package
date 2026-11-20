"""Model-based feature importance selector."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class ModelImportanceSelector(BaseFrameTransformer):
    """
    Select top features using feature importances from a tree-based model.
    """

    def __init__(
        self,
        estimator: Any = None,
        top_k: int | None = None,
        threshold: float | None = None,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.top_k = top_k
        self.threshold = threshold
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.importances_: dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        if y is None:
            self.selected_features_ = list(X.columns)
            return self

        # Keep only numeric columns or numeric representations for fitting tree model
        col_types = infer_column_types(X)
        usable_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
        if not usable_cols:
            self.selected_features_ = list(X.columns)
            return self

        X_numeric = X[usable_cols].fillna(0)
        y_arr = np.asarray(y)

        if self.estimator is None:
            # Check if binary/multiclass or continuous
            n_unique_y = len(np.unique(y_arr[~pd.isna(y_arr)]))
            if n_unique_y <= 20:
                est = ExtraTreesClassifier(n_estimators=50, random_state=self.random_state)
            else:
                est = ExtraTreesRegressor(n_estimators=50, random_state=self.random_state)
        else:
            est = self.estimator

        est.fit(X_numeric, y_arr)
        raw_importances = est.feature_importances_

        sorted_pairs = sorted(
            zip(usable_cols, raw_importances), key=lambda x: x[1], reverse=True
        )
        self.importances_ = {col: float(imp) for col, imp in sorted_pairs}

        if self.top_k is not None:
            chosen = [col for col, _ in sorted_pairs[: self.top_k]]
        elif self.threshold is not None:
            chosen = [col for col, imp in sorted_pairs if imp >= self.threshold]
        else:
            # Default: keep features with non-zero importance
            chosen = [col for col, imp in sorted_pairs if imp > 0]

        # Re-include non-numeric columns untouched
        non_numeric = [c for c in X.columns if c not in usable_cols]
        self.selected_features_ = chosen + non_numeric
        self.dropped_features_ = [c for c in X.columns if c not in self.selected_features_]

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self.selected_features_)
