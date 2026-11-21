"""SHAP-based feature importance selector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


class SHAPSelector(BaseFrameTransformer):
    """
    Select features based on mean absolute SHAP values.
    """

    def __init__(
        self,
        top_k: int | None = None,
        threshold: float | None = None,
        sample_size: int = 500,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.threshold = threshold
        self.sample_size = sample_size
        self.random_state = random_state

        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.shap_importances_: dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        if y is None:
            self.selected_features_ = list(X.columns)
            return self

        col_types = infer_column_types(X)
        usable_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
        if not usable_cols:
            self.selected_features_ = list(X.columns)
            return self

        X_num = X[usable_cols].fillna(0)
        y_arr = np.asarray(y)

        # Subsample for fast SHAP computation
        if len(X_num) > self.sample_size:
            sample_df = X_num.sample(n=self.sample_size, random_state=self.random_state)
            sample_y = y_arr[sample_df.index]
        else:
            sample_df = X_num
            sample_y = y_arr

        # Fit simple surrogate tree
        n_unique_y = len(np.unique(sample_y[~pd.isna(sample_y)]))
        if n_unique_y <= 20:
            model = ExtraTreesClassifier(
                n_estimators=30, max_depth=5, random_state=self.random_state
            )
        else:
            model = ExtraTreesRegressor(
                n_estimators=30, max_depth=5, random_state=self.random_state
            )

        model.fit(sample_df, sample_y)

        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_df)

            # For binary classification, shap_values can be a list [class_0, class_1]
            if isinstance(shap_values, list):
                shap_vals = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0])
            elif hasattr(shap_values, "values"):
                shap_vals = np.abs(shap_values.values)
            else:
                shap_vals = np.abs(shap_values)

            mean_abs_shap = np.mean(shap_vals, axis=0)
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = np.mean(mean_abs_shap, axis=1)

            sorted_pairs = sorted(zip(usable_cols, mean_abs_shap), key=lambda x: x[1], reverse=True)
            self.shap_importances_ = {col: float(imp) for col, imp in sorted_pairs}

        except Exception:
            # Fallback to feature_importances_ if shap library is absent or fails on environment
            sorted_pairs = sorted(
                zip(usable_cols, model.feature_importances_), key=lambda x: x[1], reverse=True
            )
            self.shap_importances_ = {col: float(imp) for col, imp in sorted_pairs}

        if self.top_k is not None:
            chosen = [col for col, _ in sorted_pairs[: self.top_k]]
        elif self.threshold is not None:
            chosen = [col for col, imp in sorted_pairs if imp >= self.threshold]
        else:
            chosen = [col for col, imp in sorted_pairs if imp > 0]

        non_numeric = [c for c in X.columns if c not in usable_cols]
        self.selected_features_ = chosen + non_numeric
        self.dropped_features_ = [c for c in X.columns if c not in self.selected_features_]

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self.selected_features_)
