"""Feature selection transformers with drop reason auditing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep


@register("selection.variance")
class VarianceFilter(BaseStep):
    """Filter out features with variance below threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.selected_features_: list[str] = []
        self.dropped_features_: dict[str, str] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> VarianceFilter:
        self.dropped_features_ = {}
        selected = []
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                var = float(X[c].var())
                if np.isnan(var) or var <= self.threshold:
                    self.dropped_features_[c] = f"Variance {var:.6f} <= threshold {self.threshold}"
                else:
                    selected.append(c)
            else:
                selected.append(c)

        self.selected_features_ = selected
        self.report_.add(
            "variance_filter",
            "ok",
            f"Retained {len(self.selected_features_)} features, dropped {len(self.dropped_features_)} low-variance features.",
            {"dropped_reasons": self.dropped_features_},
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.selected_features_ if c in X.columns]
        return X[keep]


@register("selection.correlation")
class CorrelationFilter(BaseStep):
    """Remove highly correlated collinear features."""

    def __init__(self, threshold: float = 0.95) -> None:
        super().__init__()
        self.threshold = threshold
        self.selected_features_: list[str] = []
        self.dropped_features_: dict[str, str] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> CorrelationFilter:
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        self.dropped_features_ = {}

        if len(num_cols) > 1:
            corr_matrix = X[num_cols].corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for col in upper_tri.columns:
                high_corr = upper_tri[col][upper_tri[col] >= self.threshold]
                if not high_corr.empty:
                    lead_col = high_corr.index[0]
                    self.dropped_features_[col] = f"Correlation {high_corr.iloc[0]:.4f} >= {self.threshold} with '{lead_col}'"

        non_num = [c for c in X.columns if c not in num_cols]
        kept_num = [c for c in num_cols if c not in self.dropped_features_]
        self.selected_features_ = kept_num + non_num

        self.report_.add(
            "correlation_filter",
            "ok",
            f"Kept {len(self.selected_features_)} features, dropped {len(self.dropped_features_)} collinear features.",
            {"dropped_reasons": self.dropped_features_},
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.selected_features_ if c in X.columns]
        return X[keep]


@register("selection.mutual_info")
class MutualInformationSelector(BaseStep):
    """Select top-k features based on mutual information with target."""

    def __init__(self, top_k: int = 20, random_state: int = 42) -> None:
        super().__init__()
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.scores_: dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> MutualInformationSelector:
        if y is None:
            self.selected_features_ = list(X.columns)
            return self

        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            self.selected_features_ = list(X.columns)
            return self

        X_num = X[num_cols].fillna(0)
        y_arr = np.asarray(y)

        is_classif = cfg.task_type == "classification" if cfg else len(np.unique(y_arr)) <= 20
        if is_classif:
            mi = mutual_info_classif(X_num, y_arr, random_state=self.random_state)
        else:
            mi = mutual_info_regression(X_num, y_arr, random_state=self.random_state)

        ranked = sorted(zip(num_cols, mi), key=lambda x: x[1], reverse=True)
        self.scores_ = {c: float(s) for c, s in ranked}
        chosen = [c for c, _ in ranked[: self.top_k]]
        non_num = [c for c in X.columns if c not in num_cols]
        self.selected_features_ = chosen + non_num

        self.report_.add(
            "mutual_info_selector",
            "ok",
            f"Selected top {len(chosen)} numeric features by mutual information.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.selected_features_ if c in X.columns]
        return X[keep]


@register("selection.model_importance")
class ModelImportanceSelector(BaseStep):
    """Select top features using tree-based importance weights."""

    def __init__(self, top_k: int | None = None, threshold: float | None = None, random_state: int = 42) -> None:
        super().__init__()
        self.top_k = top_k
        self.threshold = threshold
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.importances_: dict[str, float] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> ModelImportanceSelector:
        if y is None:
            self.selected_features_ = list(X.columns)
            return self

        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            self.selected_features_ = list(X.columns)
            return self

        X_num = X[num_cols].fillna(0)
        y_arr = np.asarray(y)

        is_classif = cfg.task_type == "classification" if cfg else len(np.unique(y_arr)) <= 20
        est = (
            ExtraTreesClassifier(n_estimators=50, random_state=self.random_state)
            if is_classif
            else ExtraTreesRegressor(n_estimators=50, random_state=self.random_state)
        )
        est.fit(X_num, y_arr)

        sorted_pairs = sorted(zip(num_cols, est.feature_importances_), key=lambda x: x[1], reverse=True)
        self.importances_ = {c: float(v) for c, v in sorted_pairs}

        if self.top_k is not None:
            chosen = [c for c, _ in sorted_pairs[: self.top_k]]
        elif self.threshold is not None:
            chosen = [c for c, v in sorted_pairs if v >= self.threshold]
        else:
            chosen = [c for c, v in sorted_pairs if v > 0]

        non_num = [c for c in X.columns if c not in num_cols]
        self.selected_features_ = chosen + non_num

        self.report_.add(
            "model_importance_selector",
            "ok",
            f"Selected {len(chosen)} features based on tree feature importances.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep = [c for c in self.selected_features_ if c in X.columns]
        return X[keep]
