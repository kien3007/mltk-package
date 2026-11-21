"""Statistical and rule-based feature filters."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class VarianceFilter(BaseFrameTransformer):
    """
    Remove numeric columns with variance less than or equal to threshold.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.reasons_: dict[str, str] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        col_types = infer_column_types(X)
        self.selected_features_ = []
        self.dropped_features_ = []
        self.reasons_ = {}

        for col in X.columns:
            if col_types.get(col) == ColumnType.NUMERIC:
                var = float(X[col].var())
                if np.isnan(var) or var <= self.threshold:
                    self.dropped_features_.append(col)
                    self.reasons_[col] = f"Variance ({var}) <= threshold ({self.threshold})"
                else:
                    self.selected_features_.append(col)
            else:
                self.selected_features_.append(col)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self.selected_features_)


class MissingRateFilter(BaseFrameTransformer):
    """
    Remove columns whose missing value ratio exceeds threshold.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.reasons_: dict[str, str] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        missing_rates = X.isnull().mean()
        self.selected_features_ = []
        self.dropped_features_ = []
        self.reasons_ = {}

        for col, rate in missing_rates.items():
            if rate > self.threshold:
                self.dropped_features_.append(col)
                self.reasons_[col] = f"Missing rate {rate:.2f} > threshold {self.threshold:.2f}"
            else:
                self.selected_features_.append(col)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self.selected_features_)


class CorrelationFilter(BaseFrameTransformer):
    """
    Remove collinear numerical features whose pairwise correlation exceeds threshold.
    """

    def __init__(
        self, threshold: float = 0.95, method: Literal["pearson", "spearman"] = "pearson"
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.method = method
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.reasons_: dict[str, str] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        col_types = infer_column_types(X)
        numeric_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
        non_numeric = [c for c in X.columns if c not in numeric_cols]

        self.dropped_features_ = []
        self.reasons_ = {}

        if len(numeric_cols) > 1:
            corr_matrix = X[numeric_cols].corr(method=self.method).abs()
            to_drop = set()
            for i in range(len(numeric_cols)):
                col_i = numeric_cols[i]
                if col_i in to_drop:
                    continue
                for j in range(i + 1, len(numeric_cols)):
                    col_j = numeric_cols[j]
                    if col_j in to_drop:
                        continue
                    val = corr_matrix.loc[col_i, col_j]
                    if val >= self.threshold:
                        to_drop.add(col_j)
                        self.dropped_features_.append(col_j)
                        self.reasons_[col_j] = f"High correlation ({val:.3f}) with {col_i}"

            self.selected_features_ = [c for c in numeric_cols if c not in to_drop] + non_numeric
        else:
            self.selected_features_ = list(X.columns)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols].copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return list(self.selected_features_)


class StabilityFilter(BaseFrameTransformer):
    """
    Drop features that exhibit high Population Stability Index (PSI) between train and validation.
    """

    def __init__(self, threshold_psi: float = 0.25, val_df: pd.DataFrame | None = None) -> None:
        super().__init__()
        self.threshold_psi = threshold_psi
        self.val_df = val_df
        self.selected_features_: list[str] = []
        self.dropped_features_: list[str] = []
        self.psi_values_: dict[str, float] = {}

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """Calculate Population Stability Index between two 1D arrays."""
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) <= 1:
            return 0.0

        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        # Smooth counts with small constant to avoid divide by zero
        expected_pct = (expected_counts + 1e-4) / (len(expected) + 1e-4 * len(expected_counts))
        actual_pct = (actual_counts + 1e-4) / (len(actual) + 1e-4 * len(actual_counts))

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        self.selected_features_ = list(X.columns)
        self.dropped_features_ = []
        self.psi_values_ = {}

        if self.val_df is None:
            return self

        col_types = infer_column_types(X)
        for col in X.columns:
            if col in self.val_df.columns and col_types.get(col) == ColumnType.NUMERIC:
                psi = self.calculate_psi(
                    X[col].values.astype(float), self.val_df[col].values.astype(float)
                )
                self.psi_values_[col] = psi
                if psi > self.threshold_psi:
                    self.dropped_features_.append(col)

        self.selected_features_ = [c for c in X.columns if c not in self.dropped_features_]
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols].copy()
