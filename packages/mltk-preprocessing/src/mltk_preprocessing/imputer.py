"""Leakage-free missing value imputation transformer."""

from __future__ import annotations

from typing import Any, Literal
import pandas as pd

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class MissingImputer(BaseFrameTransformer):
    """
    Impute missing values with automatic column type detection.

    Strategies learned on the training set are preserved and applied to test sets,
    preventing data leakage.
    """

    def __init__(
        self,
        numeric_strategy: Literal["median", "mean", "zero"] = "median",
        categorical_strategy: Literal["most_frequent", "constant"] = "most_frequent",
        fill_value: Any = "missing",
        column_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.fill_value = fill_value
        self.column_overrides = column_overrides
        self.statistics_: dict[str, Any] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        col_types = infer_column_types(X)
        self.statistics_ = {}
        overrides = self.column_overrides or {}

        for col, ctype in col_types.items():
            if col in overrides:
                override = overrides[col]
                if isinstance(override, str) and override == "median":
                    val = X[col].median()
                elif isinstance(override, str) and override == "mean":
                    val = X[col].mean()
                elif isinstance(override, str) and override == "most_frequent":
                    mode_series = X[col].mode(dropna=True)
                    val = mode_series.iloc[0] if not mode_series.empty else self.fill_value
                else:
                    val = override
                self.statistics_[col] = val
                continue

            if ctype == ColumnType.NUMERIC:
                if self.numeric_strategy == "median":
                    val = X[col].median()
                elif self.numeric_strategy == "mean":
                    val = X[col].mean()
                elif self.numeric_strategy == "zero":
                    val = 0.0
                else:
                    val = 0.0
                # If entire column was null on train
                if pd.isna(val):
                    val = 0.0
                self.statistics_[col] = val

            elif ctype == ColumnType.CATEGORICAL:
                if self.categorical_strategy == "most_frequent":
                    mode_series = X[col].mode(dropna=True)
                    val = mode_series.iloc[0] if not mode_series.empty else str(self.fill_value)
                else:
                    val = str(self.fill_value)
                self.statistics_[col] = val

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, fill_val in self.statistics_.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill_val)
        return out
