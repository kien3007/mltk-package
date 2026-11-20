"""Feature scaler transformer preserving DataFrame column structure."""

from __future__ import annotations

from typing import Any, Literal
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class FeatureScaler(BaseFrameTransformer):
    """
    Scale numeric features while preserving DataFrame structure and column headers.
    """

    def __init__(
        self,
        method: Literal["standard", "robust", "minmax"] = "standard",
        columns: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.method = method
        self.columns = columns
        self.numeric_cols_: list[str] = []
        self._scaler: Any = None

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        if self.columns is not None:
            self.numeric_cols_ = [c for c in self.columns if c in X.columns]
        else:
            col_types = infer_column_types(X)
            self.numeric_cols_ = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]

        if not self.numeric_cols_:
            return self

        if self.method == "standard":
            self._scaler = StandardScaler()
        elif self.method == "robust":
            self._scaler = RobustScaler()
        elif self.method == "minmax":
            self._scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        self._scaler.fit(X[self.numeric_cols_])
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.numeric_cols_ or self._scaler is None:
            return X.copy()

        out = X.copy()
        scaled_vals = self._scaler.transform(out[self.numeric_cols_])
        out[self.numeric_cols_] = scaled_vals
        return out
