"""Ratio and difference feature creation transformer."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class RatioFeatures(BaseFrameTransformer):
    """
    Generate ratio and difference interactions between pairs of numeric columns.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]] | str = "auto",
        create_difference: bool = True,
        missing_column: Literal["skip", "raise"] = "skip",
        fill_inf: float = 0.0,
    ) -> None:
        super().__init__()
        self.pairs = pairs
        self.create_difference = create_difference
        self.missing_column = missing_column
        self.fill_inf = fill_inf
        self.active_pairs_: list[tuple[str, str]] = []

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        if isinstance(self.pairs, list):
            self.active_pairs_ = self.pairs
        else:
            col_types = infer_column_types(X)
            numeric_cols = [c for c, t in col_types.items() if t == ColumnType.NUMERIC]
            # Suggest pairs between first few numeric features if auto
            self.active_pairs_ = []
            if len(numeric_cols) >= 2:
                for i in range(min(5, len(numeric_cols))):
                    for j in range(i + 1, min(5, len(numeric_cols))):
                        self.active_pairs_.append((numeric_cols[i], numeric_cols[j]))

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for num_col, den_col in self.active_pairs_:
            if num_col not in out.columns or den_col not in out.columns:
                if self.missing_column == "raise":
                    raise KeyError(f"Missing columns for ratio: ({num_col}, {den_col})")
                continue

            num_vals = pd.to_numeric(out[num_col], errors="coerce")
            den_vals = pd.to_numeric(out[den_col], errors="coerce")

            # Ratio: num / den
            ratio_name = f"{num_col}_ratio_{den_col}"
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_vals = num_vals / den_vals
                ratio_vals = ratio_vals.replace([np.inf, -np.inf], self.fill_inf).fillna(self.fill_inf)
            out[ratio_name] = ratio_vals.astype(float)

            # Difference: num - den
            if self.create_difference:
                diff_name = f"{num_col}_diff_{den_col}"
                out[diff_name] = (num_vals - den_vals).astype(float)

        return out
