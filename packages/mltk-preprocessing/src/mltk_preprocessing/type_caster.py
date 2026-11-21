"""Type casting and whitespace trimming transformer."""

from __future__ import annotations

from typing import Any
import pandas as pd

from mltk_core.base import BaseFrameTransformer


class TypeCaster(BaseFrameTransformer):
    """
    Ensure column datatypes conform to declared types and strip whitespace on strings.
    """

    def __init__(self, dtypes: dict[str, str | type] | None = None, strip_whitespace: bool = True) -> None:
        super().__init__()
        self.dtypes = dtypes
        self.strip_whitespace = strip_whitespace

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.strip_whitespace:
            for col in out.select_dtypes(include=["object"]).columns:
                out[col] = out[col].astype(str).str.strip().replace("nan", pd.NA)

        active_dtypes = self.dtypes or {}
        for col, target_dtype in active_dtypes.items():
            if col in out.columns:
                out[col] = out[col].astype(target_dtype)
        return out
