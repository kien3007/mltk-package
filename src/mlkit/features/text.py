"""Lightweight features from free-text columns (no external NLP dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep
from mlkit.qc_eda.profile import ProfileOptions, infer_column


@register("features.text")
class TextFeatures(BaseStep):
    """Replace free-text columns with simple statistics.

    Columns are taken from ``columns`` or detected at fit time as semantic type ``text``
    (long strings / many words, mostly unique). Each becomes:
    ``<col>_n_chars``, ``<col>_n_words``, ``<col>_digit_ratio``, ``<col>_upper_ratio``,
    ``<col>_punct_ratio`` and ``<col>_is_missing``; the raw text is dropped (``drop_original``).

    This keeps text from being treated as a useless million-level categorical. For semantic
    content (topics, sentiment) plug in a dedicated NLP step before this one.
    """

    def __init__(
        self,
        columns: list[str] | None = None,
        drop_original: bool = True,
        profile_options: ProfileOptions | None = None,
    ) -> None:
        super().__init__()
        self.columns = columns
        self.drop_original = drop_original
        self.profile_options = profile_options
        self.text_columns_: list[Any] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> TextFeatures:
        if self.columns is not None:
            self.text_columns_ = [c for c in self.columns if c in X.columns]
        else:
            candidates = [
                c for c in X.columns if X[c].dtype == object or pd.api.types.is_string_dtype(X[c])
            ]
            self.text_columns_ = [
                c
                for c in candidates
                if infer_column(X[c], str(c), self.profile_options).semantic_type == "text"
            ]
        self.report_.add(
            "text_features",
            "ok",
            f"Detected {len(self.text_columns_)} free-text columns: {self.text_columns_}",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.text_columns_:
            if col not in out.columns:
                continue
            raw = out[col]
            s = raw.astype("string").fillna("")
            n_chars = s.str.len().astype(float)
            denom = n_chars.replace(0, np.nan)
            out[f"{col}_n_chars"] = n_chars
            out[f"{col}_n_words"] = s.str.count(r"\S+").astype(float)
            out[f"{col}_digit_ratio"] = (s.str.count(r"\d") / denom).fillna(0.0).astype(float)
            out[f"{col}_upper_ratio"] = (s.str.count(r"[A-Z]") / denom).fillna(0.0).astype(float)
            out[f"{col}_punct_ratio"] = (s.str.count(r"[^\w\s]") / denom).fillna(0.0).astype(float)
            out[f"{col}_is_missing"] = raw.isna().astype(int)
            if self.drop_original:
                out = out.drop(columns=[col])
        return out
