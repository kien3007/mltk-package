"""Categorical encoder with cardinality auto-switching and leakage-free cross-fitting."""

from __future__ import annotations

from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from mltk_core.base import BaseFrameTransformer, ColumnType, infer_column_types


class CategoricalEncoder(BaseFrameTransformer):
    """
    Automatic categorical encoder that switches strategy based on cardinality:
    - low cardinality (< 10): One-Hot Encoding
    - mid cardinality (10 - 50): Ordinal Encoding
    - high cardinality (> 50): Target Encoding with K-Fold out-of-fold cross-fitting

    Strictly prevents target leakage by calculating out-of-fold target averages on train,
    and smoothed empirical priors on new test data.
    """

    def __init__(
        self,
        low_card_threshold: int = 10,
        high_card_threshold: int = 50,
        low_strategy: Literal["onehot", "ordinal", "target"] = "onehot",
        mid_strategy: Literal["ordinal", "target", "onehot"] = "ordinal",
        high_strategy: Literal["target", "ordinal", "onehot"] = "target",
        column_overrides: dict[str, str] | None = None,
        cv: int = 5,
        smoothing: float = 10.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.low_card_threshold = low_card_threshold
        self.high_card_threshold = high_card_threshold
        self.low_strategy = low_strategy
        self.mid_strategy = mid_strategy
        self.high_strategy = high_strategy
        self.column_overrides = column_overrides or {}
        self.cv = cv
        self.smoothing = smoothing
        self.random_state = random_state

        # Fitted parameters
        self.column_strategies_: dict[str, str] = {}
        self.onehot_categories_: dict[str, list[Any]] = {}
        self.ordinal_mappings_: dict[str, dict[Any, int]] = {}
        self.target_mappings_: dict[str, dict[Any, float]] = {}
        self.global_target_mean_: float = 0.0

    def _determine_strategy(self, col: str, n_unique: int) -> str:
        if col in self.column_overrides:
            return self.column_overrides[col]
        if n_unique <= self.low_card_threshold:
            return self.low_strategy
        elif n_unique <= self.high_card_threshold:
            return self.mid_strategy
        else:
            return self.high_strategy

    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        col_types = infer_column_types(X)
        categorical_cols = [c for c, t in col_types.items() if t == ColumnType.CATEGORICAL]

        if y is not None:
            y_arr = np.asarray(y, dtype=float)
            self.global_target_mean_ = float(np.nanmean(y_arr))
        else:
            self.global_target_mean_ = 0.0

        for col in categorical_cols:
            s = X[col].astype(str)
            unique_vals = sorted(list(s.unique()))
            strat = self._determine_strategy(col, len(unique_vals))
            self.column_strategies_[col] = strat

            if strat == "onehot":
                self.onehot_categories_[col] = unique_vals
            elif strat == "ordinal":
                self.ordinal_mappings_[col] = {val: idx for idx, val in enumerate(unique_vals, start=1)}
            elif strat == "target":
                if y is None:
                    # Fallback to ordinal if target not provided during fit
                    self.column_strategies_[col] = "ordinal"
                    self.ordinal_mappings_[col] = {val: idx for idx, val in enumerate(unique_vals, start=1)}
                else:
                    # Empirical smoothed target encoding
                    stats = pd.DataFrame({"cat": s, "y": y_arr}).groupby("cat")["y"].agg(["count", "mean"])
                    counts = stats["count"]
                    means = stats["mean"]
                    # Empirical Bayes smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
                    smoothed = (counts * means + self.smoothing * self.global_target_mean_) / (counts + self.smoothing)
                    self.target_mappings_[col] = smoothed.to_dict()

        return self

    def _fit_transform(self, X: pd.DataFrame, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        """Custom fit_transform with K-Fold cross-fitting for target encoding to prevent leakage."""
        self._fit(X, y)
        out = X.copy()

        # If target encoding is used on any column and y is present, apply out-of-fold target encoding
        target_cols = [c for c, strat in self.column_strategies_.items() if strat == "target"]
        if target_cols and y is not None:
            y_arr = np.asarray(y, dtype=float)
            n_unique_y = len(np.unique(y_arr[~np.isnan(y_arr)]))
            # Choose StratifiedKFold for binary classification, KFold for regression
            if n_unique_y == 2:
                kf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                splits = kf.split(X, y_arr)
            else:
                kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                splits = kf.split(X)

            oof_encodings = {col: np.full(len(X), fill_value=self.global_target_mean_, dtype=float) for col in target_cols}

            for train_idx, val_idx in splits:
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y_arr[train_idx]
                global_mean_fold = float(np.nanmean(y_train_fold))

                for col in target_cols:
                    s_train = X_train_fold[col].astype(str)
                    stats = pd.DataFrame({"cat": s_train, "y": y_train_fold}).groupby("cat")["y"].agg(["count", "mean"])
                    counts = stats["count"]
                    means = stats["mean"]
                    smoothed = (counts * means + self.smoothing * global_mean_fold) / (counts + self.smoothing)
                    mapping = smoothed.to_dict()

                    # Apply to validation fold
                    s_val = X.iloc[val_idx][col].astype(str)
                    oof_encodings[col][val_idx] = s_val.map(mapping).fillna(global_mean_fold).values

            for col in target_cols:
                out[col] = oof_encodings[col]

        # Apply ordinal and onehot encoding
        return self._apply_non_target_encoding(out)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        # For transform on new/test data, use global smoothed mappings
        for col, mapping in self.target_mappings_.items():
            if col in out.columns:
                s = out[col].astype(str)
                out[col] = s.map(mapping).fillna(self.global_target_mean_).astype(float)

        return self._apply_non_target_encoding(out)

    def _apply_non_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Apply Ordinal
        for col, mapping in self.ordinal_mappings_.items():
            if col in out.columns:
                s = out[col].astype(str)
                out[col] = s.map(mapping).fillna(0).astype(int)

        # Apply One-Hot
        for col, cats in self.onehot_categories_.items():
            if col in out.columns:
                s = out[col].astype(str)
                for cat in cats:
                    col_name = f"{col}_{cat}"
                    out[col_name] = (s == cat).astype(int)
                out.drop(columns=[col], inplace=True)

        return out
