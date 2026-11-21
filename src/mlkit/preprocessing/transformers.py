"""Preprocessing transformers with pandas DataFrame preservation and leakage protection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from mlkit.core.config import TaskConfig
from mlkit.core.registry import register
from mlkit.core.step import BaseStep


@register("preprocessing.type_caster")
class TypeCaster(BaseStep):
    """Cast column types, drop ID column, and normalize strings."""

    def __init__(
        self,
        dtypes: dict[str, str] | None = None,
        drop_id: bool = True,
    ) -> None:
        super().__init__()
        self.dtypes = dtypes
        self.drop_id = drop_id
        self.inferred_dtypes_: dict[str, str] = {}
        self.dropped_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> TypeCaster:
        id_col = cfg.id_col if cfg else None
        if self.drop_id and id_col and id_col in X.columns:
            self.dropped_cols_.append(id_col)

        # Detect prospective ID columns by name or uniqueness
        for c in X.columns:
            if c not in self.dropped_cols_:
                if c.lower() == "id" or (
                    (X[c].dtype == object or pd.api.types.is_string_dtype(X[c]))
                    and X[c].nunique() == len(X)
                    and "id" in c.lower()
                ):
                    self.dropped_cols_.append(c)

        for col in X.columns:
            if col in self.dropped_cols_:
                continue
            if self.dtypes and col in self.dtypes:
                self.inferred_dtypes_[col] = self.dtypes[col]
            elif pd.api.types.is_numeric_dtype(X[col]):
                self.inferred_dtypes_[col] = "float64"
            elif pd.api.types.is_datetime64_any_dtype(X[col]):
                self.inferred_dtypes_[col] = "datetime64[ns]"
            else:
                self.inferred_dtypes_[col] = "category"

        self.report_.add(
            "type_caster",
            "ok",
            f"Configured dtypes for {len(self.inferred_dtypes_)} columns, dropped ID cols: {self.dropped_cols_}",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=[c for c in self.dropped_cols_ if c in X.columns], errors="ignore")
        for col, target_dtype in self.inferred_dtypes_.items():
            if col in out.columns:
                try:
                    if target_dtype == "category":
                        out[col] = out[col].astype(str).astype("category")
                    else:
                        out[col] = out[col].astype(target_dtype)
                except Exception:
                    pass
        return out


@register("preprocessing.imputer")
class MissingImputer(BaseStep):
    """
    Impute numeric missing with median, categorical with mode, with missing indicators.
    """

    def __init__(
        self,
        numeric_strategy: Literal["median", "mean", "zero"] = "median",
        categorical_strategy: Literal["mode", "missing_value"] = "mode",
        add_indicator: bool = False,
    ) -> None:
        super().__init__()
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.add_indicator = add_indicator

        self.fill_values_: dict[str, Any] = {}
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.indicator_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> MissingImputer:
        self.fill_values_ = {}
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        for col in self.numeric_cols_:
            s = X[col].dropna()
            if self.add_indicator and X[col].isna().any():
                self.indicator_cols_.append(col)

            if s.empty:
                self.fill_values_[col] = 0.0
            elif self.numeric_strategy == "median":
                self.fill_values_[col] = float(s.median())
            elif self.numeric_strategy == "mean":
                self.fill_values_[col] = float(s.mean())
            else:
                self.fill_values_[col] = 0.0

        for col in self.categorical_cols_:
            s = X[col].dropna()
            if self.add_indicator and X[col].isna().any():
                self.indicator_cols_.append(col)

            if s.empty:
                self.fill_values_[col] = "__missing__"
            elif self.categorical_strategy == "mode":
                mode_val = s.mode()
                self.fill_values_[col] = mode_val.iloc[0] if not mode_val.empty else "__missing__"
            else:
                self.fill_values_[col] = "__missing__"

        self.report_.add(
            "missing_imputer",
            "ok",
            f"Computed impute statistics for {len(self.fill_values_)} columns (indicators: {len(self.indicator_cols_)}).",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.add_indicator:
            for col in self.indicator_cols_:
                if col in out.columns:
                    out[f"{col}_is_missing"] = out[col].isna().astype(int)

        for col, fill_val in self.fill_values_.items():
            if col in out.columns:
                out[col] = out[col].fillna(fill_val)
        return out


@register("preprocessing.rare_grouper")
class RareCategoryGrouper(BaseStep):
    """Group rare categories (frequency < min_freq) into '__rare__'."""

    def __init__(self, min_freq: float = 0.01, rare_label: str = "__rare__") -> None:
        super().__init__()
        self.min_freq = min_freq
        self.rare_label = rare_label
        self.frequent_categories_: dict[str, set[str]] = {}

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> RareCategoryGrouper:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        self.frequent_categories_ = {}

        for col in cat_cols:
            s = X[col].dropna().astype(str)
            freqs = s.value_counts(normalize=True)
            frequent = set(freqs[freqs >= self.min_freq].index)
            self.frequent_categories_[col] = frequent

        self.report_.add(
            "rare_grouper",
            "ok",
            f"Fitted rare grouping on {len(self.frequent_categories_)} categorical columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, frequent in self.frequent_categories_.items():
            if col in out.columns:
                out[col] = (
                    out[col]
                    .astype(str)
                    .apply(lambda val: val if val in frequent else self.rare_label)
                )
        return out


@register("preprocessing.categorical_encoder")
class CategoricalEncoder(BaseStep):
    """
    Adaptive categorical encoding:
    - Low cardinality (< 10): One-Hot
    - Mid cardinality (10–50): Ordinal
    - High cardinality (> 50): Out-of-fold Target Encoding
    """

    def __init__(
        self,
        low_card_threshold: int = 10,
        high_card_threshold: int = 50,
        cv: int = 5,
        smoothing: float = 10.0,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.low_card_threshold = low_card_threshold
        self.high_card_threshold = high_card_threshold
        self.cv = cv
        self.smoothing = smoothing
        self.random_state = random_state

        self.encoding_plan_: dict[str, str] = {}
        self.onehot_categories_: dict[str, list[str]] = {}
        self.ordinal_maps_: dict[str, dict[str, int]] = {}
        self.target_maps_: dict[str, dict[str, float]] = {}
        self.global_target_mean_: float = 0.0

    def _fit(
        self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None
    ) -> CategoricalEncoder:
        cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        self.encoding_plan_ = {}
        y_arr = np.asarray(y) if y is not None else None
        if y_arr is not None:
            if not pd.api.types.is_numeric_dtype(y_arr):
                unique_y = np.unique(y_arr[~pd.isna(y_arr)])
                y_num = np.zeros(len(y_arr), dtype=float)
                for i, u in enumerate(unique_y):
                    y_num[y_arr == u] = float(i)
                self.global_target_mean_ = float(np.nanmean(y_num))
                y_fit = y_num
            else:
                y_fit = y_arr.astype(float)
                self.global_target_mean_ = float(np.nanmean(y_fit))
        else:
            y_fit = None

        for col in cat_cols:
            n_unique = X[col].nunique(dropna=False)
            if n_unique <= self.low_card_threshold:
                strategy = "onehot"
                cats = [str(val) for val in X[col].dropna().unique()]
                self.onehot_categories_[col] = cats
            elif n_unique <= self.high_card_threshold or y_arr is None:
                strategy = "ordinal"
                cats = [str(val) for val in X[col].dropna().unique()]
                self.ordinal_maps_[col] = {val: idx for idx, val in enumerate(cats)}
            else:
                strategy = "target"
                # Compute smoothed global target encodings
                df_col = X[[col]].copy()
                df_col["_y"] = y_fit
                stats = df_col.groupby(col, observed=True)["_y"].agg(["count", "mean"])
                counts = stats["count"]
                means = stats["mean"]
                smoothed = (counts * means + self.smoothing * self.global_target_mean_) / (
                    counts + self.smoothing
                )
                self.target_maps_[col] = smoothed.to_dict()

            self.encoding_plan_[col] = strategy

        self.report_.add(
            "categorical_encoder",
            "ok",
            f"Encoded {len(self.encoding_plan_)} columns: {self.encoding_plan_}",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, strategy in self.encoding_plan_.items():
            if col not in out.columns:
                continue

            if strategy == "onehot":
                col_str = out[col].astype(str)
                for cat in self.onehot_categories_.get(col, []):
                    out[f"{col}_{cat}"] = (col_str == cat).astype(int)
                out.drop(columns=[col], inplace=True)

            elif strategy == "ordinal":
                col_map = self.ordinal_maps_.get(col, {})
                out[col] = out[col].astype(str).map(col_map).fillna(-1).astype(int)

            elif strategy == "target":
                col_map = self.target_maps_.get(col, {})
                out[col] = out[col].astype(str).map(col_map).fillna(self.global_target_mean_)

        return out


@register("preprocessing.scaler")
class FeatureScaler(BaseStep):
    """Scale numeric features with DataFrame column preservation."""

    def __init__(self, method: Literal["standard", "robust", "minmax"] = "standard") -> None:
        super().__init__()
        self.method = method
        self.scaler_: Any = None
        self.numeric_cols_: list[str] = []

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> FeatureScaler:
        self.numeric_cols_ = list(X.select_dtypes(include=[np.number]).columns)
        if self.method == "standard":
            self.scaler_ = StandardScaler()
        elif self.method == "robust":
            self.scaler_ = RobustScaler()
        else:
            self.scaler_ = MinMaxScaler()

        if self.numeric_cols_:
            self.scaler_.fit(X[self.numeric_cols_])

        self.report_.add(
            "feature_scaler",
            "ok",
            f"Fitted {self.method} scaler on {len(self.numeric_cols_)} columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.scaler_ is not None and self.numeric_cols_:
            usable = [c for c in self.numeric_cols_ if c in out.columns]
            if usable:
                out[usable] = self.scaler_.transform(out[usable])
        return out


@register("preprocessing.outlier_clipper")
class OutlierClipper(BaseStep):
    """Clip numeric feature outliers using IQR boundaries."""

    def __init__(self, factor: float = 1.5) -> None:
        super().__init__()
        self.factor = factor
        self.bounds_: dict[str, tuple[float, float]] = {}

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> OutlierClipper:
        num_cols = X.select_dtypes(include=[np.number]).columns
        self.bounds_ = {}
        for c in num_cols:
            s = X[c].dropna()
            if not s.empty:
                q25, q75 = float(np.percentile(s, 25)), float(np.percentile(s, 75))
                iqr = q75 - q25
                self.bounds_[c] = (q25 - self.factor * iqr, q75 + self.factor * iqr)

        self.report_.add(
            "outlier_clipper",
            "ok",
            f"Computed IQR bounds for {len(self.bounds_)} numeric columns.",
        )
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, (lower, upper) in self.bounds_.items():
            if c in out.columns:
                out[c] = out[c].clip(lower=lower, upper=upper)
        return out
