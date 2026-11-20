"""Base classes and column type inference for MLTK components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mltk_core.exceptions import NotFittedError, SchemaError


class ColumnType(str, Enum):
    """Inferred column semantic type."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    OTHER = "other"


def infer_column_types(df: pd.DataFrame) -> dict[str, ColumnType]:
    """
    Infer semantic types for all columns in a pandas DataFrame.
    
    Returns a mapping of column name to ColumnType (NUMERIC, CATEGORICAL, DATETIME, OTHER).
    """
    col_types: dict[str, ColumnType] = {}
    for col in df.columns:
        s = df[col]
        dtype = s.dtype

        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            col_types[str(col)] = ColumnType.NUMERIC
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            col_types[str(col)] = ColumnType.DATETIME
        elif isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            col_types[str(col)] = ColumnType.CATEGORICAL
        else:
            col_types[str(col)] = ColumnType.OTHER
    return col_types


class BaseFrameTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for DataFrame-in, DataFrame-out transformers.

    Features:
    - Fully compatible with scikit-learn Pipeline and GridSearchCV.
    - Preserves column names and returns pandas.DataFrame.
    - Never mutates input DataFrame in place.
    - Tracks fitted feature names and validates consistency on transform.
    """

    def __init__(self) -> None:
        self._is_fitted: bool = False
        self._feature_names_in: list[str] = []
        self._n_features_in: int = 0

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> BaseFrameTransformer:
        """Validate input, record feature names, and fit transformer parameters."""
        df = self._ensure_dataframe(X)
        self._feature_names_in = [str(c) for c in df.columns]
        self._n_features_in = len(self._feature_names_in)
        self._fit(df, y)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: Any = None) -> BaseFrameTransformer:
        """Internal fit implementation. Must be implemented by subclasses."""

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Validate input and apply transformation to a clean copy of the DataFrame."""
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"Transformer {self.__class__.__name__} is not fitted yet.")

        df = self._ensure_dataframe(X)
        self._validate_feature_names(df)

        # Always work on a copy to prevent in-place mutation of user data
        out = self._transform(df.copy())
        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"{self.__class__.__name__} must return a pandas.DataFrame, got {type(out)}."
            )
        return out

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal transform implementation. Must be implemented by subclasses."""

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params: Any) -> pd.DataFrame:
        """Fit to data, then transform it. Subclasses can override _fit_transform for optimizations."""
        df = self._ensure_dataframe(X)
        if hasattr(self, "_fit_transform"):
            self._feature_names_in = [str(c) for c in df.columns]
            self._n_features_in = len(self._feature_names_in)
            out = getattr(self, "_fit_transform")(df.copy(), y, **fit_params)
            self._is_fitted = True
            if not isinstance(out, pd.DataFrame):
                raise TypeError(f"_fit_transform must return pd.DataFrame, got {type(out)}.")
            return out
        return self.fit(df, y).transform(df)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names after transform."""
        if hasattr(self, "_feature_names_out") and getattr(self, "_feature_names_out"):
            return list(getattr(self, "_feature_names_out"))
        if self._feature_names_in:
            return list(self._feature_names_in)
        if input_features is not None:
            return list(input_features)
        return []

    def _ensure_dataframe(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Convert array to DataFrame if needed, preserving names if available."""
        if isinstance(X, pd.DataFrame):
            return X
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # Use cached feature names if dimension matches
            if getattr(self, "_feature_names_in", None) and len(self._feature_names_in) == X.shape[1]:
                cols = self._feature_names_in
            else:
                cols = [f"x{i}" for i in range(X.shape[1])]
            return pd.DataFrame(X, columns=cols)
        raise TypeError(f"Expected pandas.DataFrame or numpy.ndarray, got {type(X)}")

    def _validate_feature_names(self, X: pd.DataFrame) -> None:
        """Check that columns required by the fitted transformer are present."""
        if not getattr(self, "_feature_names_in", None):
            return
        current_cols = [str(c) for c in X.columns]
        missing = set(self._feature_names_in) - set(current_cols)
        if missing:
            raise SchemaError(
                f"{self.__class__.__name__} missing expected columns from fit: {sorted(list(missing))}"
            )


class BaseMonitor(ABC):
    """
    Base class for statistical monitoring and drift detection.
    """

    def __init__(self) -> None:
        self.reference_stats_: dict[str, Any] = {}
        self._is_fitted: bool = False

    def fit(self, reference: pd.DataFrame) -> BaseMonitor:
        """Calculate and store baseline statistics from the reference (training) dataset."""
        if not isinstance(reference, pd.DataFrame):
            raise TypeError(f"Reference must be a pandas.DataFrame, got {type(reference)}")
        self._fit_reference(reference)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit_reference(self, reference: pd.DataFrame) -> None:
        """Extract baseline distributions and statistics."""

    @abstractmethod
    def evaluate(self, current: pd.DataFrame) -> Any:
        """Compare current dataset against reference baseline and return a Report."""
