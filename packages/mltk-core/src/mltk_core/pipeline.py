"""DataFrame-native Pipeline and train/test data splitting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline

from mltk_core.logging import get_logger

logger = get_logger("mltk.pipeline")


class Pipeline(SklearnPipeline):
    """
    DataFrame-preserving Pipeline for MLTK components.

    Fully compatible with scikit-learn, with built-in logging and progress visibility.
    """

    def __init__(
        self, steps: list[tuple[str, Any]], *, memory: Any = None, verbose: bool = False
    ) -> None:
        super().__init__(steps=steps, memory=memory, verbose=verbose)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params: Any) -> Pipeline:
        total = len(self.steps)
        for idx, (name, step) in enumerate(self.steps, start=1):
            if step is not None and step != "passthrough":
                logger.info(f"[{idx}/{total}] Fitting step: {name} ({step.__class__.__name__})")
        return super().fit(X, y, **fit_params)

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        return super().transform(X)

    def fit_transform(
        self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params: Any
    ) -> pd.DataFrame | np.ndarray:
        return super().fit_transform(X, y, **fit_params)


def split_data(
    df: pd.DataFrame,
    target: str | pd.Series,
    test_size: float = 0.20,
    stratify: bool = True,
    drop_target_from_X: bool = True,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into train and test sets with automatic target extraction and stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features (and optionally target).
    target : str or pd.Series
        Target column name in df, or a separate target pd.Series.
    test_size : float, default=0.20
        Fraction of dataset to reserve for test split.
    stratify : bool, default=True
        Automatically stratify if target is binary or multiclass classification.
    drop_target_from_X : bool, default=True
        Whether to drop target column from X if target was passed as a column name.
    random_state : int or None, default=42
        Seed for deterministic split reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        DataFrames for train/test features and Series for train/test targets.
    """
    if isinstance(target, str):
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found in DataFrame.")
        y = df[target].copy()
        X = df.drop(columns=[target]).copy() if drop_target_from_X else df.copy()
    else:
        y = target.copy()
        X = df.copy()

    # Determine whether stratification is viable
    stratify_target = None
    if stratify:
        y_clean = y.dropna()
        n_unique = y_clean.nunique()
        # Stratify only for classification tasks (cardinality <= 20) and no singletons
        if n_unique <= 20:
            counts = y_clean.value_counts()
            if (counts >= 2).all():
                stratify_target = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_target,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test
