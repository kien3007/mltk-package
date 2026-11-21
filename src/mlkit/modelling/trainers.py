"""Model trainers supporting pandas DataFrame, unified predict_proba, and feature importances."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from mlkit.core.errors import NotFittedError
from mlkit.core.registry import register


@register("modelling.dummy")
class DummyTrainer(BaseEstimator):
    """Naive baseline estimator for baseline gating."""

    def __init__(self, task_type: Literal["classification", "regression"] = "classification") -> None:
        self.task_type = task_type
        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> DummyTrainer:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        y_arr = np.asarray(y)

        if self.task_type == "classification":
            self._estimator = DummyClassifier(strategy="prior")
        else:
            self._estimator = DummyRegressor(strategy="mean")

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("DummyTrainer is not fitted.")
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("DummyTrainer is not fitted.")
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        return self._estimator.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.zeros(len(self.feature_names_in_))


@register("modelling.linear")
class LinearTrainer(BaseEstimator):
    """Linear / Logistic model trainer with auto balancing and weight importances."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        handle_imbalance: bool = True,
        task_type: Literal["classification", "regression"] = "classification",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> LinearTrainer:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        y_arr = np.asarray(y)

        if self.task_type == "classification":
            params = {
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                **self.kwargs,
            }
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            self._estimator = LogisticRegression(**params)
        else:
            alpha = 1.0 / max(self.C, 1e-5)
            self._estimator = Ridge(alpha=alpha, random_state=self.random_state, **self.kwargs)

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("LinearTrainer is not fitted.")
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("LinearTrainer is not fitted.")
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        return self._estimator.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted:
            return np.array([])
        coef = getattr(self._estimator, "coef_", None)
        if coef is None:
            return np.array([])
        coef_arr = np.asarray(coef)
        if coef_arr.ndim == 1:
            return np.abs(coef_arr)
        return np.mean(np.abs(coef_arr), axis=0)


@register("modelling.random_forest")
class RandomForestTrainer(BaseEstimator):
    """Random Forest trainer supporting both classification and regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        handle_imbalance: bool = True,
        task_type: Literal["classification", "regression"] = "classification",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> RandomForestTrainer:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        y_arr = np.asarray(y)

        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            **self.kwargs,
        }

        if self.task_type == "classification":
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            self._estimator = RandomForestClassifier(**params)
        else:
            self._estimator = RandomForestRegressor(**params)

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("RandomForestTrainer is not fitted.")
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("RandomForestTrainer is not fitted.")
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        return self._estimator.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted:
            return np.array([])
        return getattr(self._estimator, "feature_importances_", np.array([]))


@register("modelling.lightgbm")
class LGBMTrainer(BaseEstimator):
    """LightGBM wrapper with automatic DataFrame compatibility."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        task_type: Literal["classification", "regression"] = "classification",
        handle_imbalance: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.task_type = task_type
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> LGBMTrainer:
        import lightgbm as lgb

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        y_arr = np.asarray(y)

        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "verbosity": -1,
            **self.kwargs,
        }

        if self.task_type == "classification":
            unique_targets = np.unique(y_arr[~pd.isna(y_arr)])
            if self.handle_imbalance and len(unique_targets) == 2:
                neg_count = np.sum(y_arr == unique_targets[0])
                pos_count = np.sum(y_arr == unique_targets[1])
                if pos_count > 0:
                    params["scale_pos_weight"] = float(neg_count / pos_count)
            self._estimator = lgb.LGBMClassifier(**params)
        else:
            self._estimator = lgb.LGBMRegressor(**params)

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("LGBMTrainer is not fitted.")
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("LGBMTrainer is not fitted.")
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        return self._estimator.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted:
            return np.array([])
        return getattr(self._estimator, "feature_importances_", np.array([]))


@register("modelling.xgboost")
class XGBTrainer(BaseEstimator):
    """XGBoost wrapper with native DataFrame support."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        task_type: Literal["classification", "regression"] = "classification",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> XGBTrainer:
        import xgboost as xgb

        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        y_arr = np.asarray(y)

        params = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "enable_categorical": True,
            **self.kwargs,
        }

        if self.task_type == "classification":
            self._estimator = xgb.XGBClassifier(**params)
        else:
            self._estimator = xgb.XGBRegressor(**params)

        self._estimator.fit(df, y_arr)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("XGBTrainer is not fitted.")
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("XGBTrainer is not fitted.")
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification.")
        return self._estimator.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted:
            return np.array([])
        return getattr(self._estimator, "feature_importances_", np.array([]))


@register("modelling.sklearn_wrapper")
class SklearnModelWrapper(BaseEstimator):
    """Generic wrapper for any custom scikit-learn estimator."""

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator
        self._estimator: Any = None
        self._is_fitted: bool = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> SklearnModelWrapper:
        if self.estimator is None:
            raise ValueError("An estimator instance must be provided.")
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        self._estimator = clone(self.estimator)
        self._estimator.fit(df, np.asarray(y))
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("SklearnModelWrapper is not fitted.")
        return self._estimator.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise NotFittedError("SklearnModelWrapper is not fitted.")
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)
        raise AttributeError(f"{type(self._estimator).__name__} does not support predict_proba.")

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self._is_fitted:
            return np.array([])
        if hasattr(self._estimator, "feature_importances_"):
            return np.asarray(self._estimator.feature_importances_)
        elif hasattr(self._estimator, "coef_"):
            coef = np.asarray(self._estimator.coef_)
            return np.abs(coef) if coef.ndim == 1 else np.mean(np.abs(coef), axis=0)
        return np.array([])
