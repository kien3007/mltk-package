"""Model trainers supporting pandas DataFrame, unified predict_proba, and feature importances.

All trainers share :class:`BaseTrainer`, which implements fitting plumbing, prediction,
importances and sklearn parameter handling once. Subclasses only describe how to build
the underlying estimator (``_build``).

Extra keyword arguments (``**kwargs``) are forwarded to the underlying library estimator
and are part of ``get_params()``, so ``sklearn.base.clone`` (used by cross-validation and
Optuna tuning) preserves them.
"""

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

TaskType = Literal["classification", "regression"]


class BaseTrainer(BaseEstimator):
    """Shared implementation for all mlkit model trainers."""

    task_type: TaskType | None = None  # None => infer from the wrapped estimator
    kwargs: dict[str, Any]

    # ----------------------------------------------------- sklearn params
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = super().get_params(deep=deep)
        params.update(getattr(self, "kwargs", {}) or {})
        return params

    def set_params(self, **params: Any) -> BaseTrainer:
        explicit = set(self._get_param_names())
        extra = {k: v for k, v in params.items() if k not in explicit}
        super().set_params(**{k: v for k, v in params.items() if k in explicit})
        if extra:
            if not hasattr(self, "kwargs"):
                raise ValueError(
                    f"{type(self).__name__} does not accept extra parameters: {sorted(extra)}"
                )
            self.kwargs = {**self.kwargs, **extra}
        return self

    # ------------------------------------------------------- sklearn type
    def _resolved_type(self) -> str | None:
        """'classifier' / 'regressor' (from task_type, or the wrapped estimator) or None."""
        if self.task_type == "classification":
            return "classifier"
        if self.task_type == "regression":
            return "regressor"
        inner = getattr(self, "estimator", None)
        if inner is None:
            return None
        from sklearn.base import is_classifier, is_regressor

        return (
            "classifier" if is_classifier(inner) else "regressor" if is_regressor(inner) else None
        )

    @property
    def _estimator_type(self) -> str | None:  # sklearn < 1.6
        return self._resolved_type()

    def __sklearn_tags__(self) -> Any:  # sklearn >= 1.6
        tags = super().__sklearn_tags__()
        from sklearn.utils import ClassifierTags, RegressorTags

        kind = self._resolved_type()
        if kind == "classifier":
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
            tags.regressor_tags = None
        elif kind == "regressor":
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
            tags.classifier_tags = None
        return tags

    # ------------------------------------------------------------ fitting
    def _build(self, y: np.ndarray) -> Any:
        """Return an unfitted library estimator for target ``y``."""
        raise NotImplementedError

    #: Rename columns to f0..fN before handing data to the library (LightGBM/XGBoost reject
    #: names containing JSON characters such as [ ] { } " , :). Importances stay positional.
    _positional_names: bool = False

    def _prep(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if not self._positional_names:
            return X
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return df.set_axis([f"f{i}" for i in range(df.shape[1])], axis=1)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> BaseTrainer:
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.feature_names_in_ = [str(c) for c in df.columns]
        y_arr = np.asarray(y)
        self._estimator = self._build(y_arr)
        self._estimator.fit(self._prep(df), y_arr)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"{type(self).__name__} is not fitted.")

    # --------------------------------------------------------- prediction
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._estimator.predict(self._prep(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        if self.task_type == "regression":
            raise AttributeError("predict_proba is only available for classification.")
        if not hasattr(self._estimator, "predict_proba"):
            raise AttributeError(
                f"{type(self._estimator).__name__} does not support predict_proba."
            )
        return self._estimator.predict_proba(self._prep(X))

    @property
    def classes_(self) -> np.ndarray:
        self._check_fitted()
        return self._estimator.classes_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Native importances, |coef| (averaged over classes) for linear models, else zeros."""
        if not getattr(self, "_is_fitted", False):
            return np.array([])
        est = self._estimator
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_)
        coef = getattr(est, "coef_", None)
        if coef is not None:
            coef_arr = np.abs(np.asarray(coef))
            return coef_arr if coef_arr.ndim == 1 else coef_arr.mean(axis=0)
        return np.zeros(len(self.feature_names_in_))

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "_is_fitted", False))


def _binary_pos_weight(y: np.ndarray) -> float | None:
    """neg/pos ratio for binary targets (used as ``scale_pos_weight``), else None."""
    classes = np.unique(y[~pd.isna(y)])
    if len(classes) != 2:
        return None
    pos = np.sum(y == classes[1])
    return float(np.sum(y == classes[0]) / pos) if pos > 0 else None


@register("modelling.dummy")
class DummyTrainer(BaseTrainer):
    """Naive baseline estimator for baseline gating."""

    def __init__(self, task_type: TaskType = "classification") -> None:
        self.task_type = task_type

    def _build(self, y: np.ndarray) -> Any:
        if self.task_type == "classification":
            return DummyClassifier(strategy="prior")
        return DummyRegressor(strategy="mean")


@register("modelling.linear")
class LinearTrainer(BaseTrainer):
    """Logistic regression (classification) or Ridge (regression, ``alpha = 1/C``)."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        handle_imbalance: bool = True,
        task_type: TaskType = "classification",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

    def _build(self, y: np.ndarray) -> Any:
        if self.task_type == "classification":
            params: dict[str, Any] = {
                "C": self.C,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                **self.kwargs,
            }
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            return LogisticRegression(**params)
        return Ridge(alpha=1.0 / max(self.C, 1e-5), random_state=self.random_state, **self.kwargs)


@register("modelling.random_forest")
class RandomForestTrainer(BaseTrainer):
    """Random Forest trainer supporting both classification and regression."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        handle_imbalance: bool = True,
        task_type: TaskType = "classification",
        random_state: int = 42,
        n_jobs: int | None = -1,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_jobs = n_jobs
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

    def _build(self, y: np.ndarray) -> Any:
        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            **self.kwargs,
        }
        if self.task_type == "classification":
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)


@register("modelling.lightgbm")
class LGBMTrainer(BaseTrainer):
    """LightGBM wrapper; binary imbalance handled with ``scale_pos_weight``."""

    _positional_names = True

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        task_type: TaskType = "classification",
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

    def _build(self, y: np.ndarray) -> Any:
        import lightgbm as lgb

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
            if self.handle_imbalance:
                weight = _binary_pos_weight(y)
                if weight is not None:
                    params["scale_pos_weight"] = weight
                elif len(np.unique(y)) > 2:
                    params["class_weight"] = "balanced"  # multiclass
            return lgb.LGBMClassifier(**params)
        return lgb.LGBMRegressor(**params)


@register("modelling.hist_gb")
class HistGBTrainer(BaseTrainer):
    """scikit-learn HistGradientBoosting: fast gradient boosting with native missing values."""

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 31,
        handle_imbalance: bool = True,
        task_type: TaskType = "classification",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

    def _build(self, y: np.ndarray) -> Any:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        params: dict[str, Any] = {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "random_state": self.random_state,
            **self.kwargs,
        }
        if self.task_type == "classification":
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            return HistGradientBoostingClassifier(**params)
        return HistGradientBoostingRegressor(**params)

    @property
    def feature_importances_(self) -> np.ndarray:
        # HistGradientBoosting has no impurity importances; report zeros rather than fail
        return np.zeros(len(getattr(self, "feature_names_in_", [])))


@register("modelling.extra_trees")
class ExtraTreesTrainer(BaseTrainer):
    """Extremely randomised trees (often as accurate as Random Forest, faster to train)."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        handle_imbalance: bool = True,
        task_type: TaskType = "classification",
        random_state: int = 42,
        n_jobs: int | None = -1,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.handle_imbalance = handle_imbalance
        self.task_type = task_type
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def _build(self, y: np.ndarray) -> Any:
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            **self.kwargs,
        }
        if self.task_type == "classification":
            if self.handle_imbalance:
                params["class_weight"] = "balanced"
            return ExtraTreesClassifier(**params)
        return ExtraTreesRegressor(**params)


@register("modelling.xgboost")
class XGBTrainer(BaseTrainer):
    """XGBoost wrapper with native DataFrame support (requires the ``xgboost`` extra)."""

    _positional_names = True

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        task_type: TaskType = "classification",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.task_type = task_type
        self.random_state = random_state
        self.kwargs = kwargs

    def _build(self, y: np.ndarray) -> Any:
        import xgboost as xgb

        params: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "enable_categorical": True,
            **self.kwargs,
        }
        if self.task_type == "classification":
            return xgb.XGBClassifier(**params)
        return xgb.XGBRegressor(**params)


@register("modelling.sklearn_wrapper")
class SklearnModelWrapper(BaseTrainer):
    """Generic wrapper for any scikit-learn compatible estimator."""

    def __init__(self, estimator: Any = None) -> None:
        self.estimator = estimator

    def _build(self, y: np.ndarray) -> Any:
        if self.estimator is None:
            raise ValueError("An estimator instance must be provided.")
        return clone(self.estimator)
