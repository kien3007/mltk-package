"""Pipeline-level cross-validation: upstream steps are re-fitted inside every CV fold."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from mlkit.core import BaseStep, Pipeline, TaskConfig
from mlkit.modelling import ModelArena
from mlkit.preprocessing import MissingImputer
from mlkit.selection import ModelImportanceSelector

rng = np.random.default_rng(11)


class SpyStep(BaseStep):
    """Records the values of ``col`` it is fitted on / transforms (class-level log)."""

    log: ClassVar[list[tuple[str, frozenset]]] = []

    def __init__(self, col: str = "g") -> None:
        super().__init__()
        self.col = col

    def _fit(self, X: pd.DataFrame, y: Any = None, cfg: TaskConfig | None = None) -> SpyStep:
        SpyStep.log.append(("fit", frozenset(X[self.col])))
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        SpyStep.log.append(("transform", frozenset(X[self.col])))
        return X


def _arena() -> ModelArena:
    return ModelArena(cv=4, enforce_baseline_gate=False, candidates=["linear"])


def test_selection_leakage_removed_on_noise_target():
    X = pd.DataFrame(rng.normal(size=(300, 300)), columns=[f"f{i}" for i in range(300)])
    y = pd.Series(rng.integers(0, 2, 300))
    scores = {}
    for scope in ("model", "pipeline"):
        arena = _arena()
        pipe = Pipeline(
            [("sel", ModelImportanceSelector(top_k=10)), ("arena", arena)], cv_scope=scope
        )
        pipe.fit(X, y, cfg=TaskConfig(target="y"))
        assert arena.cv_scope_ == scope
        scores[scope] = arena.cv_results_["Linear Model"]["roc_auc_mean"]
    assert scores["model"] > 0.62, "sanity: model-level CV should show the selection leak"
    assert scores["pipeline"] < 0.6


def _spy_fold_pairs(n_folds: int) -> list[tuple[frozenset, frozenset]]:
    """(fitted values, validated values) per fold from the spy log."""
    entries = SpyStep.log
    pairs = []
    i = 0
    while i < len(entries) and len(pairs) < n_folds:
        kind, fit_vals = entries[i]
        assert kind == "fit"
        # fit_transform -> fit, transform(train); then transform(valid)
        pairs.append((fit_vals, entries[i + 2][1]))
        i += 3
    return pairs


def test_group_folds_are_disjoint_at_pipeline_level():
    SpyStep.log = []
    X = pd.DataFrame({"g": np.repeat(np.arange(20), 10), "x": rng.normal(size=200)})
    y = (X["x"] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    Pipeline([("spy", SpyStep("g")), ("arena", _arena())]).fit(
        X, y, cfg=TaskConfig(target="y"), arena__groups=X["g"].to_numpy()
    )
    pairs = _spy_fold_pairs(4)
    assert len(pairs) == 4
    for fitted, valid in pairs:
        assert fitted and valid and not fitted & valid


def test_time_folds_validate_on_later_rows_at_pipeline_level():
    SpyStep.log = []
    n = 240
    X = pd.DataFrame({"t": rng.permutation(n), "x": rng.normal(size=n)})
    y = (X["x"] > 0).astype(int)
    Pipeline([("spy", SpyStep("t")), ("arena", _arena())]).fit(
        X, y, cfg=TaskConfig(target="y"), arena__time_index=X["t"].to_numpy()
    )
    for fitted, valid in _spy_fold_pairs(4):
        assert max(fitted) < min(valid)


def test_original_steps_fitted_once_on_full_training_data():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0] * 30, "b": rng.normal(size=120)})
    y = pd.Series(rng.integers(0, 2, 120))
    imputer = MissingImputer()
    Pipeline([("imp", imputer), ("arena", _arena())]).fit(X, y, cfg=TaskConfig(target="y"))
    assert imputer.fill_values_["a"] == pytest.approx(X["a"].median())


def test_pipeline_without_upstream_or_non_arena_final_step():
    from mlkit.modelling import LinearTrainer

    X = pd.DataFrame({"x": rng.normal(size=80)})
    y = (X["x"] > 0).astype(int)
    arena = _arena()
    Pipeline([("arena", arena)]).fit(X, y, cfg=TaskConfig(target="y"))
    assert arena.cv_scope_ == "model"  # nothing upstream to re-fit
    Pipeline([("imp", MissingImputer()), ("m", LinearTrainer())]).fit(X, y)  # no fold data needed


def test_invalid_cv_scope():
    with pytest.raises(ValueError, match="cv_scope"):
        Pipeline([("arena", _arena())], cv_scope="everything")


def test_all_registered_steps_are_cloneable():
    from sklearn.base import clone

    import mlkit.runner  # noqa: F401  (populates the registry)
    from mlkit.core.registry import _REGISTRY

    special = {
        "features.multi_table": {
            "secondary_df": pd.DataFrame({"k": [1], "v": [2.0]}),
            "join_key": "k",
        },
        "features.point_in_time": {
            "events": pd.DataFrame({"k": [1], "t": [0.0], "v": [2.0]}),
            "key": "k",
            "as_of": "cutoff",
            "event_time": "t",
        },
    }
    for name, cls in _REGISTRY.items():
        clone(cls(**special.get(name, {})))  # raises RuntimeError if __init__ mutates params
