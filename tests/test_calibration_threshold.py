"""Calibration inside ModelArena and decision-threshold tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.modelling import ModelArena
from mlkit.modelling.evaluation import CalibratedModelWrapper, optimal_threshold

rng = np.random.default_rng(5)


def _imbalanced(n: int = 3000) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    logit = 1.3 * X["x1"] - 0.7 * X["x2"] + 1.0  # ~70% positives
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


class TestOptimalThreshold:
    def test_f1_and_youden_recover_separating_point(self):
        p = np.linspace(0.01, 0.99, 400)
        y = (p > 0.3).astype(int)  # perfectly separable at 0.3
        for crit in ("f1", "youden"):
            assert optimal_threshold(y, p, crit) == pytest.approx(0.3, abs=0.01)

    def test_degenerate_and_invalid(self):
        assert optimal_threshold(np.ones(10), np.linspace(0, 1, 10)) == 0.5
        with pytest.raises(ValueError, match="criterion"):
            optimal_threshold(np.array([0, 1]), np.array([0.2, 0.8]), "accuracy")  # type: ignore[arg-type]


class TestArenaCalibration:
    def test_calibration_kept_only_when_cv_brier_improves(self):
        X, y = _imbalanced()
        arena = ModelArena(
            cv=3,
            candidates=["linear"],
            handle_imbalance=True,
            calibration="auto",
            enforce_baseline_gate=False,
        )
        arena.fit(X, y, cfg=TaskConfig(target="y"))
        raw = arena.cv_results_["Linear Model"]["brier_mean"]
        tried = {k: v["brier_mean"] for k, v in arena.cv_results_.items() if " + " in k}
        assert set(tried) == {"Linear Model + sigmoid", "Linear Model + isotonic"}
        if arena.calibration_ is None:
            assert min(tried.values()) >= raw
        else:
            assert tried[f"Linear Model + {arena.calibration_}"] == min(tried.values()) < raw
            assert isinstance(arena.best_estimator_, CalibratedModelWrapper)

    def test_class_weighted_model_gets_calibrated_and_bias_shrinks(self):
        X, y = _imbalanced()
        tr, te = slice(0, 2200), slice(2200, None)
        bias = {}
        for calib in (None, "sigmoid"):
            arena = ModelArena(
                cv=3,
                candidates=["linear"],
                handle_imbalance=True,
                calibration=calib,
                enforce_baseline_gate=False,
            ).fit(X[tr], y[tr], cfg=TaskConfig(target="y"))
            bias[calib] = abs(arena.predict_proba(X[te])[:, 1].mean() - y[te].mean())
        assert bias["sigmoid"] < bias[None]

    def test_invalid_calibration(self):
        X, y = _imbalanced(200)
        with pytest.raises(ValueError, match="calibration"):
            ModelArena(cv=2, candidates=["linear"], calibration="platt").fit(X, y)

    def test_wrapper_exposes_classes_and_importances(self):
        from mlkit.modelling import LGBMTrainer

        X, y = _imbalanced(600)
        w = CalibratedModelWrapper(LGBMTrainer(n_estimators=20), cv=2).fit(X, y)
        assert list(w.classes_) == [0, 1]
        assert w.feature_importances_ is not None and len(w.feature_importances_) == 2


class TestDecisionThreshold:
    def test_numeric_threshold_drives_predict(self):
        X, y = _imbalanced(800)
        arena = ModelArena(
            cv=3, candidates=["linear"], threshold=0.8, enforce_baseline_gate=False
        ).fit(X, y)
        assert arena.decision_threshold_ == 0.8
        proba = arena.predict_proba(X)[:, 1]
        np.testing.assert_array_equal(arena.predict(X), (proba >= 0.8).astype(int))

    def test_tuned_threshold_is_valid_and_used(self):
        X, y = _imbalanced(1500)
        arena = ModelArena(
            cv=3, candidates=["linear"], threshold="f1", enforce_baseline_gate=False
        ).fit(X, y)
        thr = arena.decision_threshold_
        assert 0.0 < thr < 1.0
        np.testing.assert_array_equal(
            arena.predict(X), (arena.predict_proba(X)[:, 1] >= thr).astype(int)
        )
        assert any(i.name == "decision_threshold" for i in arena.report_.items)

    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, "best", True])
    def test_invalid_threshold(self, bad):
        X, y = _imbalanced(200)
        with pytest.raises(ValueError, match="threshold"):
            ModelArena(cv=2, candidates=["linear"], threshold=bad, enforce_baseline_gate=False).fit(
                X, y
            )

    def test_regression_ignores_threshold(self):
        X, _ = _imbalanced(200)
        arena = ModelArena(cv=2, candidates=["linear"], threshold="f1", enforce_baseline_gate=False)
        arena.fit(X, X["x1"] * 2, cfg=TaskConfig(task_type="regression", target="y"))
        assert arena.decision_threshold_ == 0.5


def test_bundle_threshold_is_default_for_predict(tmp_path):
    from mlkit.runner import predict_bundle, train

    X, y = _imbalanced(1200)
    df = X.assign(label=y)
    cfg = TaskConfig(
        target="label",
        cv_folds=3,
        modelling={"candidates": ["linear"], "threshold": 0.85, "calibration": "sigmoid"},
    )
    result = train(cfg, df, output_dir=tmp_path / "b")
    assert result.metrics["decision_threshold"] == 0.85
    assert result.metrics["cv_scope"] == "pipeline"
    assert result.metrics["calibration"] in (None, "sigmoid")

    default = predict_bundle(tmp_path / "b", df.head(200))
    np.testing.assert_array_equal(default["prediction"], (default["score"] >= 0.85).astype(int))
    override = predict_bundle(tmp_path / "b", df.head(200), threshold=0.2)
    np.testing.assert_array_equal(override["prediction"], (override["score"] >= 0.2).astype(int))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_bootstrap_metrics_match_sklearn(seed):
    from mlkit.modelling.evaluation import _fast_metrics, compute_metrics

    r = np.random.default_rng(seed)
    y = r.integers(0, 2, 500)
    p = np.round(r.uniform(size=500), 2)  # rounding creates ties, as in real scores
    pred = (p >= 0.5).astype(int)
    proba = np.column_stack([1 - p, p])
    fast, ref = _fast_metrics(y, pred, proba, "classification"), compute_metrics(y, pred, proba)
    assert fast is not None and fast.keys() == ref.keys()
    for k in ref:
        assert fast[k] == pytest.approx(ref[k], abs=1e-12), k

    yr = r.normal(size=300)
    pr = yr + r.normal(scale=0.3, size=300)
    fast_r = _fast_metrics(yr, pr, None, "regression")
    ref_r = compute_metrics(yr, pr, task_type="regression")
    for k in ref_r:
        assert fast_r[k] == pytest.approx(ref_r[k], abs=1e-12), k

    assert _fast_metrics(y + 1, pred + 1, None, "classification") is None  # falls back to sklearn
