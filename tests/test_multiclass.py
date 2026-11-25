"""v0.3: multiclass classification end to end, new models (HistGB, ExtraTrees), new metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.base import clone

from mlkit.core import TaskConfig
from mlkit.modelling import (
    ExtraTreesTrainer,
    HistGBTrainer,
    LGBMTrainer,
    ModelArena,
    compute_metrics,
)


def _iris() -> pd.DataFrame:
    df = datasets.load_iris(as_frame=True).frame
    df["target"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})
    return df


class TestMulticlassMetrics:
    def test_keys_and_extremes(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 600)
        perfect = np.eye(3)[y] * 0.98 + 0.01 / 1  # near-certain, rows still ~sum to 1
        m = compute_metrics(y, y, perfect)
        assert {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "log_loss",
            "brier",
            "ece",
        } <= set(m)
        assert m["roc_auc"] == pytest.approx(1.0) and m["accuracy"] == 1.0 and m["log_loss"] < 0.05
        noise = rng.dirichlet([1, 1, 1], 600)
        assert compute_metrics(y, noise.argmax(1), noise)["roc_auc"] == pytest.approx(0.5, abs=0.06)

    def test_binary_gets_pr_auc_and_log_loss(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 300)
        p = np.clip(y * 0.6 + rng.uniform(size=300) * 0.4, 0, 1)
        m = compute_metrics(y, (p > 0.5).astype(int), np.column_stack([1 - p, p]))
        assert 0.5 < m["pr_auc"] <= 1.0 and m["log_loss"] > 0


class TestMulticlassEndToEnd:
    @pytest.fixture(scope="class")
    def bundle(self, tmp_path_factory):
        from mlkit.runner import train

        out = tmp_path_factory.mktemp("mc") / "iris"
        cfg = TaskConfig(
            target="target",
            cv_folds=3,
            modelling={"candidates": ["linear", "lightgbm"]},
            metric_thresholds={"roc_auc": 0.9, "log_loss": 0.5},
        )
        result = train(cfg, _iris(), output_dir=out)
        return result, out

    def test_training_metrics(self, bundle):
        result, _ = bundle
        assert result.metrics["target"]["label_map"] == {
            "setosa": 0,
            "versicolor": 1,
            "virginica": 2,
        }
        assert result.metrics["holdout"]["roc_auc"] > 0.95
        assert result.metrics["thresholds_passed"] is True

    def test_predict_returns_one_score_per_class(self, bundle):
        from mlkit.runner import predict_bundle

        _, out = bundle
        preds = predict_bundle(out, _iris().drop(columns="target"))
        score_cols = ["score_setosa", "score_versicolor", "score_virginica"]
        assert set(score_cols) <= set(preds.columns)
        np.testing.assert_allclose(preds[score_cols].sum(axis=1), 1.0, atol=1e-6)
        assert (preds["label"] == _iris()["target"]).mean() > 0.9
        np.testing.assert_allclose(preds["score"], preds[score_cols].max(axis=1))

    def test_evaluate_and_monitor(self, bundle):
        from mlkit.runner import evaluate_bundle, monitor_bundle

        _, out = bundle
        ev = evaluate_bundle(out, _iris(), by="target")
        assert ev.overall["accuracy"] > 0.9 and "mean_confidence" in ev.overall
        assert list(ev.by_group["target"]) == ["setosa", "versicolor", "virginica"]
        assert monitor_bundle(out, _iris().drop(columns="target")).is_passed

    def test_unknown_label_rejected_in_evaluation(self, bundle):
        from mlkit.core.errors import ValidationError
        from mlkit.runner import evaluate_bundle

        _, out = bundle
        bad = _iris().assign(target="unknown")
        with pytest.raises(ValidationError, match="not seen"):
            evaluate_bundle(out, bad)


class TestArenaMulticlass:
    def _data(self):
        X, y = datasets.load_wine(return_X_y=True, as_frame=True)
        return X, y.to_numpy()

    def test_threshold_ignored_with_warning_and_calibration_works(self):
        X, y = self._data()
        arena = ModelArena(
            cv=3,
            candidates=["linear"],
            threshold="f1",
            calibration="auto",
            enforce_baseline_gate=False,
        )
        arena.fit(X, y, cfg=TaskConfig(target="y"))
        assert arena.decision_threshold_ == 0.5
        assert any(
            i.name == "decision_threshold" and i.status == "warn" for i in arena.report_.items
        )
        assert arena.predict_proba(X).shape == (len(X), 3)
        assert set(arena.predict(X)) <= {0, 1, 2}

    @pytest.mark.parametrize("cls", [HistGBTrainer, ExtraTreesTrainer, LGBMTrainer])
    def test_new_and_lightgbm_trainers_handle_multiclass_binary_regression(self, cls):
        X, y = self._data()
        mc = clone(cls()).fit(X, y)
        assert mc.predict_proba(X).shape == (len(X), 3)
        yb = (y == 0).astype(int)
        assert clone(cls()).fit(X, yb).predict_proba(X).shape == (len(X), 2)
        reg = cls(task_type="regression").fit(X, X.iloc[:, 0])
        assert reg.predict(X).shape == (len(X),)
        assert len(mc.feature_importances_) == X.shape[1]

    def test_new_candidates_in_arena(self):
        X, y = self._data()
        arena = ModelArena(cv=3, candidates=["hist_gb", "extra_trees"], enforce_baseline_gate=False)
        arena.fit(X, y, cfg=TaskConfig(target="y"))
        assert set(arena.cv_results_) == {"Baseline (Dummy)", "HistGradientBoosting", "Extra Trees"}


def test_lower_is_better_log_loss_gate():
    from mlkit.runner import train

    cfg = TaskConfig(
        target="target",
        cv_folds=3,
        modelling={"candidates": ["linear"]},
        metric_thresholds={"log_loss": 0.0001},
    )
    res = train(cfg, _iris())
    assert res.metrics["thresholds_passed"] is False
    assert any("log_loss" in f and "must be <=" in f for f in res.metrics["threshold_failures"])
