"""v0.5: top-k ensembles inside ModelArena, modelling presets and the time budget."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.core.errors import ValidationError
from mlkit.modelling import ModelArena
from mlkit.modelling.presets import PRESETS, resolve_modelling

rng = np.random.default_rng(5)


def _cls(n: int = 900, k: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    z = 2 * X["x0"] - X["x1"] + 0.5 * X["x2"] * X["x3"] + rng.normal(scale=0.7, size=n)
    y = (z > 0).astype(int) if k == 2 else np.digitize(z, np.quantile(z, [1 / 3, 2 / 3]))
    return X, np.asarray(y)


class TestEnsemble:
    @pytest.mark.parametrize("kind", ["vote", "stack"])
    @pytest.mark.parametrize("k", [2, 3])
    def test_classification(self, kind, k):
        X, y = _cls(k=k)
        arena = ModelArena(
            cv=3,
            candidates=["linear", "lightgbm", "random_forest"],
            ensemble=kind,
            enforce_baseline_gate=False,
        ).fit(X, y, cfg=TaskConfig(target="y"))
        label = f"Ensemble ({kind})"
        assert label in arena.cv_results_ and len(arena.ensemble_members_) == 3
        assert "Baseline (Dummy)" not in arena.ensemble_members_
        proba = arena.predict_proba(X.head(20))
        assert proba.shape == (20, k) and np.allclose(proba.sum(axis=1), 1)

    @pytest.mark.parametrize("kind", ["vote", "stack"])
    def test_regression(self, kind):
        X, _ = _cls()
        y = 3 * X["x0"].to_numpy() + np.sin(X["x1"].to_numpy()) + rng.normal(scale=0.3, size=len(X))
        arena = ModelArena(
            cv=3, candidates=["linear", "lightgbm"], ensemble=kind, enforce_baseline_gate=False
        ).fit(X, y, cfg=TaskConfig(task_type="regression", target="y"))
        assert f"Ensemble ({kind})" in arena.cv_results_
        assert arena.predict(X.head(5)).shape == (5,)

    def test_top_k_and_validation(self):
        X, y = _cls(500)
        arena = ModelArena(
            cv=3,
            candidates=["linear", "lightgbm", "random_forest"],
            ensemble="vote",
            ensemble_top_k=2,
            enforce_baseline_gate=False,
        ).fit(X, y, cfg=TaskConfig(target="y"))
        assert len(arena.ensemble_members_) == 2
        with pytest.raises(ValueError, match="ensemble"):
            ModelArena(cv=3, candidates=["linear"], ensemble="blend").fit(
                X, y, cfg=TaskConfig(target="y")
            )

    def test_single_candidate_skips_ensemble(self):
        X, y = _cls(400)
        arena = ModelArena(
            cv=3, candidates=["linear"], ensemble="stack", enforce_baseline_gate=False
        ).fit(X, y, cfg=TaskConfig(target="y"))
        assert not any(k.startswith("Ensemble") for k in arena.cv_results_)

    def test_bundle_roundtrip_and_explain(self, tmp_path):
        from mlkit.runner import predict_bundle, train

        X, y = _cls(700)
        cfg = TaskConfig(
            target="y",
            cv_folds=3,
            modelling={"candidates": ["linear", "lightgbm"], "ensemble": "vote"},
        )
        res = train(cfg, X.assign(y=y), output_dir=tmp_path / "b")
        assert "Ensemble (vote)" in {r["Model"] for r in res.metrics["arena"]}
        out = predict_bundle(tmp_path / "b", X.head(8), explain=2)
        assert len(out) == 8 and {"score", "reason_1"} <= set(out.columns)


class TestPresets:
    def test_resolution_and_precedence(self):
        assert resolve_modelling(None) == {}
        assert resolve_modelling({"candidates": ["linear"]}) == {"candidates": ["linear"]}
        fast = resolve_modelling({"preset": "fast"})
        assert fast["candidates"] == ["linear", "lightgbm"] and fast["preset"] == "fast"
        mine = resolve_modelling(
            {"preset": "best_quality", "ensemble": "vote", "candidates": ["linear"]}
        )
        assert mine["ensemble"] == "vote" and mine["candidates"] == ["linear"]
        assert mine["tuning"] == {"n_trials": 30}
        # presets are copied, never mutated
        mine["tuning"]["n_trials"] = 1
        assert PRESETS["best_quality"]["tuning"] == {"n_trials": 30}

    def test_unknown_preset(self):
        with pytest.raises(ValidationError, match="preset"):
            resolve_modelling({"preset": "turbo"})

    def test_preset_cv_folds_unless_explicit(self):
        from mlkit.runner import build_default_pipeline

        def arena_cv(cfg: TaskConfig) -> int:
            return build_default_pipeline(cfg).named_steps["modelling_arena"].cv

        assert arena_cv(TaskConfig(target="y", modelling={"preset": "fast"})) == 3
        assert arena_cv(TaskConfig(target="y", cv_folds=7, modelling={"preset": "fast"})) == 7

    def test_balanced_preset_trains_an_ensemble(self):
        from mlkit.runner import train

        X, y = _cls(600)
        res = train(TaskConfig(target="y", modelling={"preset": "balanced"}), X.assign(y=y))
        assert "Ensemble (vote)" in {r["Model"] for r in res.metrics["arena"]}


class TestTimeBudget:
    def test_budget_skips_candidates(self):
        X, y = _cls(3000)
        t = time.perf_counter()
        arena = ModelArena(
            cv=3,
            candidates=["linear", "random_forest", "extra_trees", "hist_gb", "lightgbm"],
            time_budget=0.01,
            ensemble="vote",
            enforce_baseline_gate=False,
        ).fit(X, y, cfg=TaskConfig(target="y"))
        assert arena.skipped_, "nothing skipped although the budget was exhausted"
        assert not any(k.startswith("Ensemble") for k in arena.cv_results_)
        assert "time_budget" in {c.name for c in arena.report_.items}
        assert time.perf_counter() - t < 60

    def test_no_budget_skips_nothing(self):
        X, y = _cls(400)
        arena = ModelArena(
            cv=3, candidates=["linear", "lightgbm"], enforce_baseline_gate=False
        ).fit(X, y, cfg=TaskConfig(target="y"))
        assert arena.skipped_ == []
