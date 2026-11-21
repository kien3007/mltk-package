"""Tests for mlkit.viz and mlkit.modelling.tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
N = 200

# Numeric-only X for arena/model tests (models need encoded features)
X_num = pd.DataFrame(
    {
        "age": rng.integers(20, 70, size=N).astype(float),
        "income": rng.normal(50_000, 15_000, size=N),
        "score": rng.uniform(0, 1, size=N),
        "region_enc": rng.integers(0, 3, size=N).astype(float),
    }
)
y_cls = pd.Series(rng.choice([0, 1], size=N))

# X_cls with categorical column (for missing/correlation viz tests only)
X_cls = pd.DataFrame(
    {
        "age": rng.integers(20, 70, size=N).astype(float),
        "income": rng.normal(50_000, 15_000, size=N),
        "score": rng.uniform(0, 1, size=N),
        "region": rng.choice(["A", "B", "C"], size=N),
    }
)

X_reg = pd.DataFrame(
    {
        "size_sqft": rng.uniform(500, 3000, size=N),
        "rooms": rng.integers(1, 6, size=N).astype(float),
        "age_years": rng.integers(0, 50, size=N).astype(float),
    }
)
y_reg = pd.Series(rng.uniform(100_000, 800_000, size=N))


# ===========================================================================
# mlkit.viz tests
# ===========================================================================

class TestVizMissingRates:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_missing_rates

        df = X_cls.copy()
        df.loc[:10, "income"] = np.nan  # inject some missing
        fig = plot_missing_rates(df)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_no_missing(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_missing_rates

        fig = plot_missing_rates(X_cls.select_dtypes(include=[np.number]))
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizTargetDistribution:
    def test_classification(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_target_distribution

        fig = plot_target_distribution(y_cls, task_type="classification")
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_regression(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_target_distribution

        fig = plot_target_distribution(y_reg, task_type="regression")
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizCorrelationHeatmap:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_correlation_heatmap

        fig = plot_correlation_heatmap(X_reg)
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizArenaComparison:
    def test_from_dataframe(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_arena_comparison

        df = pd.DataFrame({
            "Model": ["Baseline (Dummy)", "Linear Model", "LightGBM"],
            "Score": [0.5, 0.72, 0.85],
        })
        fig = plot_arena_comparison(df)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_from_fitted_arena(self):
        import matplotlib.pyplot as plt

        from mlkit.core.config import TaskConfig
        from mlkit.modelling import ModelArena
        from mlkit.viz import plot_arena_comparison

        cfg = TaskConfig(task_type="classification", target="label")
        arena = ModelArena(cv=3, random_state=0)
        arena.fit(X_num, y=y_cls, cfg=cfg)  # numeric-only X
        fig = plot_arena_comparison(arena)
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizFeatureImportance:
    def test_from_arena(self):
        import matplotlib.pyplot as plt

        from mlkit.core.config import TaskConfig
        from mlkit.modelling import ModelArena
        from mlkit.viz import plot_feature_importance

        cfg = TaskConfig(task_type="classification", target="label")
        arena = ModelArena(cv=3, random_state=0)
        arena.fit(X_num, y=y_cls, cfg=cfg)  # numeric-only X
        fig = plot_feature_importance(arena, feature_names=list(X_num.columns))
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizDriftPSI:
    def test_with_drifted_data(self):
        import matplotlib.pyplot as plt

        from mlkit.monitoring import DriftMonitor
        from mlkit.viz import plot_drift_psi

        monitor = DriftMonitor()
        monitor.fit(X_reg)

        X_drifted = X_reg.copy()
        X_drifted["size_sqft"] = X_drifted["size_sqft"] * 5  # inject drift
        report = monitor.monitor(X_drifted)

        fig = plot_drift_psi(report)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_empty_report_no_crash(self):
        import matplotlib.pyplot as plt

        from mlkit.core.report import Report
        from mlkit.viz import plot_drift_psi

        empty_report = Report("empty")
        fig = plot_drift_psi(empty_report)
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizConfusionMatrix:
    def test_normalized(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_confusion_matrix

        fig = plot_confusion_matrix(y_cls, y_cls)  # perfect prediction
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_raw_counts(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_confusion_matrix

        y_pred = pd.Series(rng.choice([0, 1], size=N))
        fig = plot_confusion_matrix(y_cls, y_pred, normalize=False)
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestVizRocCurve:
    def test_returns_figure(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_roc_curve

        y_prob = rng.uniform(0, 1, size=N)
        fig = plot_roc_curve(y_cls, y_prob)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_2d_proba(self):
        import matplotlib.pyplot as plt

        from mlkit.viz import plot_roc_curve

        y_prob_2d = np.column_stack([1 - rng.uniform(0, 1, N), rng.uniform(0, 1, N)])
        fig = plot_roc_curve(y_cls, y_prob_2d)
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ===========================================================================
# mlkit.modelling.tuning tests
# ===========================================================================

class TestOptunaSearch:
    def test_basic_classification(self):
        from mlkit.core.config import TaskConfig
        from mlkit.modelling.trainers import LGBMTrainer
        from mlkit.modelling.tuning import LGBM_SPACE, OptunaSearch

        cfg = TaskConfig(task_type="classification", target="label")
        searcher = OptunaSearch(
            estimator_cls=LGBMTrainer,
            space=LGBM_SPACE,
            n_trials=5,
            cv=3,
            random_state=0,
        )
        searcher.fit(X_num, y=y_cls, cfg=cfg)  # numeric-only X
        assert isinstance(searcher.best_params_, dict)
        assert searcher.best_score_ > 0.0
        preds = searcher.predict(X_num)
        assert len(preds) == N

    def test_basic_regression(self):
        from mlkit.core.config import TaskConfig
        from mlkit.modelling.trainers import RandomForestTrainer
        from mlkit.modelling.tuning import RF_SPACE, OptunaSearch

        cfg = TaskConfig(task_type="regression", target="price")
        searcher = OptunaSearch(
            estimator_cls=RandomForestTrainer,
            space=RF_SPACE,
            n_trials=5,
            cv=3,
            random_state=0,
        )
        searcher.fit(X_reg, y=y_reg, cfg=cfg)
        assert isinstance(searcher.best_params_, dict)
        assert not np.isnan(searcher.best_score_)
        preds = searcher.predict(X_reg)
        assert len(preds) == N

    def test_trials_dataframe(self):
        from mlkit.core.config import TaskConfig
        from mlkit.modelling.trainers import LinearTrainer
        from mlkit.modelling.tuning import LINEAR_SPACE, OptunaSearch

        cfg = TaskConfig(task_type="classification", target="label")
        searcher = OptunaSearch(
            estimator_cls=LinearTrainer,
            space=LINEAR_SPACE,
            n_trials=4,
            cv=3,
            random_state=0,
        )
        searcher.fit(X_num, y=y_cls, cfg=cfg)  # numeric-only X
        df = searcher.trials_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    def test_not_fitted_raises(self):
        from mlkit.core.errors import MLKitError
        from mlkit.modelling.trainers import LGBMTrainer
        from mlkit.modelling.tuning import LGBM_SPACE, OptunaSearch

        searcher = OptunaSearch(estimator_cls=LGBMTrainer, space=LGBM_SPACE)
        with pytest.raises(MLKitError):
            searcher.trials_dataframe()
