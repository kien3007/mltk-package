"""v0.6: forecasting (lag features, seasonal-naive baseline, backtest, seasonality EDA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlkit.core import TaskConfig
from mlkit.core.errors import ValidationError
from mlkit.timeseries import (
    autocorrelation,
    backtest,
    detect_season,
    forecast_metrics,
    make_lag_features,
    seasonal_decompose,
    seasonal_lag,
)

rng = np.random.default_rng(6)


def _panel(n_series: int = 3, n: int = 300, season: int = 7) -> pd.DataFrame:
    frames = []
    for s in range(n_series):
        t = np.arange(n)
        y = 40 + 10 * s + 0.05 * t + 8 * np.sin(2 * np.pi * t / season) + rng.normal(0, 1.5, n)
        frames.append(
            pd.DataFrame({"shop": f"s{s}", "day": pd.date_range("2024-01-01", periods=n), "qty": y})
        )
    # shuffled on purpose: features must not depend on the incoming row order
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=0)


def _cfg(**kw) -> TaskConfig:
    base = {
        "task_type": "forecasting",
        "target": "qty",
        "time_col": "day",
        "group_col": "shop",
        "cv_folds": 3,
        "forecasting": {"horizon": 7},
        "modelling": {"candidates": ["linear", "lightgbm"]},
    }
    base.update(kw)
    return TaskConfig(**base)


class TestSeasonality:
    def test_detect_and_acf(self):
        y = np.sin(2 * np.pi * np.arange(240) / 12) + rng.normal(0, 0.1, 240)
        assert detect_season(y) == 12
        acf = autocorrelation(y, 24)
        assert acf[0] == 1 and acf[12] > 0.8 and acf[6] < -0.8
        assert detect_season(rng.normal(size=300)) is None

    def test_decompose_recovers_components(self):
        t = np.arange(120)
        y = 0.5 * t + 5 * np.sin(2 * np.pi * t / 12)
        d = seasonal_decompose(y, 12)
        assert d["resid"].abs().max() < 1e-9 and d["trend"].isna().sum() == 12
        with pytest.raises(ValidationError):
            seasonal_decompose(y[:20], 12)

    def test_charts(self):
        from mlkit.viz import plot_acf, plot_decomposition, plot_forecast

        y = np.sin(np.arange(100) * 2 * np.pi / 10)
        for fig in (
            plot_acf(y),
            plot_decomposition(y),
            plot_forecast(np.arange(100), y, y + 0.1),
            plot_acf([1.0]),
            plot_decomposition(np.ones(4)),
        ):
            assert fig.axes


class TestLagFeatures:
    def test_no_leakage_and_original_order(self):
        df = _panel()
        out, opts = make_lag_features(df, "qty", "day", "shop", horizon=7)
        assert opts["season"] == 7 and min(opts["lags"]) == 7
        assert list(out.index) == list(df.index)
        one = out[out.shop == "s1"].sort_values("day")
        # lag k is exactly the value k periods earlier in the same series
        assert np.allclose(one["qty__lag7"].iloc[7:].to_numpy(), one["qty"].iloc[:-7].to_numpy())
        # rolling stats only use values at t - horizon or earlier
        w = opts["windows"][0]
        expect = one["qty"].shift(7).rolling(w, min_periods=1).mean()
        assert np.allclose(one[f"qty__roll{w}_mean"], expect, equal_nan=True)

    def test_lags_below_horizon_rejected(self):
        with pytest.raises(ValidationError, match="horizon"):
            make_lag_features(_panel(1), "qty", "day", horizon=7, lags=[1, 7])

    def test_seasonal_lag(self):
        assert (
            seasonal_lag(3, 12) == 12 and seasonal_lag(14, 7) == 14 and seasonal_lag(5, None) == 5
        )

    def test_metrics(self):
        m = forecast_metrics([1, 2, 3], [1, 2, 5], scale=2.0)
        assert m["mae"] == pytest.approx(2 / 3) and m["mase"] == pytest.approx(1 / 3)


class TestForecastingTask:
    def test_config_alias(self):
        cfg = _cfg()
        assert cfg.task_type == "regression" and cfg.is_forecasting
        assert TaskConfig(task_type="forecasting", time_col="t").forecasting == {"horizon": 1}
        with pytest.raises(ValidationError, match="time_col"):
            from mlkit.runner import train

            train(TaskConfig(task_type="forecasting", target="qty"), _panel(1))

    def test_train_beats_naive_and_forecasts_future(self, tmp_path):
        from mlkit.runner import predict_bundle, train

        df = _panel()
        res = train(_cfg(), df, output_dir=tmp_path / "b")
        m = res.metrics
        assert m["beats_seasonal_naive"] and m["holdout_forecast"]["mase"] < 1
        assert (
            m["forecasting"]["season"] == 7 and "qty__lag7" in m["forecasting"]["feature_columns"]
        )

        last = df["day"].max()
        future = pd.DataFrame(
            [
                {"shop": s, "day": last + pd.Timedelta(days=k), "qty": np.nan}
                for s in ("s0", "s1", "s2")
                for k in range(1, 8)
            ]
        )
        out = predict_bundle(tmp_path / "b", pd.concat([df, future], ignore_index=True))
        fut = out[out["day"] > last]
        assert len(fut) == 21 and fut["prediction"].notna().all()
        with pytest.raises(ValidationError, match="history"):
            predict_bundle(tmp_path / "b", future.drop(columns="qty"))

    def test_backtest(self):
        bt = backtest(_cfg(), _panel(2, 260), n_windows=2)
        assert len(bt) == 2 and bt["beats_naive"].all()
        assert (bt["cutoff"].diff().dropna() > pd.Timedelta(0)).all()

    def test_lag_features_not_flagged_as_leakage(self):
        from mlkit.runner import train

        res = train(_cfg(modelling={"candidates": ["linear"]}), _panel(1, 200))
        assert res.metrics["champion"]
