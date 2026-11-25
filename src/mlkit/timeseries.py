"""Time-series forecasting on top of the tabular pipeline (ADR-010).

A forecasting task is a regression task whose rows are periods of one or more series
(``time_col``, optional ``group_col`` = series id). ``make_lag_features`` adds, for every row
``t``, features computed only from target values at ``t - horizon`` or earlier, so a model
trained on them forecasts ``horizon`` steps ahead (direct strategy). Everything downstream
(QC, preprocessing, ModelArena with chronological CV, bundles, monitoring) is unchanged.

Configuration::

    task_type: forecasting          # = regression + forecasting defaults
    target: sales
    time_col: date
    group_col: store                # optional: several series, one global model
    forecasting:
      horizon: 7                    # steps ahead
      season: 7                     # seasonal period (auto-detected from the ACF when omitted)
      lags: [7, 8, 14, 21, 28]      # optional, default derived from horizon / season
      windows: [7, 28]              # rolling mean/std windows, default derived from season

Rows must be regularly spaced within each series (one row per period); gaps are not filled.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from mlkit.core.errors import ValidationError

FORECAST_KEYS = {"horizon", "season", "lags", "windows", "max_season"}


# ------------------------------------------------------------------ seasonality / EDA
def autocorrelation(series: Any, max_lag: int = 40) -> np.ndarray:
    """Sample autocorrelation for lags ``0..max_lag`` (NaNs are ignored pairwise)."""
    x = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float)
    x = x - np.nanmean(x)
    n = len(x)
    denom = np.nansum(x * x)
    out = np.full(max_lag + 1, np.nan)
    if n < 3 or denom == 0:
        return out
    out[0] = 1.0
    for k in range(1, min(max_lag, n - 2) + 1):
        out[k] = np.nansum(x[k:] * x[:-k]) / denom
    return out


def detect_season(series: Any, max_lag: int = 60, min_acf: float = 0.2) -> int | None:
    """Most likely seasonal period: the highest local ACF peak (lag >= 2) above ``min_acf``."""
    acf = autocorrelation(series, max_lag)
    best, best_val = None, min_acf
    for k in range(2, len(acf) - 1):
        v = acf[k]
        if np.isfinite(v) and v > best_val and v >= acf[k - 1] and v >= acf[k + 1]:
            best, best_val = k, v
    return best


def seasonal_decompose(series: Any, period: int) -> pd.DataFrame:
    """Classical additive decomposition: centred moving-average trend, mean seasonal profile."""
    y = pd.to_numeric(pd.Series(series).reset_index(drop=True), errors="coerce").astype(float)
    if period < 2 or len(y) < 2 * period:
        raise ValidationError(f"Need at least 2 full periods ({2 * period} rows) to decompose.")
    trend = y.rolling(period, center=True, min_periods=period).mean()
    if period % 2 == 0:  # 2 x m moving average keeps the trend centred for even periods
        trend = trend.rolling(2).mean().shift(-1)
    detr = y - trend
    phase = np.arange(len(y)) % period
    profile = detr.groupby(phase).mean()
    profile = profile - profile.mean()
    seasonal = pd.Series(profile.reindex(phase).to_numpy(), index=y.index)
    return pd.DataFrame(
        {"observed": y, "trend": trend, "seasonal": seasonal, "resid": y - trend - seasonal}
    )


# ------------------------------------------------------------------ configuration
def forecast_options(cfg: Any) -> dict[str, Any] | None:
    """Normalised ``cfg.forecasting`` (``None`` when the task is not a forecasting task)."""
    fc = getattr(cfg, "forecasting", None)
    if not fc:
        return None
    unknown = set(fc) - FORECAST_KEYS
    if unknown:
        raise ValidationError(
            f"Unknown forecasting keys {sorted(unknown)}; use {sorted(FORECAST_KEYS)}"
        )
    if not cfg.time_col:
        raise ValidationError("Forecasting needs time_col (the period of each row).")
    horizon = int(fc.get("horizon", 1))
    if horizon < 1:
        raise ValidationError("forecasting.horizon must be >= 1")
    season = fc.get("season")
    return {
        "horizon": horizon,
        "season": int(season) if season else None,
        "lags": [int(v) for v in fc["lags"]] if fc.get("lags") else None,
        "windows": [int(v) for v in fc["windows"]] if fc.get("windows") else None,
        "max_season": int(fc.get("max_season", 60)),
    }


def _order(df: pd.DataFrame, time_col: str, group_col: str | None) -> np.ndarray:
    t = pd.to_datetime(df[time_col], errors="coerce")
    if t.isna().all():
        t = pd.to_numeric(df[time_col], errors="coerce")
    keys = pd.DataFrame({"t": t.to_numpy()}, index=df.index)
    by = ["t"]
    if group_col:
        keys["g"] = df[group_col].astype(str).to_numpy()
        by = ["g", "t"]
    return keys.reset_index(drop=True).sort_values(by, kind="stable").index.to_numpy()


def seasonal_lag(horizon: int, season: int | None) -> int:
    """Lag used by the seasonal-naive forecast: the same phase one (or more) seasons back."""
    if not season:
        return horizon
    return season * math.ceil(horizon / season)


def resolve_lags(opts: dict[str, Any], y: pd.Series | None = None) -> dict[str, Any]:
    """Fill season (ACF), lags and windows defaults; returns the options actually used."""
    out = dict(opts)
    h = out["horizon"]
    if out["season"] is None and y is not None:
        out["season"] = detect_season(y, max_lag=out["max_season"])
    s = out["season"]
    if out["lags"] is None:
        lags = {h, h + 1, h + 2}
        if s:
            lags |= {k * s for k in (1, 2, 3, 4) if k * s >= h}
        out["lags"] = sorted(lags)
    if min(out["lags"]) < h:
        raise ValidationError(f"forecasting.lags must all be >= horizon ({h}) to avoid leakage")
    if out["windows"] is None:
        out["windows"] = sorted({w for w in (s or 7, 4 * (s or 7)) if w >= 2})
    return out


# ------------------------------------------------------------------ features
def make_lag_features(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    group_col: str | None = None,
    horizon: int = 1,
    season: int | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
    max_season: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add lag / rolling / seasonal-naive features of ``target`` (all shifted by >= horizon).

    Returns ``(frame, options)``: the frame keeps the original row order and index; ``options``
    holds the resolved season / lags / windows (store them to rebuild identical features later).
    Rows whose target is missing (future periods to forecast) still get features from history.
    """
    if target not in df.columns or time_col not in df.columns:
        raise ValidationError(f"Forecasting needs columns '{target}' and '{time_col}'.")
    if group_col and group_col not in df.columns:
        raise ValidationError(f"group_col '{group_col}' not found.")
    order = _order(df, time_col, group_col)
    d = df.iloc[order]
    y = pd.to_numeric(d[target], errors="coerce").astype(float)
    if group_col:
        first = d[group_col].astype(str).iloc[0]
        y_for_acf = y[d[group_col].astype(str).to_numpy() == first]
    else:
        y_for_acf = y
    opts = resolve_lags(
        {
            "horizon": horizon,
            "season": season,
            "lags": lags,
            "windows": windows,
            "max_season": max_season,
        },
        y_for_acf.dropna(),
    )
    keys = d[group_col].astype(str).to_numpy() if group_col else np.zeros(len(d))
    g = y.groupby(keys, sort=False)
    feats: dict[str, pd.Series] = {}
    for k in opts["lags"]:
        feats[f"{target}__lag{k}"] = g.shift(k)
    base = g.shift(horizon)
    gb = base.groupby(keys, sort=False)
    for w in opts["windows"]:
        feats[f"{target}__roll{w}_mean"] = gb.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
        feats[f"{target}__roll{w}_std"] = gb.transform(
            lambda s, w=w: s.rolling(w, min_periods=2).std()
        )
    feats[f"{target}__snaive"] = g.shift(seasonal_lag(horizon, opts["season"]))
    out = d.assign(**feats)
    out = out.iloc[np.argsort(order, kind="stable")]  # back to the caller's row order
    opts["feature_columns"] = list(feats)
    return out, opts


def snaive_column(target: str) -> str:
    return f"{target}__snaive"


# ------------------------------------------------------------------ metrics
def forecast_metrics(y_true: Any, y_pred: Any, scale: float | None = None) -> dict[str, float]:
    """MAE, RMSE, sMAPE (%) and, given ``scale`` (the in-sample seasonal-naive MAE), MASE."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[ok], yp[ok]
    err = yt - yp
    denom = np.abs(yt) + np.abs(yp)
    out = {
        "mae": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
        "smape": float(
            100 * np.mean(np.where(denom > 0, 2 * np.abs(err) / np.where(denom > 0, denom, 1), 0))
        )
        if len(err)
        else float("nan"),
    }
    if scale is not None:
        out["mase"] = (
            float(out["mae"] / scale) if scale > 0 and np.isfinite(scale) else float("nan")
        )
    return out


def prepare_forecast_frame(
    cfg: Any, df: pd.DataFrame, options: dict[str, Any] | None = None, training: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build lag features for ``cfg`` (re-using stored ``options`` at prediction time).

    For training, rows without a target or without seasonal-naive history are dropped.
    """
    opts = options or forecast_options(cfg)
    assert opts is not None
    frame, used = make_lag_features(
        df,
        cfg.target,
        cfg.time_col,
        cfg.group_col,
        horizon=opts["horizon"],
        season=opts.get("season"),
        lags=opts.get("lags"),
        windows=opts.get("windows"),
        max_season=opts.get("max_season", 60),
    )
    if training:
        keep = frame[cfg.target].notna() & frame[snaive_column(cfg.target)].notna()
        frame = frame[keep]
        if len(frame) < 30:
            raise ValidationError(
                f"Only {len(frame)} rows have enough history for horizon={opts['horizon']} "
                f"and season={used['season']}; need more periods."
            )
    return frame, used


# ------------------------------------------------------------------ backtest
def backtest(
    cfg: Any, df: pd.DataFrame, n_windows: int = 3, window: int | None = None
) -> pd.DataFrame:
    """Rolling-origin evaluation: for each cutoff, train on the past and score the next window.

    Every window reports the model and the seasonal-naive baseline on the same rows
    (``mae``, ``rmse``, ``smape``, ``mase``) plus ``beats_naive``.
    """
    from mlkit.runner import train

    opts = forecast_options(cfg)
    if opts is None:
        raise ValidationError("backtest needs a forecasting config (task_type='forecasting').")
    t = pd.to_datetime(df[cfg.time_col], errors="coerce")
    if t.isna().all():
        t = pd.to_numeric(df[cfg.time_col], errors="coerce")
    periods = np.sort(pd.unique(t.dropna()))
    window = int(window or max(opts["horizon"], len(periods) // (n_windows + 4)))
    if len(periods) < window * (n_windows + 2):
        raise ValidationError(
            f"Not enough periods ({len(periods)}) for {n_windows} windows of {window}."
        )
    rows = []
    for i in range(n_windows, 0, -1):
        cut = periods[-i * window - 1]
        end = periods[min(len(periods) - 1, len(periods) - (i - 1) * window - 1)]
        hist = df[t <= end]
        test_mask = (t[t <= end] > cut).to_numpy()
        frac = float(test_mask.mean())
        wcfg = cfg.model_copy(
            update={"test_size": min(0.5, max(0.01, frac)), "quality_gate": False}
        )
        res = train(wcfg, hist)
        m, nv = res.metrics["holdout_forecast"], res.metrics["seasonal_naive"]
        rows.append(
            {
                "cutoff": cut,
                "end": end,
                "rows": int(test_mask.sum()),
                **{f"model_{k}": v for k, v in m.items()},
                **{f"naive_{k}": v for k, v in nv.items()},
                "beats_naive": m["mae"] < nv["mae"],
                "champion": res.metrics.get("champion"),
            }
        )
    return pd.DataFrame(rows)
