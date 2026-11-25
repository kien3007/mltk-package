"""Time-series charts: autocorrelation, classical decomposition, forecast vs actual."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlkit.timeseries import autocorrelation, detect_season, seasonal_decompose
from mlkit.viz.eda_charts import AXIS, GRID, INK_2, MUTED, SERIES, _empty, _figure, _style


def plot_acf(series: Any, max_lag: int = 40, title: str | None = None) -> plt.Figure:
    """Autocorrelation bars with the ±1.96/√n white-noise band; the detected season is marked."""
    x = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    acf = autocorrelation(x, max_lag)
    fig, ax = _figure(10, 3.2)
    if len(x) < 3 or not np.isfinite(acf[1:]).any():
        _empty(ax, "Not enough data for autocorrelation")
        _style(ax, title or "Autocorrelation", grid_axis="")
        return fig
    lags = np.arange(len(acf))
    band = 1.96 / np.sqrt(len(x))
    ax.axhspan(-band, band, color=GRID, alpha=0.7, linewidth=0)
    ax.vlines(lags[1:], 0, acf[1:], color=SERIES, linewidth=2)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    season = detect_season(x, max_lag=max_lag)
    if season:
        ax.annotate(
            f"season ≈ {season}",
            (season, acf[season]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    _style(ax, title or "Autocorrelation by lag", "lag", "ACF")
    fig.tight_layout()
    return fig


def plot_decomposition(series: Any, period: int | None = None) -> plt.Figure:
    """Observed / trend / seasonal / residual as four stacked panels sharing the x axis."""
    x = pd.to_numeric(pd.Series(series), errors="coerce").reset_index(drop=True)
    period = period or detect_season(x.dropna()) or 0
    if period < 2 or x.notna().sum() < 2 * period:
        fig, ax = _figure(10, 3)
        _empty(ax, "No seasonal period found (pass period=...)")
        _style(ax, "Seasonal decomposition", grid_axis="")
        return fig
    dec = seasonal_decompose(x, period)
    fig, axes = _figure(10, 7, 4, 1, sharex=True)
    for ax, col in zip(axes, ["observed", "trend", "seasonal", "resid"], strict=True):
        ax.plot(dec.index, dec[col], color=SERIES if col != "resid" else MUTED, linewidth=1.2)
        _style(ax, f"{col.capitalize()}" + (f" (period {period})" if col == "seasonal" else ""))
    axes[-1].set_xlabel("period index", fontsize=9, color=INK_2)
    fig.tight_layout()
    return fig


def plot_forecast(
    time: Any, actual: Any, forecast: Any, title: str = "Forecast vs actual"
) -> plt.Figure:
    """Actual (dark) and forecast (blue) over time — one series per line, direct labels."""
    d = pd.DataFrame({"t": pd.Series(time).to_numpy(), "a": actual, "f": forecast}).sort_values("t")
    fig, ax = _figure(10, 3.6)
    ax.plot(d["t"], d["a"], color=INK_2, linewidth=1.3)
    ax.plot(d["t"], d["f"], color=SERIES, linewidth=1.6)
    for col, label, color in (("a", "actual", INK_2), ("f", "forecast", SERIES)):
        last = d[col].dropna()
        if len(last):
            ax.annotate(
                label,
                (d["t"].loc[last.index[-1]], last.iloc[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
            )
    _style(ax, title, "", "value")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
