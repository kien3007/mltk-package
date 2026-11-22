"""Type-aware EDA charts (matplotlib) for :class:`mlkit.qc_eda.eda.EDAReport`.

Design rules (validated reference palette):
- one hue per chart for a single series (blue), recessive hairline grid and muted axes;
- magnitude heatmaps use a single-hue light->dark blue ramp; signed differences use the
  blue <-> red diverging pair around a neutral midpoint;
- never a dual y-axis: "target rate" and "row count" are two stacked panels sharing x;
- every chart has a title naming the measure; numbers are labelled selectively.
Every function returns a ``matplotlib.figure.Figure`` and never calls ``plt.show()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

if TYPE_CHECKING:
    from mlkit.qc_eda.eda import EDAReport
    from mlkit.qc_eda.profile import DatasetProfile

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"  # categorical slot 1 (blue)
POS = "#e34948"  # diverging warm pole (red)
NEG = "#2a78d6"  # diverging cool pole (blue)
NEUTRAL = "#f0efec"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "mlkit_seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style(
    ax: plt.Axes,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid_axis: Literal["x", "y", "both", ""] = "y",
) -> None:
    ax.set_facecolor(SURFACE)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _figure(
    w: float, h: float, nrows: int = 1, ncols: int = 1, **kw: Any
) -> tuple[plt.Figure, Any]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _short(label: Any, n: int = 22) -> str:
    s = str(label)
    return s if len(s) <= n else s[: n - 1] + "…"


def _empty(ax: plt.Axes, msg: str) -> None:
    ax.text(
        0.5, 0.5, msg, ha="center", va="center", color=MUTED, fontsize=9, transform=ax.transAxes
    )
    ax.set_xticks([])
    ax.set_yticks([])


# --------------------------------------------------------------------------- overview
def plot_column_overview(profile: DatasetProfile, max_columns: int = 40) -> plt.Figure:
    """Left: number of columns per semantic type. Right: missing rate per column (worst first)."""
    types = pd.Series({c: p.semantic_type for c, p in profile.columns.items()}).value_counts()
    miss = pd.Series({c: p.missing_rate for c, p in profile.columns.items()}).sort_values(
        ascending=False
    )
    miss = miss[miss > 0].head(max_columns)

    fig, (a1, a2) = _figure(12, max(3.0, 0.28 * max(len(types), len(miss), 6)), 1, 2)
    a1.barh(
        types.index[::-1],
        types.values[::-1],
        color=SERIES,
        height=0.6,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    for i, v in enumerate(types.values[::-1]):
        a1.text(v, i, f" {v}", va="center", fontsize=8, color=INK_2)
    _style(a1, "Columns by semantic type", "columns", grid_axis="x")

    if miss.empty:
        _empty(a2, "No missing values")
        _style(a2, "Missing rate per column", grid_axis="")
    else:
        a2.barh(
            [_short(c) for c in miss.index[::-1]],
            miss.values[::-1],
            color=SERIES,
            height=0.6,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        a2.set_xlim(0, 1)
        a2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for i, v in enumerate(miss.values[::-1]):
            a2.text(v, i, f" {v:.0%}", va="center", fontsize=8, color=INK_2)
        _style(
            a2, f"Missing rate per column (top {len(miss)})", "share of rows missing", grid_axis="x"
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- distributions
def _plot_one(ax: plt.Axes, s: pd.Series, sem: str, name: str) -> None:
    nn = s.dropna()
    title = f"{_short(name, 28)}  ·  {sem}"
    if nn.empty:
        _empty(ax, "all missing")
        _style(ax, title, grid_axis="")
        return
    if sem == "datetime":
        dt = pd.to_datetime(nn, unit="s")
        ax.hist(dt, bins=30, color=SERIES, edgecolor=SURFACE, linewidth=1)
        ax.tick_params(axis="x", rotation=30)
        _style(ax, title, ylabel="rows")
    elif pd.api.types.is_numeric_dtype(nn) and sem not in (
        "categorical",
        "high_cardinality",
        "boolean",
        "binary",
    ):
        lo, hi = np.nanpercentile(nn, [1, 99])
        clipped = nn.clip(lo, hi)
        arr = nn.to_numpy(dtype=float)
        log = bool(arr.min() > 0 and pd.Series(arr).skew() > 2 and hi / max(lo, 1e-12) > 100)
        ax.hist(
            np.log10(clipped) if log else clipped,
            bins=30,
            color=SERIES,
            edgecolor=SURFACE,
            linewidth=1,
        )
        _style(
            ax,
            title + ("  (log10)" if log else ""),
            "length (chars)" if sem == "text" else "",
            "rows",
        )
    else:
        counts = nn.astype(str).value_counts()
        top = counts.head(10)
        if len(counts) > 10:
            top = pd.concat([top, pd.Series({"(other)": counts.iloc[10:].sum()})])
        ax.barh(
            [_short(i, 18) for i in top.index[::-1]],
            top.values[::-1],
            color=SERIES,
            height=0.65,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        _style(
            ax,
            title + (f"  ({len(counts)} levels)" if len(counts) > 10 else ""),
            "rows",
            grid_axis="x",
        )


def plot_distributions(
    analysed: pd.DataFrame,
    profile: DatasetProfile,
    columns: list[str] | None = None,
    max_columns: int = 24,
) -> plt.Figure:
    """Small multiples: one panel per column, chart type chosen by semantic type.

    numeric -> histogram (1-99th percentile, log10 when heavily right-skewed), datetime ->
    histogram over time, text -> length histogram, categorical/boolean -> top-10 bars + other.
    """
    cols = [c for c in (columns or list(analysed.columns)) if c in analysed.columns][:max_columns]
    if not cols:
        fig, ax = _figure(6, 2)
        _empty(ax, "No analysable columns")
        return fig
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = _figure(13, 2.9 * nrows, nrows, ncols, squeeze=False)
    for ax, c in zip(axes.ravel(), cols, strict=False):
        _plot_one(
            ax, analysed[c], profile.columns[c].semantic_type if c in profile.columns else "", c
        )
    for ax in axes.ravel()[len(cols) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- target
def plot_target_association(report: EDAReport, top_n: int = 25) -> plt.Figure:
    """Features ranked by association with the target (0 = none, 1 = determines it)."""
    ta = report.target_association.dropna(subset=["strength"]).head(top_n)
    fig, ax = _figure(9, max(2.5, 0.3 * len(ta) + 1))
    if ta.empty:
        _empty(ax, "No associations computed")
        return fig
    labels = [f"{_short(f, 26)}  ({m})" for f, m in zip(ta["feature"], ta["method"], strict=False)]
    ax.barh(
        labels[::-1],
        ta["strength"].to_numpy()[::-1],
        color=SERIES,
        height=0.6,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    for i, v in enumerate(ta["strength"].to_numpy()[::-1]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, 1.05)
    _style(
        ax,
        f"Association with target '{report.target}' ({report.target_kind})",
        "strength [0, 1]",
        grid_axis="x",
    )
    fig.tight_layout()
    return fig


def plot_target_by_feature(report: EDAReport, column: str) -> plt.Figure:
    """Target rate/mean per bin or category (top) and row count per bin (bottom), shared x."""
    g = report.target_by_feature[column]
    value_col = g.columns[2]
    fig, (top, bottom) = _figure(9, 4.6, 2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(len(g))
    top.bar(x, g[value_col], color=SERIES, width=0.7, edgecolor=SURFACE, linewidth=1.5)
    overall = report.target_by_feature[column][value_col].mul(g["n"]).sum() / max(g["n"].sum(), 1)
    top.axhline(overall, color=MUTED, linewidth=1, linestyle="--")
    top.text(
        len(g) - 0.5,
        overall,
        f"overall {overall:.3g}",
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK_2,
    )
    _style(top, f"{value_col} by {_short(column, 40)}", ylabel=value_col)
    bottom.bar(x, g["n"], color=MUTED, width=0.7, edgecolor=SURFACE, linewidth=1.5)
    _style(bottom, ylabel="rows")
    bottom.set_xticks(x)
    bottom.set_xticklabels([_short(b, 16) for b in g["bucket"]], rotation=40, ha="right")
    fig.tight_layout()
    return fig


def plot_missingness(report: EDAReport, top_n: int = 25) -> plt.Figure:
    """Missing rate per column; with a target, also how the target differs when a value is missing."""
    m = report.missingness.head(top_n)
    has_diff = "difference" in m.columns
    fig, axes = _figure(
        12 if has_diff else 7, max(2.5, 0.3 * len(m) + 1), 1, 2 if has_diff else 1, squeeze=False
    )
    labels = [_short(f, 26) for f in m["feature"]][::-1]
    a = axes[0, 0]
    a.barh(
        labels,
        m["missing_rate"].to_numpy()[::-1],
        color=SERIES,
        height=0.6,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    a.set_xlim(0, 1)
    a.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _style(a, "Missing rate", "share of rows", grid_axis="x")
    if has_diff:
        b = axes[0, 1]
        d = m["difference"].to_numpy()[::-1]
        b.barh(
            labels,
            d,
            color=[POS if v > 0 else NEG for v in d],
            height=0.6,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        b.axvline(0, color=AXIS, linewidth=1)
        for i, v in enumerate(d):
            b.text(
                v,
                i,
                f" {v:+.3g} ",
                va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8,
                color=INK_2,
            )
        _style(
            b,
            "Target when missing minus when present",
            "difference (red = higher when missing)",
            grid_axis="x",
        )
        b.set_yticklabels([])
    fig.tight_layout()
    return fig


def plot_association_matrix(report: EDAReport) -> plt.Figure:
    """Mixed-type association heatmap (|Spearman| / eta / Cramer's V, all in [0, 1])."""
    mat = report.associations
    n = len(mat)
    fig, ax = _figure(max(6, 0.38 * n + 3), max(5, 0.38 * n + 2))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_short(c, 18) for c in mat.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels([_short(c, 18) for c in mat.index], fontsize=7)
    if n <= 15:
        vals = mat.to_numpy(dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(vals[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{vals[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="#ffffff" if vals[i, j] > 0.55 else INK,
                    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    cbar.outline.set_visible(False)  # type: ignore[operator]
    _style(ax, "Feature associations (0 = independent, 1 = redundant)", grid_axis="")
    fig.tight_layout()
    return fig
